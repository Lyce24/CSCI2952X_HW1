from utils.utils import LARS
import torch.optim as optim
import math
import torch
import shutil
import time
from torch.utils.tensorboard import SummaryWriter
import os
from torch.amp import GradScaler, autocast

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, lr, warmup_epochs, epochs):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_moco_momentum(epoch, moco_m, epochs):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / epochs)) * (1. - moco_m)
    return m
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
def train_one_epoch(
    model,
    train_loader,
    lr,
    warmup_epochs,
    epochs,
    moco_m,
    moco_m_cos,
    optimizer,
    scaler,
    epoch,
    device,
    summary_writer,
    print_freq
):    
    batch_time = AverageMeter('Time', ':6.3f')
    h2d_time   = AverageMeter('H2D',  ':6.3f')
    lrs        = AverageMeter('LR',   ':.4e')
    losses     = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, h2d_time, lrs, losses],
                             prefix=f"Epoch: [{epoch}]")
    # switch to train mode
    model.train()
    iters_per_epoch = len(train_loader)    
    end = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    for i, (images, _) in enumerate(train_loader):        
        # ------------------- load + H2D -------------------
        t_iter_start = time.perf_counter()
        x1 = images[0].to(device, non_blocking=True)
        x2 = images[1].to(device, non_blocking=True)
        
        # if i == 0:
        #     # check the shape of the first batch
        #     print("Batch shape:", x1.shape, x2.shape)

        t_after_h2d = time.perf_counter()
        h2d_time.update(t_after_h2d - t_iter_start)
        
        lr_now = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, lr, warmup_epochs, epochs)
        lrs.update(lr_now)
        m = adjust_moco_momentum(epoch + i / iters_per_epoch,
                                moco_m, epochs) if moco_m_cos else moco_m

        # ------------------- forward/backward -------------------
        with autocast(device_type=device.type, dtype=torch.float16):
            loss = model(x1, x2, m=m)

        losses.update(loss.detach().float().item(), x1.size(0))

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ------------------- timing/logging -------------------
        t_iter_end = time.perf_counter()
        step_time = t_iter_end - end # in the unit of second
        batch_time.update(step_time)
        end = t_iter_end

        if (i + 1) % print_freq == 0:
            progress.display(i + 1)
            eff_img_per_s = (2 * x1.size(0)) / batch_time.val  # two views
            gi = epoch * iters_per_epoch + i
            summary_writer.add_scalar("train/throughput_img_per_s", eff_img_per_s, gi)
            summary_writer.add_scalar("train/loss", losses.val, gi)
            summary_writer.add_scalar("train/lr", lrs.val, gi)

def train_moco(
    model,
    train_loader,
    epochs,
    warmup_epochs,
    moco_m,
    moco_m_cos,
    lr,
    wd,
    momentum,
    optimizer,
    device,
    ckpt_dir,
    print_freq=10,
    save_freq=50,
):
    model.to(device)
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp)


    if optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr,
                                        weight_decay=wd,
                                        momentum=momentum)
    elif optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr,
                                weight_decay=wd)
        
    summary_writer = SummaryWriter()
    
    for epoch in range(0, epochs):
        # train for one epoch
        train_one_epoch(
            model,
            train_loader,
            lr,
            warmup_epochs,
            epochs,
            moco_m,
            moco_m_cos,
            optimizer,
            scaler,
            epoch,
            device,
            summary_writer,
            print_freq
        )

        # save per save_freq epochs
        os.makedirs(ckpt_dir, exist_ok=True)
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename=f'{ckpt_dir}/checkpoint_{epoch+1:04d}.pth.tar')

    summary_writer.close()
    