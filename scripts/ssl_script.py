from utils.utils import LARS, CSVLogger
import torch.optim as optim
import math
import torch
import shutil
import time
import os
from torch.amp import GradScaler, autocast

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
    print_freq,
    logger: CSVLogger
):    
    # switch to train mode
    model.train()
    iters_per_epoch = len(train_loader)    
    end = time.perf_counter()
    
    losses = 0.0
    epoch_time = 0.0

    optimizer.zero_grad(set_to_none=True)
    for i, (images, _) in enumerate(train_loader):        
        # ------------------- load + H2D -------------------
        x1 = images[0].to(device, non_blocking=True)
        x2 = images[1].to(device, non_blocking=True)
        
        # if i == 0:
        #     # check the shape of the first batch
        #     print("Batch shape:", x1.shape, x2.shape)

        lr_now = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, lr, warmup_epochs, epochs)
        m = adjust_moco_momentum(epoch + i / iters_per_epoch,
                                moco_m, epochs) if moco_m_cos else moco_m

        # ------------------- forward/backward -------------------
        with autocast(device_type=device.type, dtype=torch.float16):
            loss = model(x1, x2, m=m)

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ------------------- timing/logging -------------------
        t_iter_end = time.perf_counter()
        step_time = t_iter_end - end # in the unit of second
        end = t_iter_end
        
        epoch_time += step_time
        loss_f = loss.detach().float().item()
        losses += loss_f

        if (i + 1) % print_freq == 0:
            print(f'Epoch: [{epoch}][{i+1}/{iters_per_epoch}]\t'
                  f'Time {step_time:.3f}\t'
                  f'LR {lr_now:.6f}\t'
                  f'Loss {loss_f:.4f}')
            
            logger.log({
                "phase": "train_iter",
                "epoch": epoch,
                "iter": i + 1,
                "iters_per_epoch": iters_per_epoch,
                "lr": f"{lr_now:.8f}",
                "moco_m": f"{m:.6f}",
                "loss": f"{loss_f:.6f}",
                "step_time_sec": f"{step_time:.6f}",
            })
            
    avg_time = epoch_time / iters_per_epoch
    avg_loss = losses / iters_per_epoch
    print(f' * Epoch: [{epoch}] Average Loss {avg_loss:.4f}\tAverage Time {avg_time:.3f}')
    # -------- CSV epoch summary --------
    logger.log({
        "phase": "epoch_summary",
        "epoch": epoch,
        "iter": iters_per_epoch,
        "iters_per_epoch": iters_per_epoch,
        "lr": f"{lr_now:.8f}",
        "moco_m": f"{m:.6f}",
        "loss": f"{avg_loss:.6f}",
        "step_time_sec": f"{avg_time:.6f}",
    })

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
        optimizer = optim.AdamW(model.parameters(), lr,
                                weight_decay=wd)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "train_log.csv")
    logger = CSVLogger(
        log_path,
        fieldnames=[
            "phase", "epoch", "iter", "iters_per_epoch",
            "lr", "moco_m", "loss",
            "step_time_sec"
        ],
    )

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
            print_freq,
            logger
        )

        # save per save_freq epochs
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            save_checkpoint({
                'state_dict': model.state_dict(),
            }, is_best=False, filename=f'{ckpt_dir}/checkpoint_{epoch+1:04d}.pth.tar')    