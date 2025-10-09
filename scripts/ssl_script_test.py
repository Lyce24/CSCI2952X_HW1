from utils.utils import LARS, CSVLogger, save_checkpoint
import torch.optim as optim
import torch
import torch.nn as nn
import math
import time
import os
import random
import numpy as np
from torch.amp import GradScaler, autocast

# -------------------------
# Schedules (step-aware)
# -------------------------
def cosine_with_warmup_step(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    """
    Per-step cosine LR with linear warmup in steps.
    """
    step = min(step, total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    # cosine from 1.0 -> 0.0 over [warmup_steps, total_steps]
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

def cosine_momentum_099_to_1(step: int, total_steps: int, m_start: float = 0.99) -> float:
    """
    Cosine ramp of momentum from m_start -> 1.0 over total_steps.
    """
    step = min(step, total_steps)
    progress = step / float(max(1, total_steps))
    # m = 1 - (1-m_start)*0.5*(1+cos(pi * progress))
    return 1.0 - (1.0 - m_start) * 0.5 * (1.0 + math.cos(math.pi * progress))

# -------------------------
# Optimizer helpers
# -------------------------
def add_weight_decay_exclusions(model: nn.Module, weight_decay: float):
    """
    Create AdamW param groups with no weight decay for bias and norm parameters.
    Heuristic: no-decay for 1D params (e.g., LN/Bias), and names containing 'bias' or 'norm'.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower() or "ln" in name.lower() or "layernorm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def train_one_epoch(
    model,
    train_loader,
    base_lr,
    moco_m,             
    moco_m_cos,
    optimizer,
    scaler,
    epoch,
    device,
    print_freq,
    logger: CSVLogger,
    total_steps: int,
    warmup_steps: int,
    global_step: int,
    max_grad_norm: float = 3.0,
    amp_dtype=torch.float16,
    iters_per_epoch=None,
):
    # switch to train mode
    model.train()
    end = time.perf_counter()

    running_loss = 0.0
    epoch_time = 0.0

    optimizer.zero_grad(set_to_none=True)

    for i, (images, _) in enumerate(train_loader):        
        # ------------------- H2D -------------------
        x1 = images[0].to(device, non_blocking=True)
        x2 = images[1].to(device, non_blocking=True)
        batch_size = x1.size(0)  # number of original images

        # ------------------- per-step schedules -------------------
        lr_now = cosine_with_warmup_step(base_lr, global_step, total_steps, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        if moco_m_cos:
            m = cosine_momentum_099_to_1(global_step, total_steps, m_start=0.99)
        else:
            m = moco_m

        # ------------------- forward/backward -------------------
        with autocast(device_type=device.type, dtype=amp_dtype):
            loss = model(x1, x2, m=m)

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        
        # Clip only if needed
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            # quick check
            total_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in model.parameters() if p.grad is not None]), p=2)
            if total_norm > max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # ------------------- timing/logging -------------------
        t_iter_end = time.perf_counter()
        step_time = t_iter_end - end
        end = t_iter_end

        epoch_time += step_time
        loss_f = float(loss.detach())
        running_loss += loss_f

        # per-image timing/throughput
        time_per_image = step_time / max(1, batch_size)
        imgs_per_sec = batch_size / max(1e-12, step_time)

        if (i + 1) % print_freq == 0:
            print(
                f'Epoch: [{epoch}][{i+1}/{iters_per_epoch}] '
                f'Time {step_time:.3f}s  '
                f'LR {lr_now:.6e}  '
                f'm {m:.6f}  '
                f'Loss {loss_f:.4f}  '
                f't/img {time_per_image*1e3:.2f}ms  '
                f'img/s {imgs_per_sec:.1f}'
            )
            if logger is not None:
                logger.log({
                    "phase": "train_iter",
                    "epoch": epoch,
                    "iter": i + 1,
                    "iters_per_epoch": iters_per_epoch,
                    "lr": f"{lr_now:.8e}",
                    "moco_m": f"{m:.6f}",
                    "loss": f"{loss_f:.6f}",
                    "step_time_sec": f"{step_time:.6f}",
                    "time_per_image_sec": f"{time_per_image:.8f}",
                    "images_per_sec": f"{imgs_per_sec:.2f}",
                    "global_step": global_step,
                })

        global_step += 1

    avg_time = epoch_time / max(1, iters_per_epoch)
    avg_loss = running_loss / max(1, iters_per_epoch)
    print(f' * Epoch [{epoch}]  Avg Loss {avg_loss:.4f}  Avg Step {avg_time:.3f}s')

    # epoch summary
    if logger is not None:
        logger.log({
            "phase": "epoch_summary",
            "epoch": epoch,
            "iter": iters_per_epoch,
            "iters_per_epoch": iters_per_epoch,
            "lr": f"{lr_now:.8e}",
            "moco_m": f"{m:.6f}",
            "loss": f"{avg_loss:.6f}",
            "step_time_sec": f"{avg_time:.6f}",
            "time_per_image_sec": f"{(avg_time / max(1, batch_size)):.8f}",
            "images_per_sec": f"{(batch_size / max(1e-12, avg_time)):.2f}",
            "global_step": global_step,
        })
        
    # loss
    return global_step, avg_loss

# -------------------------
# Top-level train
# -------------------------
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
    ckpt_dir=None,
    print_freq=10,
    save_freq=50,
    max_grad_norm: float = 3.0,
):
    model.to(device)
    use_amp = (device.type == 'cuda')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    scaler = GradScaler(enabled=use_amp)

    if optimizer == 'lars':
        opt = LARS(model.parameters(), lr, weight_decay=wd, momentum=momentum)
    elif optimizer == 'adamw':
        param_groups = add_weight_decay_exclusions(model, weight_decay=wd)
        opt = optim.AdamW(param_groups, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # logging
    logger = None
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        log_path = os.path.join(ckpt_dir, "train_log.csv")
        logger = CSVLogger(
            log_path,
            fieldnames=[
                "phase", "epoch", "iter", "iters_per_epoch",
                "lr", "moco_m", "loss",
                "step_time_sec", "time_per_image_sec", "images_per_sec",
                "global_step"
            ],
        )

    # step accounting
    iters_per_epoch = len(train_loader)
    total_steps = epochs * iters_per_epoch
    warmup_steps = int(warmup_epochs * iters_per_epoch)
    global_step = 0

    # training loop
    for epoch in range(0, epochs):
        # train for one epoch
        global_step, avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            base_lr=lr,
            moco_m=moco_m,
            moco_m_cos=moco_m_cos,
            optimizer=opt,
            scaler=scaler,
            epoch=epoch,
            device=device,
            print_freq=print_freq,
            logger=logger,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            global_step=global_step,
            max_grad_norm=max_grad_norm,
            amp_dtype=amp_dtype,
            iters_per_epoch=iters_per_epoch,
        )
        
        # periodic & final checkpoint
        if ckpt_dir is None:
            continue
        
        if ((epoch + 1) % save_freq == 0) or ((epoch + 1) == epochs):
            ckpt_path = f'{ckpt_dir}/checkpoint_{epoch+1:04d}.pth.tar'
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
                "rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_random_state": random.getstate(),
            }
            if torch.cuda.is_available():
                state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
            save_checkpoint(state, is_best=False, filename=ckpt_path)
        
    return model.state_dict(), avg_loss # return final model weights and loss