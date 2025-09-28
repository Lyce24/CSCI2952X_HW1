import torch
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import CSVLogger
from models.models import MoCo

def construct_backbone(arch: str, ckpt_path: str = None, moco_t: float = 0.2, moco_dim: int = 256, moco_mlp_dim: int = 4096, device: torch.device = torch.device("cpu")) -> nn.Module:
    """Construct backbone from a pre-trained MoCo model or a standard model from timm. 
    """
    model = MoCo(
    encoder_name=arch,
    dim=moco_dim,
    mlp_dim=moco_mlp_dim,
    T=moco_t,
    proj_layers=3 if 'vit' in arch else 2,
    pred_layers=2,
    )
    
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        raise ValueError("Please provide a valid checkpoint path.")
    
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    feature_extractor = model.base_encoder

    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()
    return feature_extractor

# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

# -------------------------
# Train / Val loops
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        n += batch_size

    return running_loss / max(n, 1)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, labels) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


def train_classifier(epochs, 
                    model, 
                    train_loader, 
                    val_loader,
                    lr, 
                    wd,
                    device, 
                    out_dir,
                    use_amp=True,
                    print_freq=10):
    
    model.to(device)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Model / Loss / Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Metrics log
    log_path = out / "train_log.csv"
    logger = CSVLogger(
        str(log_path),
        fieldnames=[
            "phase",         # train_epoch | val_epoch | checkpoint
            "epoch",
            "lr",
            "loss",
            "acc",
            "time_sec",
            "ckpt"           # best / last / ''
        ],
    )

    best_val_acc = -1.0
    best_path = out / "best.ckpt"
    last_path = out / "last.ckpt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        # Log
        if epoch % print_freq == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | "
                    f"lr {lr_now:.2e} | "
                    f"train_loss {train_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | "
                    f"val_acc {val_acc*100:.2f}% | "
                    f"{dt:.1f}s")
            
            # ---- CSV: train epoch summary ----
            logger.log({
                "phase": "train_epoch",
                "epoch": epoch,
                "iter": "",
                "lr": f"{lr_now:.8f}",
                "loss": f"{train_loss:.6f}",
                "acc": "",
                "time_sec": f"{dt:.3f}",
                "ckpt": ""
            })

            # ---- CSV: val epoch summary ----
            logger.log({
                "phase": "val_epoch",
                "epoch": epoch,
                "iter": "",
                "lr": f"{lr_now:.8f}",
                "loss": f"{val_loss:.6f}",
                "acc": f"{val_acc:.6f}",
                "time_sec": "",
                "ckpt": ""
            })

        # ---- checkpoint: last ----
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            best_path if val_acc > best_val_acc else last_path,
        )

        # mark which ckpt we wrote
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_mark = "best"
        else:
            ckpt_mark = "last"

        # ---- CSV: checkpoint marker ----
        logger.log({
            "phase": "checkpoint",
            "epoch": epoch,
            "iter": "",
            "lr": "",
            "loss": "",
            "acc": "",
            "time_sec": "",
            "ckpt": ckpt_mark
        })

    print(f"Done. Best val_acc: {best_val_acc*100:.2f}% | saved -> {best_path}")
    
def test(model, test_loader, device, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")