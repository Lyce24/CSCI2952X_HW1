import torch
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import CSVLogger
from models.models import MoCo

import copy
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, log_loss, roc_auc_score, confusion_matrix
)
import numpy as np
import torch.nn.functional as F
import json

def construct_backbone(arch: str, 
                       ckpt_path: str = None, 
                       model_weights = None,
                       moco_t: float = 0.2, 
                       moco_dim: int = 256, 
                       moco_mlp_dim: int = 4096,
                       device: torch.device = torch.device("cpu"),
                       moco_style: bool = True) -> nn.Module:
    """Construct backbone from a pre-trained MoCo model or a standard model from timm. 
    """
    model = MoCo(
    encoder_name=arch,
    dim=moco_dim,
    mlp_dim=moco_mlp_dim,
    T=moco_t,
    proj_layers=3 if 'vit' in arch else 2,
    pred_layers=2,
    moco_style=moco_style
    )
    
    # Load checkpoint if provided
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.to(device)
        feature_extractor = model.base_encoder
        return feature_extractor
    elif model_weights is not None:
        model.load_state_dict(model_weights, strict=True)
        model.to(device)
        feature_extractor = model.base_encoder
        return feature_extractor
    else:
        print("No checkpoint path provided, returning randomly initialized model.")
        # randomly initialized model
        model.to(device)
        feature_extractor = model.base_encoder 
        return feature_extractor

# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def add_weight_decay_exclusions(model: nn.Module, weight_decay: float, lr: float = 1e-3) -> list[dict]:
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
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]  
  
# ----------------------------
#   TRAIN / VAL EPOCHS
# ----------------------------
def train_one_epoch(model,
                    loader,
                    optimizer,
                    criterion,
                    device,
                    scheduler=None,
                    scaler: torch.amp.GradScaler | None = None,
                    grad_clip_norm: float | None = None):
    model.train()
    running_loss, n = 0.0, 0

    use_amp = (scaler is not None)
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()

            # unscale before clipping
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()  # per-iteration cosine

        bs = labels.size(0)
        running_loss += loss.item() * bs
        n += bs

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
                     out_dir=None,
                     use_amp=True,
                     backbone_lr=None,
                     backbone_wd=None,
                     label_smoothing: float = 0.05,
                     print_freq=10,
                     grad_clip_norm: float | None = None,
                     eta_min: float = 1e-6):

    model.to(device)

    # ---- Loss ----
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ---- Optimizer (supports separate backbone head lrs/wds) ----
    if backbone_lr is not None and backbone_wd is not None:
        params = (
            add_weight_decay_exclusions(model.backbone, backbone_wd, backbone_lr)
            + add_weight_decay_exclusions(model.fc, wd, lr)
        )
        optimizer = optim.AdamW(params, betas=(0.9, 0.999))
    else:
        param_groups = add_weight_decay_exclusions(model, weight_decay=wd, lr=lr)
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))

    # ---- AMP scaler ----
    scaler = torch.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # ---- LR schedule: cosine per-iteration ----
    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    # ---- I/O ----
    logger = None
    best_path = None
    best_weights = None
    if out_dir is not None:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        log_path = out / "train_log.csv"
        logger = CSVLogger(
            str(log_path),
            fieldnames=["phase", "epoch", "lr", "loss", "acc", "time_sec"]
        )
        best_path = out / "best.ckpt"

    # ---- Early stopping ----
    patience = max(1, int(0.2 * epochs))  # ~20% of total epochs
    best_val_acc = -1.0
    best_epoch = -1
    epochs_since_improve = 0

    # ---- Train loop ----
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,   # step inside train_one_epoch per iteration
            scaler=scaler,
            grad_clip_norm=grad_clip_norm
        )
        t1 = time.time()

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        t2 = time.time()

        # current LR (read from the first param group)
        lr_now = optimizer.param_groups[0]["lr"]

        # logging
        if logger is not None:
            logger.log({"phase":"train_epoch","epoch":epoch,"lr":f"{lr_now:.8f}",
                        "loss":f"{train_loss:.6f}","acc":"","time_sec":f"{t1 - t0:.3f}"})
            logger.log({"phase":"val_epoch","epoch":epoch,"lr":f"{lr_now:.8f}",
                        "loss":f"{val_loss:.6f}","acc":f"{val_acc:.6f}","time_sec":f"{t2 - t1:.3f}"})

        if epoch % print_freq == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"lr {lr_now:.2e} | "
                  f"train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | "
                  f"val_acc {val_acc*100:.2f}% | "
                  f"{t1 - t0:.1f}s/{t2 - t1:.1f}s (train/val)")

        # best/early-stopping
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_since_improve = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs). "
                  f"Best @ epoch {best_epoch} with val_acc {best_val_acc*100:.2f}%")
            break

    # ---- save best weights ----
    if (best_path is not None) and (best_weights is not None):
        torch.save({
            "model": best_weights,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
        }, best_path)


    print(f"Done. Best val_acc: {best_val_acc*100:.2f}% at epoch {best_epoch}. "
          f"Saved best -> {best_path if best_path is not None else 'N/A'}")
    
    return best_weights, best_val_acc, best_epoch

# Report ACC, PREC, RECALL, F1, AUROC (for multi-class), confusion matrix
@torch.no_grad()
def final_evaluation(model,
                     loader,
                     device,
                     criterion: nn.Module | None = None,
                     expected_num_classes: int = 10):
    """
    Multi-class evaluation for C > 2 (default C=10).
    - Uses argmax predictions.
    - Macro precision/recall/F1.
    - Macro-OVR AUROC.
    - Confusion matrix over fixed label order [0..C-1].

    Returns:
        metrics: dict
        y_true, y_pred, y_prob: np.ndarrays
    """
    model.eval()
    model.to(device)

    all_logits = []
    all_targets = []
    n_examples = 0
    running_loss = 0.0

    for batch in loader:
        if isinstance(batch, dict):
            x = batch.get("image")
            y = batch.get("label")
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)                          # (B, C)
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

        if criterion is not None:
            loss = criterion(logits, y)
            bs = y.size(0)
            running_loss += loss.item() * bs
            n_examples += bs

    if not all_logits:
        raise ValueError("Empty dataloader: no batches to evaluate.")

    logits = torch.cat(all_logits, dim=0).numpy()  # (N, C)
    y_true = torch.cat(all_targets, dim=0).numpy().astype(int)

    C = logits.shape[1]
    if expected_num_classes is not None and C != expected_num_classes:
        raise ValueError(f"Model outputs C={C}, expected {expected_num_classes}.")
    if np.any((y_true < 0) | (y_true >= C)):
        raise ValueError(f"Targets must be in [0,{C-1}] to match model outputs; got min={y_true.min()}, max={y_true.max()}.")

    # probs and predictions
    y_prob = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = y_prob.argmax(axis=1)

    # metrics
    metrics = {}
    metrics["loss"] = float(running_loss / n_examples) if (criterion is not None and n_examples > 0) else float("nan")
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    # log loss & AUROC (macro-OVR); guard for rare label-absence issues
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=np.arange(C)))
    except Exception:
        metrics["log_loss"] = float("nan")
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=np.arange(C)))
    except Exception:
        metrics["roc_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(C))
    metrics["confusion_matrix"] = cm.astype(int).tolist()

    return metrics, y_true, y_pred, y_prob

@torch.no_grad()
def test(model, test_loader, device, ckpt_path=None, model_weights=None, out_dir=None):
    """
    Load weights, evaluate on test_loader, print and optionally write results.
    Returns (test_loss, test_acc, metrics_dict).
    """
    # Load weights
    if model_weights is not None:
        model.load_state_dict(model_weights)
        print("Loaded model weights from provided state_dict.")
    elif ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        key = "model" if "model" in ckpt else None
        sd = ckpt[key] if key else ckpt
        model.load_state_dict(sd)
        print(f"Loaded model weights from {ckpt_path}.")
    else:
        raise ValueError("Provide either model_weights or ckpt_path.")

    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_pred, y_prob = final_evaluation(model, test_loader, device, criterion=criterion)

    test_loss = metrics.get("loss", float("nan"))
    test_acc = metrics.get("accuracy", float("nan"))

    print(
        f"Test Loss: {test_loss:.4f} | "
        f"ACC: {test_acc*100:.2f}% | "
        f"PREC: {metrics['precision']:.4f} | "
        f"REC: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"AUROC: {metrics['roc_auc']:.4f}"
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # save metrics JSON
        (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

        # save confusion matrix CSV
        try:
            import pandas as pd
            import numpy as np
            cm = np.array(metrics["confusion_matrix"], dtype=int)
            pd.DataFrame(cm, index=[f"t{i}" for i in range(cm.shape[0])],
                         columns=[f"p{i}" for i in range(cm.shape[1])]).to_csv(out / "confusion_matrix.csv")
        except Exception:
            # fallback to plain text if pandas isn't available
            with open(out / "confusion_matrix.txt", "w") as f:
                for row in metrics["confusion_matrix"]:
                    f.write(",".join(map(str, row)) + "\n")

        print(f"Saved metrics and artifacts to {out.resolve()}")

    return test_loss, test_acc, metrics