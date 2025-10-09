import time

from data.data import get_ssl_dataloader
from models.models import MoCo
from scripts.ssl_script_test import train_moco

from data.data import get_lp_dataloaders
from utils.utils import set_seed, CSVLogger
from scripts.lp_script import construct_backbone, train_classifier, test
from models.models import Classifier
from copy import deepcopy

import torch
from statistics import mean, pstdev  # population std (N), not N-1
from pathlib import Path

def run_once(config: dict):
    # Seed + device
    deterministic = True
    set_seed(config["seed"], deterministic=deterministic, benchmark=False if deterministic else True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    total_start = time.time()

    # ---- SSL Stage ----
    ssl_start = time.time()

    lr = float(config["ssl_lr"]) * (config["ssl_batch_size"] / 256)
    train_loader = get_ssl_dataloader(
        batch_size=config["ssl_batch_size"],
        workers=config["workers"],
        da_method=config["ssl_da_method"],
        crop_min=config["ssl_crop_min"],
        flips=config["ssl_flips"],
        rotations=config["ssl_rotations"],
        normalization=config["ssl_normalization"],
        seed=config["seed"],
    )
    print(f"Number of SSL training samples: {len(train_loader.dataset)}")

    moco_model = MoCo(
        encoder_name=config["arch"],
        dim=config["moco_dim"],
        mlp_dim=config["moco_mlp_dim"],
        T=config["moco_t"],
        proj_layers=3 if "vit" in config["arch"] else 2,
        pred_layers=2,
        moco_style=False
    )

    print(f"=> creating MoCo model '{config['arch']}'")

    print("Starting SSL training...")
    moco_weights, ssl_loss = train_moco(
        model=moco_model,
        train_loader=train_loader,
        epochs=int(config["ssl_epochs"]),
        warmup_epochs=int(config["ssl_warmup_epochs"]),
        moco_m=float(config["moco_m"]),
        moco_m_cos=bool(config["moco_m_cos"]),
        lr=lr,
        wd=float(config["ssl_wd"]),
        momentum=float(config["ssl_momentum"]),
        optimizer=str(config["ssl_optimizer"]),
        device=device,
        ckpt_dir=None,
        print_freq=int(config["ssl_print_freq"]),
        save_freq=int(config["ssl_save_freq"]),
    )

    ssl_time = time.time() - ssl_start
    print(f"SSL training completed in {ssl_time:.2f} seconds.\n")
    
    del moco_model, train_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # ---- LP Stage ----
    lp_start = time.time()

    train_loader, val_loader, test_loader = get_lp_dataloaders(
        batch_size=config["lp_batch_size"],
        workers=config["workers"],
        normalization=config["ssl_normalization"],
        seed=config["seed"],
    )
    print(f"Number of LP training samples: {len(train_loader.dataset)}")

    backbone = construct_backbone(
        arch=config["arch"],
        ckpt_path=None,
        model_weights=moco_weights,
        moco_t=config["moco_t"],
        moco_dim=config["moco_dim"],
        moco_mlp_dim=config["moco_mlp_dim"],
        device=device,
        moco_style=False
    )
    model = Classifier(
        backbone=backbone,
        num_classes=10,
        requires_grad=False,
        eval_mode=True,
        moco_style=False
    )
    print(f"=> creating classifier on backbone '{config['arch']}'")

    print("Starting LP training...")
    classifier_weights, best_val_acc, best_epoch = train_classifier(
        epochs=int(config["lp_epochs"]),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(config["lp_lr"]),
        wd=float(config["lp_wd"]),
        device=device,
        out_dir=None,
        use_amp=True,
        label_smoothing=float(config["lp_label_smoothing"]),
        print_freq=int(config["lp_print_freq"]),
    )

    test_loss, test_acc = test(
        model, test_loader, device, ckpt_path=None, out_dir=None, model_weights=classifier_weights
    )

    lp_time = time.time() - lp_start
    total_time = time.time() - total_start
    print(f"LP training completed in {lp_time:.2f} seconds.\n")
    print(f"Total time: {total_time:.2f} seconds.\n")
    print("Summary:")
    print(f"SSL loss: {ssl_loss:.4f}")
    print(f"Best val acc: {best_val_acc:.2f} at epoch {best_epoch}")
    print(f"Test acc: {test_acc:.2f} with test loss {test_loss:.4f}\n")
    
    # Cleanup
    del backbone, model, train_loader, val_loader, test_loader, classifier_weights, moco_weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {
        "ssl_loss": ssl_loss,
        "lp_best_val_acc": best_val_acc,
        "lp_best_epoch": best_epoch,
        "lp_test_acc": test_acc,
        "lp_test_loss": test_loss,
        "ssl_time_sec": ssl_time,
        "lp_time_sec": lp_time,
        "total_time_sec": total_time,
    }

def run_multi_seed(base_config: dict, seeds: list[int], logger: CSVLogger = None):
    per_seed = []
    for s in seeds:
        cfg = deepcopy(base_config)
        cfg["seed"] = int(s)
        out = run_once(cfg)
        out["seed"] = s
        per_seed.append(out)
        if logger:
            logger.log({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "seed": s,
                "ssl_da_method": cfg["ssl_da_method"],
                "ssl_normalization": cfg["ssl_normalization"],
                "ssl_flips": cfg["ssl_flips"],
                "ssl_rotations": cfg["ssl_rotations"],
                "ssl_crop_min": cfg["ssl_crop_min"],
                "ssl_lr": cfg["ssl_lr"], "ssl_wd": cfg["ssl_wd"],
                "ssl_loss": out["ssl_loss"],
                "lp_best_val_acc": out["lp_best_val_acc"],
                "lp_best_epoch": out["lp_best_epoch"],
                "lp_test_acc": out["lp_test_acc"],
                "lp_test_loss": out["lp_test_loss"],
                "seed": out["seed"],
                "ssl_time_sec": out["ssl_time_sec"],
                "lp_time_sec": out["lp_time_sec"],
                "total_time_sec": out["total_time_sec"],
                })

    def mstd(key):
        vals = [x[key] for x in per_seed]
        return float(mean(vals)), float(pstdev(vals))  # std over seeds

    summary = {
        "ssl_loss_mean": mstd("ssl_loss")[0],
        "ssl_loss_std":  mstd("ssl_loss")[1],
        "lp_best_val_acc_mean": mstd("lp_best_val_acc")[0],
        "lp_best_val_acc_std":  mstd("lp_best_val_acc")[1],
        "lp_test_acc_mean":     mstd("lp_test_acc")[0],
        "lp_test_acc_std":      mstd("lp_test_acc")[1],
        "lp_test_loss_mean":    mstd("lp_test_loss")[0],
        "lp_test_loss_std":     mstd("lp_test_loss")[1],
        "ssl_time_sec_mean":    mstd("ssl_time_sec")[0],
        "lp_time_sec_mean":     mstd("lp_time_sec")[0],
        "total_time_sec_mean":  mstd("total_time_sec")[0],
    }
    return per_seed, summary

def main(base_config: dict, exp_pairs: list[tuple], seeds: list[int] = [2952]):
    results_csv = "results/ssl_lp_sweep.csv"
    summary_csv = "results/ssl_lp_summary.csv"
    Path("results").mkdir(parents=True, exist_ok=True)

    # per-seed rows
    seed_fields = [
        "timestamp","seed", "ssl_da_method","ssl_normalization","ssl_flips","ssl_rotations",
        "ssl_crop_min","ssl_lr","ssl_wd",
        "ssl_loss","lp_best_val_acc","lp_best_epoch","lp_test_acc","lp_test_loss",
        "seed","ssl_time_sec","lp_time_sec","total_time_sec"
    ]
    logger = CSVLogger(results_csv, seed_fields, overwrite=False)

    # aggregated rows
    summary_fields = [
        "timestamp","ssl_da_method","ssl_normalization","ssl_flips","ssl_rotations",
        "ssl_crop_min","ssl_lr","ssl_wd","seeds",
        "ssl_loss_mean","ssl_loss_std",
        "lp_best_val_acc_mean","lp_best_val_acc_std",
        "lp_test_acc_mean","lp_test_acc_std",
        "lp_test_loss_mean","lp_test_loss_std",
        "ssl_time_sec_mean","lp_time_sec_mean","total_time_sec_mean"
    ]
    summary_logger = CSVLogger(summary_csv, summary_fields, overwrite=False)

    for (ssl_lr, ssl_wd, da_method, norm, flips, rotations, crop_min) in exp_pairs:
        cfg = deepcopy(base_config)
        cfg.update({
            "ssl_lr": ssl_lr, "ssl_wd": ssl_wd,
            "ssl_da_method": da_method, "ssl_normalization": norm,
            "ssl_flips": bool(flips), "ssl_rotations": bool(rotations),
            "ssl_crop_min": float(crop_min),
        })
        print("="*80)
        print(f"DA={da_method} Norm={norm} LR={ssl_lr} WD={ssl_wd} "
              f"flips={flips} rotations={rotations} crop_min={crop_min} | seeds={seeds}")
        print("="*80)

        _, summary = run_multi_seed(cfg, seeds, logger=logger)

        # log aggregated row
        summary_logger.log({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "ssl_da_method": cfg["ssl_da_method"],
            "ssl_normalization": cfg["ssl_normalization"],
            "ssl_flips": cfg["ssl_flips"],
            "ssl_rotations": cfg["ssl_rotations"],
            "ssl_crop_min": cfg["ssl_crop_min"],
            "ssl_lr": cfg["ssl_lr"], "ssl_wd": cfg["ssl_wd"],
            "seeds": ";".join(map(str, seeds)),
            **summary
        })
    print(f"Saved per-seed to {results_csv} and summaries to {summary_csv}")

if __name__ == "__main__":
    # changing values
    # 1. ssl_da_method: "BYOL" or "GA"
    # 2. ssl_normalization: "IN" or "GA"
    # 3. (ssl_lr, ssl_wd) pairs
    
    EXP_PAIRS = [
        # LR, WD, DA method, Normalization, random flips, random rotations, crop_min
        (3e-4, 0.05, "GA",   "GAFG", True, True, 0.2),
        (3e-4, 0.05, "GA",   "IN", True, True, 0.2),   # GA + IN
        (3e-4, 0.05, "BYOL", "IN", True, True, 0.2),   # BYOL + IN
        (3e-4, 0.05, "GA",   "GAFG", True, True, 0.4),
        (3e-4, 0.05, "GA",   "IN", True, True, 0.4),   # GA + IN
        (3e-4, 0.05, "BYOL", "IN", True, True, 0.4),
        (3e-4, 0.05, "GA",   "GAFG", True, True, 0.6),
        (3e-4, 0.05, "GA",   "IN", True, True, 0.6),   # GA + IN
    ]

    INVARIANTS_CONFIG = {
        # Repro & system
        "workers": 32,

        # Architecture & MoCo params
        "arch": "vit_s",
        "moco_dim": 256,
        "moco_mlp_dim": 4096,
        "moco_t": 0.2,
        "moco_m_cos": True,
        "moco_m": 0.99,
        
        # SSL training
        "ssl_batch_size": 256,
        "ssl_optimizer": "adamw",
        "ssl_momentum": 0.9,
        "ssl_epochs": 100,            # quick sweep; extend to 300 for confirmatory run
        "ssl_warmup_epochs": 10,
        "ssl_print_freq": 10,
        "ssl_save_freq": 50,

        # Linear probing
        "lp_batch_size": 256,
        "lp_optimizer": "adamw",
        "lp_epochs": 100,
        "lp_lr": 3e-3,
        "lp_wd": 0.05,
        "lp_label_smoothing": 0.05,
        "lp_print_freq": 10,
    }

    main(INVARIANTS_CONFIG, EXP_PAIRS)