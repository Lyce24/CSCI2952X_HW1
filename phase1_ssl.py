import time

from data.data import get_ssl_dataloader
from models.models import MoCo
from scripts.ssl_script import train_moco

from utils.utils import set_seed

import torch
import argparse
import yaml

def ssl(config: dict):
    # Seed + device
    deterministic = True
    set_seed(config["seed"], deterministic=deterministic, benchmark=False if deterministic else True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
        artifacts=config["ssl_artifacts"],
        noise=config["ssl_noise"],
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

    if "ckpt_dir" in config:
        ckpt_dir = config["ckpt_dir"]
    else:
        ckpt_dir = (
            f"./checkpoints/checkpoints_{config['arch']}_"
            f"{config['ssl_lr']}_"
            f"{config['ssl_wd']}_"
            f"{config['ssl_epochs']}_"
            f"{config['ssl_da_method']}_"
            f"{config['ssl_normalization']}_"
            f"{config['ssl_flips']}_"
            f"{config['ssl_rotations']}_"
            f"{config['ssl_artifacts']}_"
            f"{config['ssl_noise']}_"
            f"{config['ssl_crop_min']}"
        )

    print("Starting SSL training...")
    train_moco(
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
        ckpt_dir=str(ckpt_dir),
        print_freq=int(config["ssl_print_freq"]),
        save_freq=int(config["ssl_save_freq"]),
    )

    ssl_time = time.time() - ssl_start
    print(f"SSL training completed in {ssl_time:.2f} seconds.\n")

    del train_loader, moco_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    parser.add_argument("-ckpt", "--ckpt_dir", type=str, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.ckpt_dir is not None:
        config["ckpt_dir"] = args.ckpt_dir

    ssl(config)
