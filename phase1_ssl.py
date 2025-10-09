import argparse
import yaml
import torch

from utils.utils import set_seed
from data.data import get_ssl_dataloader
from models.models import MoCo
from scripts.ssl_script_test import train_moco

def ssl(config: dict):
    # Seed + device
    set_seed(config["seed"], deterministic=True, benchmark=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Derived values (linear LR scaling by batch size)
    lr = float(config["ssl_lr"]) * (config["ssl_batch_size"] / 256)

    # Data
    train_loader = get_ssl_dataloader(
        batch_size=config["ssl_batch_size"],
        workers=config["workers"],
    )
    print(f"Number of SSL training samples: {len(train_loader.dataset)}")

    # Model
    model = MoCo(
        encoder_name=config["arch"],
        dim=config["moco_dim"],
        mlp_dim=config["moco_mlp_dim"],
        T=config["moco_t"],
        proj_layers=3 if "vit" in config["arch"] else 2,
        pred_layers=2,
    )
    print(f"=> creating MoCo model '{config['arch']}'")

    # ckpt dir => arch + batch_size + lr + wd + epochs + ssl_da_method
    ckpt_dir = (
        f"./checkpoints/checkpoints_{config['arch']}_"
        f"{config['ssl_batch_size']}_"
        f"{config['ssl_lr']}_"
        f"{config['ssl_wd']}_"
        f"{config['ssl_epochs']}_"
        f"{config['ssl_da_method']}"
    )

    print("Starting SSL training...")
    train_moco(
        model=model,
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
        ckpt_dir=str(config["ckpt_dir"]),
        print_freq=int(config["ssl_print_freq"]),
        save_freq=int(config["ssl_save_freq"]),
    )
    
    # delete model and free up memory
    del model, train_loader
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    ssl(config)
