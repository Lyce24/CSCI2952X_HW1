import argparse
import yaml
import torch

from data.data import get_lp_dataloaders
from utils.utils import set_seed
from scripts.lp_script import construct_backbone, train_classifier, test
from models.models import Classifier


def main(config):
    # Seed + deterministic knobs match your original intent
    set_seed(config["seed"], deterministic=True, benchmark=False)

    # Device selection with safe fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = get_lp_dataloaders(
        batch_size=config["lp_batch_size"],
        workers=config["workers"],
    )

    # Backbone + classifier
    backbone = construct_backbone(
        arch=config["arch"],
        ckpt_path= f"./checkpoints_{config['arch']}_{config['ssl_batch_size']}_{config['ssl_lr']}_{config['ssl_wd']}/checkpoint_{config['ssl_epochs']:04d}.pth.tar",
        moco_t=config["moco_t"],
        moco_dim=config["moco_dim"],
        moco_mlp_dim=config["moco_mlp_dim"],
        device=device,
    )
    model = Classifier(backbone=backbone, num_classes=10)

    # out_dir
    # arch + ssl_batch_size + ssl_lr + ssl_wd + lp_batch_size + lp_lr + lp_wd
    out_dir = f"./results/{config['arch']}_{config['ssl_batch_size']}_{config['ssl_lr']}_{config['ssl_wd']}_{config['lp_batch_size']}_{config['lp_lr']}_{config['lp_wd']}"

    # Train
    train_classifier(
        epochs=config["lp_epochs"],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config["lp_lr"],
        wd=config["lp_wd"],
        device=device,
        out_dir=out_dir,
        use_amp=True,
        print_freq=config["lp_print_freq"],
    )

    # Test using best checkpoint saved by training
    best_model_path = f'{out_dir}/best.ckpt'
    test(model, test_loader, device, ckpt_path=best_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
