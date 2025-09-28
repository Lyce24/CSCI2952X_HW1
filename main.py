import argparse
import yaml
import torch

from utils.utils import set_seed
from data.data import get_ssl_dataloader, get_lp_dataloaders
from models.models import MoCo, Classifier
from scripts.ssl_script import train_moco
from scripts.lp_script import construct_backbone, train_classifier, test

def ssl(config):
    # Seed + device
    set_seed(config['seed'], deterministic=True, benchmark=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Derived values
    lr = config['ssl_lr'] * config['ssl_batch_size'] / 256

    # Data
    train_loader = get_ssl_dataloader(
        batch_size=config['ssl_batch_size'],
        workers=config['workers']
    )
    print(f"Number of training samples: {len(train_loader.dataset)}")

    # Model
    model = MoCo(
        encoder_name=config['arch'],
        dim=config['moco_dim'],
        mlp_dim=config['moco_mlp_dim'],
        T=config['moco_t'],
        proj_layers=3 if 'vit' in config['arch'] else 2,
        pred_layers=2,
    )
    print("=> creating model '{}'".format(config['arch']))
    
    # ckpt dir => arch + batch_size + lr + wd
    ckpt_dir = f"./checkpoints_{config['arch']}_{config['ssl_batch_size']}_{config['ssl_lr']}_{config['ssl_wd']}"

    print("Starting training...")
    # Training
    train_moco(
        model=model,
        train_loader=train_loader,
        epochs=config['ssl_epochs'],
        warmup_epochs=config['ssl_warmup_epochs'],
        moco_m=config['ssl_moco_m'],
        moco_m_cos=config['ssl_moco_m_cos'],
        lr=lr,
        wd=config['ssl_wd'],
        momentum=config['ssl_momentum'],
        optimizer=config['ssl_optimizer'],
        device=device,
        ckpt_dir=ckpt_dir,
        print_freq=config['ssl_print_freq'],
        save_freq=config['ssl_save_freq'],
    )

def lp(config):
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

def main(config):
    print("Starting SSL phase...")
    ssl(config)
    print("SSL phase completed.\n")

    print("Starting Linear Probing phase...")
    lp(config)
    print("Linear Probing phase completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)