import argparse
import yaml
import torch

from data.data import get_lp_dataloaders
from utils.utils import set_seed
from scripts.lp_script import construct_backbone, train_classifier, test
from models.models import Classifier

def lp(config: dict):
    # Seed + deterministic knobs
    set_seed(config["seed"], deterministic=True, benchmark=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data
    train_loader, val_loader, test_loader = get_lp_dataloaders(
        batch_size=config["lp_batch_size"],
        workers=config["workers"],
        normalization=config["ssl_normalization"],
    )
    print(f"Number of LP training samples: {len(train_loader.dataset)}")

    # Backbone + classifier
    ckpt_path = (
        f"./checkpoints/checkpoints_{config['arch']}_"
        f"{config['ssl_batch_size']}_"
        f"{config['ssl_lr']}_"
        f"{config['ssl_wd']}_"
        f"{config['ssl_epochs']}_"
        f"{config['ssl_da_method']}/checkpoint_{int(config['ssl_epochs']):04d}.pth.tar"
    )
    backbone = construct_backbone(
        arch=config["arch"],
        ckpt_path=ckpt_path,
        moco_t=config["moco_t"],
        moco_dim=config["moco_dim"],
        moco_mlp_dim=config["moco_mlp_dim"],
        device=device,
    )
    model = Classifier(backbone=backbone, num_classes=10, requires_grad=False, eval_mode=True) # set backbone to eval mode and freeze its weights
    print(f"=> creating classifier on backbone '{config['arch']}'")

    # Output dir
    out_dir = (
        f"./results/{config['arch']}_"
        f"{config['ssl_batch_size']}_"
        f"{config['ssl_lr']}_"
        f"{config['ssl_wd']}_"
        f"{config['ssl_epochs']}_"
        f"{config['ssl_da_method']}_"
        f"{config['lp_batch_size']}_"
        f"{config['lp_lr']}_"
        f"{config['lp_wd']}"
    )

    print("Starting LP training...")
    train_classifier(
        epochs=int(config["lp_epochs"]),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=float(config["lp_lr"]),
        wd=float(config["lp_wd"]),
        device=device,
        out_dir=out_dir,
        use_amp=True,
        label_smoothing=float(config["lp_label_smoothing"]),
        print_freq=int(config["lp_print_freq"]),
    )

    # Test using best checkpoint saved during training
    best_model_path = f"{out_dir}/best.ckpt"
    test(model, test_loader, device, ckpt_path=best_model_path, out_dir=out_dir)
    
    # delete model and free up memory
    del model, backbone, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    lp(config)
