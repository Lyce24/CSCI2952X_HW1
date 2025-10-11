import argparse
import yaml
import torch
import time

from data.data import get_lp_dataloaders
from utils.utils import set_seed
from scripts.lp_script import construct_backbone, train_classifier, test
from models.models import Classifier

def lp(config: dict):
    # Seed + deterministic knobs
    deterministic = True
    set_seed(config["seed"], deterministic=deterministic, benchmark=False if deterministic else True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    lp_start = time.time()

    train_loader, val_loader, test_loader = get_lp_dataloaders(
        batch_size=config["lp_batch_size"],
        workers=config["workers"],
        normalization=config["ssl_normalization"],
        seed=config["seed"],
    )
    print(f"Number of LP training samples: {len(train_loader.dataset)}")

    if "ckpt_path" in config:
        ckpt_path = config["ckpt_path"]
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
        ) if "checkpoint" not in config else config["checkpoint"]
        
        ckpt_path = (
            f"{ckpt_dir}/checkpoint_{int(config['ssl_epochs']):04d}.pth.tar"
        )
    print(f"Loaded backbone weights from {ckpt_path}")
    backbone = construct_backbone(
        arch=config["arch"],
        ckpt_path=ckpt_path,
        model_weights=None,
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
    
    if ckpt_path is not None:
        ckpt_dir = ckpt_path.split('/checkpoint_')[0]
        out_name = ckpt_dir.split('checkpoints_')[-1]
        out_dir = f"./results/linear_probing_{out_name}_{config['lp_lr']}_{config['lp_wd']}_{config['lp_epochs']}"
    else:
        out_dir = (
            f"./results/{config['arch']}_"
            f"{config['ssl_lr']}_"
            f"{config['ssl_wd']}_"
            f"{config['ssl_epochs']}_"
            f"{config['ssl_da_method']}_"
            f"{config['ssl_normalization']}_"
            f"{config['ssl_flips']}_"
            f"{config['ssl_rotations']}_"
            f"{config['ssl_crop_min']}_"
            f"{config['ssl_artifacts']}_"
            f"{config['ssl_noise']}_"
            f"{config['lp_lr']}_"
            f"{config['lp_wd']}_"
            f"{config['lp_epochs']}"
        )
    
    classifier_weights, best_val_acc, best_epoch = train_classifier(
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

    test(
        model, test_loader, device, ckpt_path=None, model_weights=classifier_weights, out_dir=out_dir
    )

    lp_time = time.time() - lp_start
    print(f"LP training completed in {lp_time:.2f} seconds.\n")

    # Cleanup
    del backbone, model, train_loader, val_loader, test_loader, classifier_weights
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/config.yaml")
    parser.add_argument("-ckpt", "--ckpt_path", type=str, default=None, help="Path to checkpoint to load model weights from") # optionally specify checkpoint path
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.ckpt_path is not None:
        config["ckpt_path"] = args.ckpt_path

    lp(config)
