from data.data import get_lp_dataloaders
from utils.utils import set_seed, enable_fast_kernels
from scripts.lp_script import construct_backbone, train_classifier, test
import torch
from models.models import Classifier
from timm import create_model
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Floor: Randomly initialize a backbone and train a classifier on top of it
SEED = 2952
set_seed(SEED, deterministic=True, benchmark=True) # benchmark=False for reproducibility
enable_fast_kernels()
normalization = "IN"  # "IN" or "None"

######## HYPERPARAMETERS ########
batch_size = 256
workers = 32
epochs=100
lr=3e-3
wd=0.05
out_dir=f"results/baselines_{normalization}_vit_plain_test/random_init"
arch = 'vit_s'
use_amp=True
label_smoothing=0.05

######## PREPARE DATA, MODEL ########
train_loader, val_loader, test_loader = get_lp_dataloaders(batch_size=batch_size, 
                                                           workers=workers, 
                                                           normalization=normalization, 
                                                           seed=SEED)
backbone = construct_backbone(arch=arch, device=device, moco_style=False)
model = Classifier(backbone=backbone, num_classes=10, requires_grad=False, eval_mode=True, moco_style=False)

######## TRAINING ########
train_classifier(epochs=epochs,
                 model=model,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 lr=lr,
                 wd=wd,
                 device=device,
                 out_dir=out_dir,
                 use_amp=use_amp,
                 label_smoothing=label_smoothing,
                 print_freq=5)

######## TESTING ########
best_model_path = f"{out_dir}/best.ckpt"
test(model, test_loader, device, ckpt_path=best_model_path, out_dir=out_dir)

del model, backbone
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# 2. Supervised: Make backbone trainable and train the classifier End-to-End
######## HYPERPARAMETERS ########
batch_size = 256
workers = 32
epochs=100
lr=3e-3
wd=0.05
backbone_lr=3e-4
backbone_wd=0.05
out_dir=f"results/baselines_{normalization}_vit_plain/e2e_training"
arch = 'vit_s'
use_amp=True
label_smoothing=0.05

new_backbone = construct_backbone(arch=arch, device=device, moco_style=False)
model = Classifier(backbone=new_backbone, num_classes=10, requires_grad=True, eval_mode=False, moco_style=False)

######## TRAINING ########
train_classifier(epochs=epochs,
                 model=model,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 lr=lr,
                 wd=wd,
                 device=device,
                 out_dir=out_dir,
                 use_amp=use_amp,
                 backbone_lr=backbone_lr,
                 backbone_wd=backbone_wd,
                 label_smoothing=label_smoothing,
                 print_freq=5)

######## TESTING ########
best_model_path = f"{out_dir}/best.ckpt"
test(model, test_loader, device, ckpt_path=best_model_path, out_dir=out_dir)
del new_backbone, model
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# 3. Pretrained Using ImageNet ViT weights + Linear Probing
######## HYPERPARAMETERS ########
if normalization == "IN":
    batch_size = 256
    workers = 32
    epochs=100
    lr=3e-3
    wd=0.05
    out_dir=f"results/baselines_{normalization}_vit_plain/imagenet_pretrained"
    arch   = "vit_small_patch16_224"   # ViT-Small, ImageNet-1k pretrained
    use_amp=True
    label_smoothing=0.05

    ######## MODEL ########
    # Load ViT backbone with ImageNet pretrained weights
    backbone = create_model(arch, pretrained=True, num_classes=0)  # num_classes=0 removes head
    backbone.reset_classifier(num_classes=0, global_pool="token")

    # Freeze backbone (linear probe)
    for p in backbone.parameters():
        p.requires_grad = False

    # Add linear classifier
    model = Classifier(backbone=backbone, num_classes=10,
                        requires_grad=False, 
                        eval_mode=True,
                        moco_style=False)

    ######## TRAINING ########
    train_classifier(epochs=epochs,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    lr=lr,
                    wd=wd,
                    device=device,
                    out_dir=out_dir,
                    use_amp=use_amp,
                    label_smoothing=label_smoothing,
                    print_freq=5)

    ######## TESTING ########
    best_model_path = f"{out_dir}/best.ckpt"
    test(model, test_loader, device, ckpt_path=best_model_path, out_dir=out_dir)

    del model, backbone, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

