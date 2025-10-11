from data.data import get_lp_dataloaders
from utils.utils import set_seed
from scripts.lp_script import construct_backbone, train_classifier, test
import torch
from models.models import Classifier
from timm import create_model
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Floor: Randomly initialize a backbone and train a classifier on top of it
SEED = 2952
normalization = "IN"  # "IN" or "None"

set_seed(SEED, deterministic=True, benchmark=False) # benchmark=False for reproducibility

lr=3e-3
wd=0.05
output_dir = f"results/baselines_{normalization}_vit_plain_{lr}_{wd}"
print(f"Running baseline with lr={lr} and wd={wd}")

os.makedirs(output_dir, exist_ok=True)
print(f"Output dir: {output_dir}")

######## HYPERPARAMETERS ########
batch_size = 256
workers = 36
epochs=100
out_dir=f"{output_dir}/random_init"
arch = 'vit_s'
use_amp=True
label_smoothing=0.05

######## PREPARE DATA, MODEL ########
train_loader, val_loader, test_loader = get_lp_dataloaders(batch_size=batch_size, 
                                                        workers=workers, 
                                                        normalization=normalization, 
                                                        seed=SEED)
backbone = construct_backbone(arch=arch, device=device, moco_style=False)

# Freeze backbone (linear probe)
for p in backbone.parameters():
    p.requires_grad = False
    
model = Classifier(backbone=backbone, num_classes=10, requires_grad=False, eval_mode=True, moco_style=False)

######## TRAINING ########
train_classifier(epochs=epochs,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                opt="adamw",
                lr=lr,
                wd=wd,
                device=device,
                out_dir=out_dir,
                use_amp=use_amp,
                label_smoothing=label_smoothing)

######## TESTING ########
best_model_path = f"{out_dir}/best.ckpt"
test(model, test_loader, device, ckpt_path=best_model_path, out_dir=out_dir)

del model, backbone
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Pretrained Using ImageNet ViT weights + Linear Probing
######## HYPERPARAMETERS ########
out_dir=f"{output_dir}/imagenet_pretrained"
arch   = "vit_small_patch16_224"   # ViT-Small, ImageNet-1k pretrained

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
                opt="adamw",
                lr=lr,
                wd=wd,
                device=device,
                out_dir=out_dir,
                use_amp=use_amp,
                label_smoothing=label_smoothing)

######## TESTING ########
best_model_path = f"{out_dir}/best.ckpt"
test(model, test_loader, device, ckpt_path=best_model_path, out_dir=out_dir)

del model, backbone
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Supervised: Make backbone trainable and train the classifier End-to-End
######## HYPERPARAMETERS ########
backbone_lr = 5e-4
backbone_wd = 0.05
out_dir=f"{output_dir}/e2e_training"

new_backbone  = create_model(arch, pretrained=False, num_classes=0)  # num_classes=0 removes head
new_backbone.reset_classifier(num_classes=0, global_pool="token")
model = Classifier(backbone=new_backbone, num_classes=10, requires_grad=True, eval_mode=False, moco_style=False)

######## TRAINING ########
train_classifier(epochs=epochs,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=lr,
                wd=wd,
                opt="adamw",
                device=device,
                out_dir=out_dir,
                use_amp=use_amp,
                label_smoothing=label_smoothing,
                backbone_lr=backbone_lr,
                backbone_wd=backbone_wd)

######## TESTING ########
best_model_path = f"{out_dir}/best.ckpt"
test(model, test_loader, device, ckpt_path=best_model_path, out_dir=out_dir)
del new_backbone, model, train_loader, val_loader, test_loader
torch.cuda.empty_cache()
torch.cuda.ipc_collect()