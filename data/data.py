from datasets import load_dataset, ClassLabel
import datasets
import torchvision.transforms as transforms
from utils.utils import (GaussianBlur, Solarize, AdditiveGaussianNoise, PoissonNoise, CosmicRayArtifacts)
                         
import torch
import random
import numpy as np

ds = load_dataset("matthieulel/galaxy10_decals")

########################## Common transforms and functions ##########################
MEAN = [0.1678, 0.1629, 0.1592] # calculated based on dataset with the black background => since it's mostly black, the mean is very low
STD  = [0.1225, 0.1134, 0.1062] # calculated based on dataset with the black background => since it's mostly black, the std is very low

FG_MEAN = [0.5563723577581228, 0.5466399687071238, 0.5065107300061998] # calculated based on dataset using otsu thresholding
FG_STD  = [0.20517936480092377, 0.1763141906401364, 0.16977451864162768] # calculated based on dataset using otsu thresholding

IN_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
IN_STD  = [0.229, 0.224, 0.225] # ImageNet std

ga_normalize = transforms.Normalize(mean=MEAN,
                                    std=STD)  # calculated based on dataset

gafg_normalize = transforms.Normalize(mean=FG_MEAN,
                                    std=FG_STD)  # calculated based on dataset

in_normalize = transforms.Normalize(mean=IN_MEAN,
                                     std=IN_STD)

def collate_moco(batch):
    x1 = torch.stack([b["x1"] for b in batch])
    x2 = torch.stack([b["x2"] for b in batch])
    original_images = torch.stack([b["ori_image"] for b in batch])
    return [x1, x2], original_images

def collate_fn(batch): 
    x = torch.stack([b["image"] for b in batch]) 
    y  = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return x, y

############################# For SSL #############################
ssl_ds = ds.cast_column("image", datasets.Image(decode=True))

# follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
# from original MoCo V3 code

def build_byol_transforms(crop_min: float, 
                          img_size: int = 224,
                          normalization = "IN"):
    
    if normalization.upper() == "IN":
        norm = in_normalize
    elif normalization.upper() == "GA":
        norm = ga_normalize
    elif normalization.upper() == "GAFG":
        norm = gafg_normalize
    else:
        raise ValueError(f"normalization {normalization} not recognized. Use 'IN', 'GA', or 'GAFG'.")
    
    aug1 = [
        transforms.RandomResizedCrop(img_size, scale=(crop_min, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur((.1, 2.))], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ]
    aug2 = [
        transforms.RandomResizedCrop(img_size, scale=(crop_min, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur((.1, 2.))], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm
    ]
    return transforms.Compose(aug1), transforms.Compose(aug2)

def build_galaxy_aware_transforms(
    crop_min: float,
    img_size: int = 224,
    flips: bool = True,
    rotations: bool = True,
    artifacts: bool = True,
    noise: bool = True,
    normalization: str = "IN"
):
    """
    Constructs two-view data augmentation pipelines tailored for galaxy images.

    Args:
        crop_min: Minimum scale for RandomResizedCrop.
        img_size: Target image size.
        flips: If True, applies random horizontal and vertical flips.
        rotations: If True, applies random 180-degree rotations.
        psf_mode: The PSF model to use ('gaussian', 'moffat', or 'none').
        normalization: The normalization scheme to use ('IN' or 'GA').
    """
    # --- Base transforms applied to both strong and weak views ---
    base_transforms = []
    if rotations:
        base_transforms.append(transforms.RandomRotation(degrees=180, fill=0))
    base_transforms.append(transforms.RandomResizedCrop(img_size, scale=(crop_min, 1.0), ratio=(0.95, 1.05)))
    if flips:
        base_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ])
    
    # Photometric transforms are also shared
    base_transforms.extend([
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.15, hue=0.02)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.05)
    ])

    # --- Select the PSF blur model based on the switch ---
    psf_blur_transform = GaussianBlur(sigma=(0.5, 1.5))

    # Choose normalization
    if normalization.upper() == "IN":
        norm_transform = in_normalize
    elif normalization.upper() == "GA":
        norm_transform = ga_normalize
    elif normalization.upper() == "GAFG":
        norm_transform = gafg_normalize
    else:
        raise ValueError(f"normalization {normalization} not recognized. Use 'IN', 'GA', or 'GAFG'.")

    # --- Define transforms specific to each view ---
    
    # Strong view (View 1)
    strong_specific_transforms = []
    if artifacts:
        strong_specific_transforms.append(CosmicRayArtifacts(p_apply=0.15, prob_per_pixel=0.001, max_streaks=2))

    strong_specific_transforms.append(transforms.RandomApply([psf_blur_transform], p=0.9))
    strong_specific_transforms.append(transforms.ToTensor())
    if noise:
        strong_specific_transforms.extend([
            transforms.RandomApply([PoissonNoise(scale_range=(500.0, 2000.0))], p=0.8),
            transforms.RandomApply([AdditiveGaussianNoise(std=0.02)], p=0.3),
        ])
        
    strong_specific_transforms.append(norm_transform)
    
    # Weak view (View 2)
    weak_specific_transforms = []
    if artifacts:
        weak_specific_transforms.append(CosmicRayArtifacts(p_apply=0.05, prob_per_pixel=0.0005, max_streaks=1))
    
    weak_specific_transforms.append(transforms.RandomApply([psf_blur_transform], p=0.15))

    # No additional noise for the weak view
    weak_specific_transforms.extend([
        transforms.ToTensor(),
        norm_transform
    ])

    # Combine base and specific transforms
    strong_pipeline = transforms.Compose(base_transforms + strong_specific_transforms)
    weak_pipeline = transforms.Compose(base_transforms + weak_specific_transforms)

    return strong_pipeline, weak_pipeline

def get_ssl_dataloader(batch_size, 
                       workers,
                       da_method = "BYOL",
                       crop_min = 0.5,
                       flips = True,
                       rotations = True,
                       artifacts = True,
                       noise = True,
                       normalization = "IN",
                       seed = 2952):

    if normalization.upper() == "IN":
        norm = in_normalize
    elif normalization.upper() == "GA":
        norm = ga_normalize
    elif normalization.upper() == "GAFG":
        norm = gafg_normalize
        
    if da_method.upper() == "BYOL":
        base_transform1, base_transform2 = build_byol_transforms(
            crop_min=crop_min,
            img_size=224,
            normalization=normalization
        )
    elif da_method.upper() == "GA":
        base_transform1, base_transform2 = build_galaxy_aware_transforms(
            crop_min=crop_min,
            img_size=224,
            flips=flips,
            rotations=rotations,
            artifacts=artifacts,
            noise=noise,
            normalization=normalization
        )
    else:
        raise ValueError(f"da_method {da_method} not recognized. Use 'BYOL' or 'GA'.")

    def apply_ssl_transforms(batch):
        x1_list, x2_list, original_imgs = [], [], []
        for img in batch["image"]:
            x1, x2 = base_transform1(img), base_transform2(img)
            x1_list.append(x1)
            x2_list.append(x2)
            
            # convert img to tensor for visualization later
            # (we could also add a third transform for this)
            img = transforms.ToTensor()(img)
            img = norm(img)
            
            original_imgs.append(img)
            
        batch["x1"] = x1_list
        batch["x2"] = x2_list
        batch["ori_image"] = original_imgs

        return batch

    ssl_train_dataset = ssl_ds["train"].with_transform(apply_ssl_transforms)

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return torch.utils.data.DataLoader(
        ssl_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g, persistent_workers=False,
        collate_fn=collate_moco
    )

# ONLY FOR VISUALIZATION PURPOSES
def get_ssl_test_dataloader(batch_size, workers, normalization = "IN", seed = 2952):

    if normalization.upper() == "IN":
        norm = in_normalize
    elif normalization.upper() == "GA":
        norm = ga_normalize
    elif normalization.upper() == "GAFG":
        norm = gafg_normalize
    
    lp_normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm
    ])
    
    def apply_lp_transforms(batch): 
        x_list = []
        for img in batch["image"]: 
            x = lp_normalize(img)
            x_list.append(x)

        batch = {"image": torch.stack(x_list), "label": batch["label"]}
        return batch 
    
    ssl_test_dataset = ssl_ds["test"].with_transform(apply_lp_transforms) # use lp transforms for visualization

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    return torch.utils.data.DataLoader(
        ssl_test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=False
    )

############################ For linear probe ##########################
# Infer #classes and cast
labels = ds["train"]["label"]
n_classes = int(max(labels)) + 1
ds = ds.cast_column("label", ClassLabel(num_classes=n_classes))

def get_lp_dataloaders(batch_size, 
                       workers, 
                       val_size=0.2,
                       normalization="IN",
                       seed: int = 2952):

    # --- choose normalization (unchanged) ---
    if normalization.upper() == "IN":
        norm = in_normalize
    elif normalization.upper() == "GA":
        norm = ga_normalize
    elif normalization.upper() == "GAFG":
        norm = gafg_normalize
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    lp_normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm
    ])

    def apply_lp_transforms(batch):
        x_list = [lp_normalize(img) for img in batch["image"]]
        return {"image": torch.stack(x_list), "label": batch["label"]}

    # --- input checks (fix the unreachable branch) ---
    if not (0 < val_size < 1):
        raise ValueError("val_size should be between 0 and 1 (exclusive).")

    # --- deterministic split ---
    split_ds = ds["train"].train_test_split(
        test_size=val_size,
        stratify_by_column="label",
        seed=seed,
    )
    train_ds = split_ds["train"]
    val_ds   = split_ds["test"]
    test_ds  = ds["test"]

    train_dataset = train_ds.with_transform(apply_lp_transforms)
    val_dataset   = val_ds.with_transform(apply_lp_transforms)
    test_dataset  = test_ds.with_transform(apply_lp_transforms)

    # --- seed workers & shuffle deterministically ---
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=False,
    )

    return train_loader, val_loader, test_loader
