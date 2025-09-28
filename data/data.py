from datasets import load_dataset, ClassLabel
import datasets
import torchvision.transforms as transforms
from utils.utils import TwoCropsTransform, GaussianBlur, Solarize, set_seed
import torch

SEED = 2952
set_seed(SEED, deterministic=True, benchmark=False) # benchmark=False for reproducibility
crop_min = .2
ds = load_dataset("matthieulel/galaxy10_decals")

# utils
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

lp_normalize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def apply_ssl_transforms(batch):
    x1_list, x2_list, original_imgs = [], [], []
    for img in batch["image"]:
        x1, x2 = twocrops(img)         # img is a PIL.Image
        x1_list.append(x1)
        x2_list.append(x2)
        
        # convert img to tensor for visualization later
        # (we could also add a third transform for this)
        img = transforms.ToTensor()(img)
        img = normalize(img)
        
        original_imgs.append(img)
        
    batch["x1"] = x1_list
    batch["x2"] = x2_list
    batch["ori_image"] = original_imgs

    return batch

def collate_moco(batch):
    x1 = torch.stack([b["x1"] for b in batch])
    x2 = torch.stack([b["x2"] for b in batch])
    original_images = torch.stack([b["ori_image"] for b in batch])
    return [x1, x2], original_images

def apply_lp_transforms(batch): 
    x_list = []
    for img in batch["image"]: 
        x = lp_normalize(img)
        x_list.append(x)

    batch = {"image": torch.stack(x_list), "label": batch["label"]}
    return batch 

def collate_fn(batch): 
    x = torch.stack([b["image"] for b in batch]) 
    y  = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return x, y

############################# For SSL #############################
ssl_ds = ds.cast_column("image", datasets.Image(decode=True))

# follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
augmentation1 = [
    transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

augmentation2 = [
    transforms.RandomResizedCrop(224, scale=(crop_min, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
    transforms.RandomApply([Solarize()], p=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

twocrops = TwoCropsTransform(
    transforms.Compose(augmentation1),
    transforms.Compose(augmentation2)
)

ssl_train_dataset = ssl_ds["train"].with_transform(apply_ssl_transforms)
# For SSL, there is no validation set since we do not use labels
ssl_test_dataset = ssl_ds["test"].with_transform(apply_lp_transforms) # use lp transforms for visualization

def get_ssl_dataloader(batch_size, workers):
    return torch.utils.data.DataLoader(
        ssl_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        collate_fn=collate_moco
    )
    
def get_ssl_test_dataloader(batch_size, workers):
    return torch.utils.data.DataLoader(
        ssl_test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn
    )

############################ For linear probe ##########################
# Infer #classes and cast
labels = ds["train"]["label"]
n_classes = int(max(labels)) + 1
ds = ds.cast_column("label", ClassLabel(num_classes=n_classes))

def get_lp_dataloaders(batch_size, workers, val_size = 0.2):
    if val_size <= 0 or val_size >= 1:
        raise ValueError("val_size should be between 0 and 1")
    elif val_size == 0:
        print("No validation set will be used.")
        train_ds = ds["train"]
        test_ds = ds["test"]
        val_ds = None
    else:
        split_ds = ds["train"].train_test_split(test_size=val_size, seed=SEED,
                                                stratify_by_column="label")
        train_ds = split_ds["train"]
        val_ds   = split_ds["test"]
        test_ds  = ds["test"]

    train_dataset = train_ds.with_transform(apply_lp_transforms)
    val_dataset   = val_ds.with_transform(apply_lp_transforms) if val_ds is not None else None
    test_dataset  = test_ds.with_transform(apply_lp_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn,
    ) if val_dataset is not None else None
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader