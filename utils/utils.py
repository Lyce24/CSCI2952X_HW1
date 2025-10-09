from PIL import ImageFilter, ImageOps, Image, ImageDraw
import random
import torch
import csv
from pathlib import Path
import os
import tempfile
import random
import numpy as np
from pathlib import Path
from torch.amp import GradScaler
import torch.nn as nn
import shutil
from torchvision import transforms

def set_seed(seed: int = 42, deterministic: bool = False, benchmark: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Fast preset (allows small drift)
    torch.backends.cudnn.deterministic = deterministic   # False for speed
    torch.backends.cudnn.benchmark = benchmark           # True for autotuned fast kernels

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class AdditiveGaussianNoise(object):
    def __init__(self, std=0.02):
        self.std = std
    def __call__(self, x):
        if not torch.is_tensor(x):
            raise TypeError("AdditiveGaussianNoise expects a tensor after ToTensor()")
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)
    
class PoissonNoise:
    """
    Photon-counting (Poisson) noise. Works on tensors in [0,1].
    We simulate a 'counts scale': counts = x * scale, sample Poisson, then divide back.
    """
    def __init__(self, scale_range=(500.0, 2000.0)):
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x is Tensor [C,H,W] in [0,1]
        if not torch.is_tensor(x):
            raise TypeError("PoissonNoise expects a Tensor after ToTensor().")
        scale = float(random.uniform(*self.scale_range))
        counts = (x * scale).clamp(min=0.0)
        noisy = torch.poisson(counts) / scale
        return noisy.clamp(0.0, 1.0)

class CosmicRayArtifacts:
    """
    Rare bright spots and occasional streaks (on PIL image), then return PIL.
    - prob_per_pixel controls random 'hot pixel' mask (like your function)
    - streaks add 0-2 bright lines
    """
    def __init__(self,
                 p_apply=0.15,
                 prob_per_pixel=0.001,
                 max_streaks=2,
                 spot_intensity=(0.8, 1.0),
                 streak_intensity=(0.7, 1.0),
                 streak_thickness=(1, 2)):
        self.p_apply = p_apply
        self.prob_per_pixel = prob_per_pixel
        self.max_streaks = max_streaks
        self.spot_intensity = spot_intensity
        self.streak_intensity = streak_intensity
        self.streak_thickness = streak_thickness

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p_apply:
            return img

        # --- bright spots via your numpy-mask method ---
        arr = np.asarray(img).astype(np.float32) / 255.0
        h, w = arr.shape[:2]
        mask = (np.random.random((h, w)) < self.prob_per_pixel).astype(np.float32)
        # same mask for all channels
        spot_val = np.random.uniform(*self.spot_intensity, size=(h, w, 1)).astype(np.float32)
        arr = np.clip(arr * (1 - mask[..., None]) + spot_val * mask[..., None], 0.0, 1.0)
        img = Image.fromarray((arr * 255.0).astype(np.uint8))

        # --- occasional short streaks ---
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(0, self.max_streaks)):
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            # keep streaks short-ish
            x2 = int(np.clip(x1 + np.random.randint(-w // 6, w // 6), 0, w - 1))
            y2 = int(np.clip(y1 + np.random.randint(-h // 6, h // 6), 0, h - 1))
            val = int(255 * random.uniform(*self.streak_intensity))
            t = random.randint(*self.streak_thickness)
            draw.line([(x1, y1), (x2, y2)], fill=(val, val, val), width=t)

        return img

class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
     
class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

class CSVLogger:
    def __init__(self, filepath: str, fieldnames: list[str], overwrite: bool = True):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._init_file(overwrite=overwrite)

    def _init_file(self, overwrite: bool = False):
        """
        Create file with header.
        If overwrite=True, truncate file; else, create if missing and keep existing content.
        """
        mode = "w" if overwrite or not self.filepath.exists() else "a"
        new_file = (mode == "w") or (not self.filepath.exists())
        with self.filepath.open(mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if new_file:
                writer.writeheader()

    def log(self, row: dict):
        clean = {k: row.get(k, "") for k in self.fieldnames}
        with self.filepath.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean)

# -------------------------
# Checkpointing
# -------------------------
def atomic_save(obj: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp_name = tmp.name
        torch.save(obj, tmp_name)
    os.replace(tmp_name, str(path))  # atomic on POSIX/NTFS

def save_checkpoint(state: dict, is_best: bool, filename='checkpoint.pth.tar'):
    atomic_save(state, filename)
    if is_best:
        best_path = Path(filename).with_name('model_best.pth.tar')
        shutil.copyfile(filename, best_path)

def load_checkpoint_if_any(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    resume_path: str | None,
    device: torch.device
):
    """
    Returns (start_epoch, global_step). If resume_path is None or missing keys, continues gracefully.
    """
    start_epoch, global_step = 0, 0
    if not resume_path:
        return start_epoch, global_step

    ckpt_path = Path(resume_path)
    if not ckpt_path.is_file():
        print(f"[resume] No checkpoint found at {resume_path}")
        return start_epoch, global_step

    print(f"[resume] Loading checkpoint: {resume_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # model
    if "state_dict" in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing or unexpected:
            print(f"[resume] model missing keys: {missing}\n[resume] unexpected keys: {unexpected}")

    # optimizer
    if optimizer is not None and "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            print(f"[resume] optimizer state load failed: {e}")

    # scaler
    if scaler is not None and "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as e:
            print(f"[resume] scaler state load failed: {e}")

    # book-keeping
    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))

    # RNG
    if "rng_state" in checkpoint:
        try:
            torch.set_rng_state(checkpoint["rng_state"])
        except Exception:
            pass
    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        except Exception:
            pass
    if "numpy_rng_state" in checkpoint:
        try:
            np.random.set_state(checkpoint["numpy_rng_state"])
        except Exception:
            pass
    if "python_random_state" in checkpoint:
        try:
            random.setstate(checkpoint["python_random_state"])
        except Exception:
            pass

    print(f"[resume] Resumed at epoch={start_epoch}, global_step={global_step}")
    return start_epoch, global_step