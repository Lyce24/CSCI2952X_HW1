# Running Guide

## Environment Requirements
Ensure the following dependencies are installed:

- Python == 3.10.18  
- torch  
- torchvision  
- timm  
- datasets  
- scikit-learn  

## Running Modes

### 1. Full Pipeline
Configure parameters in `configs/config.yaml`, then run:
```bash
python main.py
````

This runs both the SSL (pretraining) and LP (linear probing) stages sequentially.

---

### 2. Stage-wise Execution

Configure parameters in `configs/config.yaml`.

**SSL Stage:**

```bash
python phase1_ssl.py --ckpt_dir [dir-to-save-checkpoints]  # optional
```

**LP Stage:**

```bash
python phase2_lp.py --ckpt_path [path-to-saved-model]  # optional
```

Both paths are optional — checkpoints and outputs are automatically managed based on your configuration file.

---

### 3. Experiment Mode (Hyperparameter Tuning / Ablation Study)

In this mode, you do **not** modify `config.yaml`.
Instead, open `exp.py` and define the `EXP_PAIRS` variable in the following order:

```
ssl_lr, ssl_wd, da_method, norm, flips, rotations, artifacts, noise, crop_min
```

Then run:

```bash
python exp.py
```

Results (including summaries) will be automatically saved under:

```
results/exp/
```

---

### 4. Baseline Runs

To run a baseline model:

1. Open `baseline.py`
2. Adjust the learning rate (LR) and weight decay (WD)
3. Execute:

   ```bash
   python baseline.py
   ```

Results are automatically stored in:

```
results/baselines_IN_vit_plain_{LR}_{WD}
```

---

## Repository Structure

```
configs/
 └── config.yaml          # Parameter configuration for SSL and LP

data/
 └── data.py              # Dataset downloading, augmentations, and dataloaders

models/
 └── models.py            # MoCo v3 model and classifier (frozen backbone + linear head)

notebooks/
 ├── clustering_analysis.ipynb  # Visualize UMAP embeddings for the 10-class model
 └── da_vis.ipynb               # Visualize BYOL vs Galaxy-Aware data augmentations

results/
 └── ...                 # Contains all experiment outputs and logs

scripts/
 ├── lp_script.py         # Linear probing training & evaluation
 └── ssl_script.py        # MoCo v3 pretraining script

utils/
 └── utils.py             # Utility functions (Gaussian blurring, checkpoint saving, etc.)

baseline.py               # Baseline experiment runner
exp.py                    # Hyperparameter tuning & ablation study script
main.py                   # Full MoCo v3 training pipeline
phase1_ssl.py             # Run SSL pretraining only
phase2_lp.py              # Run LP training only
```
