"""
One-batch diagnostic for loss=0 with channels=window.

Run from the image/ directory:
    cd image
    python ../Custom/debug_rsna_batch.py

Prints: image stats, label distribution, logit stats, and loss for the
first training batch so we can see exactly where things go wrong.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../image")

import torch
import numpy as np
from omegaconf import OmegaConf
import hydra
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# ── 1. Build config (same as run_classify.py would) ─────────────────────────
GlobalHydra.instance().clear()
config_dir = os.path.abspath("radfusion3/configs")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(
        config_name="classify",
        overrides=[
            "model=resnext_101_imagenet",
            "dataset=rsna",
            "dataset.transform.final_size=224",
            "dataset.batch_size=8",   # small for debug
        ],
    )

print("=" * 70)
print("Config summary")
print(f"  model         : {cfg.model.model_name}")
print(f"  channels      : {cfg.dataset.transform.channels}")
print(f"  precision     : {cfg.trainer.precision}")
print(f"  weighted_sample: {cfg.dataset.weighted_sample}")
print("=" * 70)

# ── 2. Build dataset (train split) ──────────────────────────────────────────
import radfusion3
from radfusion3 import builder

DatasetClass = builder.build_dataset(cfg)
transform    = builder.build_transformation(cfg, "train")
ds           = DatasetClass(cfg, split="train", transform=transform)

print(f"\nDataset length : {len(ds)}")
labels = np.array(ds.labels)
print(f"Label distribution: 0={int((labels==0).sum())}  1={int((labels==1).sum())}")
print(f"Positive rate     : {labels.mean():.3f}")

# ── 3. Pull a batch ──────────────────────────────────────────────────────────
from torch.utils.data import DataLoader

sampler = ds.get_sampler() if cfg.dataset.weighted_sample else None
dl = DataLoader(
    ds,
    batch_size=8,
    sampler=sampler,
    shuffle=(sampler is None),
    num_workers=0,   # single-process for clean printing
)

x, y, mask, ids = next(iter(dl))

print("\n── Batch stats ─────────────────────────────────────────────────────────")
print(f"  x shape : {x.shape}  dtype={x.dtype}")
print(f"  x range : [{x.min():.4f}, {x.max():.4f}]  mean={x.mean():.4f}")
print(f"  y values: {y.squeeze().tolist()}")
print(f"  any NaN in x: {x.isnan().any().item()}")

# ── 4. Forward pass ──────────────────────────────────────────────────────────
model_lightning = builder.build_lightning_model(cfg)
model_2d        = model_lightning.model.eval()

with torch.no_grad():
    logit, features = model_2d(x, get_features=True)

print("\n── Model output stats ──────────────────────────────────────────────────")
print(f"  logit  : {logit.squeeze().tolist()}")
print(f"  logit range : [{logit.min():.4f}, {logit.max():.4f}]")
print(f"  feature range: [{features.min():.4f}, {features.max():.4f}]")
print(f"  any NaN in logit: {logit.isnan().any().item()}")

# ── 5. Loss ──────────────────────────────────────────────────────────────────
loss_fn = builder.build_loss(cfg)
loss    = loss_fn(logit, y)
print(f"\n── Loss ────────────────────────────────────────────────────────────────")
print(f"  BCEWithLogitsLoss : {loss.item():.6f}")
print(f"  Expected (random) : ~0.693")

if loss.item() < 0.001:
    print("\n  *** LOSS IS NEAR ZERO — diagnosing further ***")
    print(f"  sigmoid(logit) = {torch.sigmoid(logit).squeeze().tolist()}")
    all_zero_labels = (y == 0).all()
    all_one_labels  = (y == 1).all()
    print(f"  All labels zero? {all_zero_labels.item()}")
    print(f"  All labels one?  {all_one_labels.item()}")
    very_neg_logits = (logit < -10).all()
    very_pos_logits = (logit >  10).all()
    print(f"  Logits all very negative? {very_neg_logits.item()}")
    print(f"  Logits all very positive? {very_pos_logits.item()}")
    print()
    print("  Root-cause interpretation:")
    if all_zero_labels and very_neg_logits:
        print("  -> Labels all 0 + logits very negative => weighted sampler not working")
    elif all_zero_labels:
        print("  -> Labels all 0 (sampler broken?), logits are:", logit.squeeze().tolist())
    elif all_one_labels and very_pos_logits:
        print("  -> Labels all 1 + logits very positive => model is trivially correct")
    elif logit.isnan().any():
        print("  -> NaN in logits => activation explosion or bad input")
    else:
        print("  -> Unknown; see raw values above")

print("=" * 70)
