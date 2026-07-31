#!/usr/bin/env python
"""
extract_1d_embeddings.py
------------------------
Extract study-level embeddings from a trained 1-D sequence model (`Model1D`),
i.e. the vector immediately before the classification head.

Why a separate script rather than a patch to radfusion3
-------------------------------------------------------
`ClassificationLightningModel.shared_step` already computes the embedding --

    logit, features = self.model(x, mask=mask, get_features=True)

-- and then discards `features`. Rather than modify upstream code, this script
re-runs inference post-hoc from a saved checkpoint. That means:

  * it works on any run that has already finished, with no retraining;
  * training behaviour is untouched;
  * extraction is deterministic and repeatable.

What gets saved
---------------
For each requested split, one row per study:

  embeddings  (N, D) float32   D = 256 for the swept configs
                               (hidden_size 128, bidirectional, max/mean/attention)
                               D = 512 if aggregation is 'attention+max'
  ids         (N,)   str       patient_datetime / SeriesInstanceUID
  labels      (N,)   float32
  probs       (N,)   float32   sigmoid(logit), for sanity-checking against the run
  split       (N,)   str

Written as `<out>/embeddings.npz` plus `<out>/index.csv` for easy joining
against the EHR side on patient/impression id.

Determinism
-----------
Two things are forced regardless of config:

  1. `sample_strategy = "fix"` -- 'random' would draw different slices per call,
     so the same study would embed differently on every run.
  2. Dataloaders are built here with `shuffle=False` and **no sampler**. The
     DataModule's `train_dataloader()` applies a `WeightedRandomSampler` with
     replacement, which would duplicate some studies and omit others.

Usage
-----
    python Custom/extract_1d_embeddings.py --run-dir <path to classify_<target>_<ts>>
    python Custom/extract_1d_embeddings.py --run-dir ... --ckpt last.ckpt --splits test
    python Custom/extract_1d_embeddings.py --run-dir ... --out ~/embeddings/pe
"""

import argparse
import csv
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent          # Custom/
REPO_ROOT = SCRIPT_DIR.parent                          # INSPECT_custom_data_preprocessing/
IMAGE_DIR = REPO_ROOT / "image"                        # contains the radfusion3 package

SPLIT_ALIASES = {"val": "valid", "validation": "valid"}


# ---------------------------------------------------------------------------
# Config / checkpoint discovery
# ---------------------------------------------------------------------------

def load_config(run_dir: Path):
    """Prefer the pickled OmegaConf written by the test epoch; fall back to
    Hydra's own snapshot two levels up from the exp directory."""
    from omegaconf import OmegaConf

    pkl = run_dir / "config.pkl"
    if pkl.is_file():
        with open(pkl, "rb") as f:
            cfg = pickle.load(f)
        print(f"config      : {pkl}")
        return cfg

    # <hydra_run>/outputs/<exp_name>/  ->  <hydra_run>/.hydra/config.yaml
    for cand in (run_dir.parent.parent / ".hydra" / "config.yaml",
                 run_dir.parent / ".hydra" / "config.yaml"):
        if cand.is_file():
            print(f"config      : {cand}  (config.pkl absent — test epoch may not have run)")
            return OmegaConf.load(cand)

    raise FileNotFoundError(
        f"No config.pkl in {run_dir} and no .hydra/config.yaml above it. "
        "Pass a run directory produced by run_classify.py."
    )


def find_checkpoint(run_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = run_dir / explicit
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    ckpts = [Path(p) for p in glob.glob(str(run_dir / "**" / "*.ckpt"), recursive=True)]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt under {run_dir}")

    # Prefer a metric-named best checkpoint over last.ckpt
    best = [p for p in ckpts if p.name != "last.ckpt"]
    chosen = max(best or ckpts, key=lambda p: p.stat().st_mtime)
    if len(ckpts) > 1:
        print(f"checkpoints : {len(ckpts)} found, using the most recent non-last")
    return chosen


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def build_loader(radfusion3, cfg, split, batch_size, num_workers):
    dataset_cls = radfusion3.builder.build_dataset(cfg)
    transform = radfusion3.builder.build_transformation(cfg, split)   # None for model_1d
    ds = dataset_cls(cfg, split=split, transform=transform)
    # shuffle=False and no sampler: preserve dataset order, one row per study
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=False,
                          drop_last=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def extract_split(inner_model, loader, device):
    feats, ids, labels, probs = [], [], [], []
    for batch in loader:
        x, y, mask, study = batch
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logit, feat = inner_model(x, mask=mask, get_features=True)

        feats.append(feat.float().cpu().numpy())
        probs.append(torch.sigmoid(logit.float()).cpu().numpy().reshape(-1))
        labels.append(np.asarray(y).reshape(-1).astype(np.float32))
        ids.extend([str(s) for s in study])
    if not feats:
        return None
    return (np.concatenate(feats, 0).astype(np.float32),
            np.array(ids, dtype=object),
            np.concatenate(labels, 0),
            np.concatenate(probs, 0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="classify_<target>_<timestamp> directory from run_classify.py")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path or name (default: most recent non-last .ckpt)")
    ap.add_argument("--splits", default="train,valid,test",
                    help="comma-separated (default: train,valid,test)")
    ap.add_argument("--out", default=None, help="output dir (default: <run-dir>/embeddings)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: cfg.dataset.batch_size")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"FAIL: not a directory: {run_dir}")
        return 1

    sys.path.insert(0, str(IMAGE_DIR))
    import radfusion3  # noqa: E402

    print("=" * 78)
    cfg = load_config(run_dir)
    ckpt_path = find_checkpoint(run_dir, args.ckpt)
    print(f"checkpoint  : {ckpt_path}")

    if cfg.model.type != "model_1d":
        print(f"FAIL: this script targets model_1d; config says '{cfg.model.type}'.")
        return 1

    # Determinism guards -----------------------------------------------------
    if getattr(cfg.dataset, "sample_strategy", "fix") != "fix":
        print(f"NOTE: forcing sample_strategy 'fix' (was "
              f"'{cfg.dataset.sample_strategy}') so embeddings are reproducible")
        cfg.dataset.sample_strategy = "fix"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = args.batch_size or int(cfg.dataset.batch_size)
    print(f"device      : {device}   batch_size: {batch_size}")
    print(f"target      : {cfg.dataset.target}")
    print(f"aggregation : {cfg.model.aggregation}   rnn: {cfg.model.seq_encoder.rnn_type}"
          f"   hidden: {cfg.model.seq_encoder.hidden_size}"
          f"   bidirectional: {cfg.model.seq_encoder.bidirectional}")

    # Model ------------------------------------------------------------------
    model = radfusion3.builder.build_lightning_model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"NOTE: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    model.eval().to(device)
    inner = model.model                                   # Model1D

    out_dir = Path(args.out).expanduser() if args.out else run_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Splits -----------------------------------------------------------------
    all_f, all_i, all_l, all_p, all_s = [], [], [], [], []
    for raw in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split = SPLIT_ALIASES.get(raw, raw)
        print("-" * 78)
        try:
            ds, loader = build_loader(radfusion3, cfg, split, batch_size, args.num_workers)
        except Exception as e:
            print(f"[{split}] could not build dataset: {e}")
            continue
        if len(ds) == 0:
            print(f"[{split}] EMPTY — skipped")
            continue

        res = extract_split(inner, loader, device)
        if res is None:
            print(f"[{split}] no batches produced — skipped")
            continue
        f, i, l, p = res
        print(f"[{split}] {len(i):,} studies   embedding dim {f.shape[1]}   "
              f"prevalence {l.mean():.4f}")
        all_f.append(f); all_i.append(i); all_l.append(l); all_p.append(p)
        all_s.append(np.array([split] * len(i), dtype=object))

    if not all_f:
        print("\nFAIL: no split produced any embeddings.")
        return 1

    emb = np.concatenate(all_f, 0)
    ids = np.concatenate(all_i, 0)
    lab = np.concatenate(all_l, 0)
    prob = np.concatenate(all_p, 0)
    spl = np.concatenate(all_s, 0)

    if len(set(ids)) != len(ids):
        print(f"WARNING: {len(ids) - len(set(ids)):,} duplicate ids across splits — "
              "check the split assignment")

    npz = out_dir / "embeddings.npz"
    np.savez_compressed(npz, embeddings=emb, ids=ids, labels=lab, probs=prob, split=spl)

    idx = out_dir / "index.csv"
    with open(idx, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "id", "split", "label", "prob"])
        for r, (i_, s_, l_, p_) in enumerate(zip(ids, spl, lab, prob)):
            w.writerow([r, i_, s_, f"{l_:.0f}", f"{p_:.6f}"])

    print("=" * 78)
    print(f"embeddings  : {emb.shape[0]:,} x {emb.shape[1]}   -> {npz}")
    print(f"index       : {idx}")
    print("\nLoad with:")
    print("    d = np.load(path, allow_pickle=True)")
    print("    X, ids, y, split = d['embeddings'], d['ids'], d['labels'], d['split']")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
