#!/usr/bin/env python
"""
Verify that the split filtering in RSNADataset2D actually does something.

RSNADataset2D (radfusion3/data/dataset_2d.py) filters like this:

    self.df = self.df[self.df.negative_exam_for_pe == 0]
    if self.split != "all":
        if "Split" in self.df.columns:
            self.df = self.df[self.df["Split"] == self.split]
        elif "split" in self.df.columns:
            self.df = self.df[self.df["split"] == self.split]
        # <-- no else: if neither column exists, NO filtering happens

DataModule requests split="train" / "valid" / "test". So:
  * no Split column      -> train == valid == test == full dataframe (total leakage)
  * column uses "val"    -> valid split is empty -> torch.cat([]) crashes at epoch end
  * study-level overlap  -> val AUROC is inflated

This script reproduces that logic exactly and reports what will happen.

Lives in <repo>/Custom/. Resolves the CSV in this order:
  1. --csv if given
  2. csv_path from image/radfusion3/configs/dataset/rsna.yaml  (what training reads)
  3. auto-discovery under ../../RSPECT relative to this file

Usage:
    python check_rsna_splits.py
    python check_rsna_splits.py --csv /path/to/train.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

STUDY_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
LABEL_COL = "pe_present_on_image"

# What DataModule actually asks for: train_dataloader -> "train",
# val_dataloader -> "valid", test_dataloader -> cfg.test_split (default "test")
REQUESTED_SPLITS = ["train", "valid", "test"]

# This file lives in <repo>/Custom/, so:
#   parents[0] = Custom
#   parents[1] = INSPECT_custom_data_preprocessing   (repo root)
#   parents[2] = the directory holding both the repo and RSPECT/
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
DATA_ROOT = HERE.parents[2] / "RSPECT"

RSNA_YAML = REPO_ROOT / "image" / "radfusion3" / "configs" / "dataset" / "rsna.yaml"

# Columns that identify a real RSPECT slice-level CSV
RSPECT_MARKERS = {STUDY_COL, SERIES_COL, "pe_present_on_image"}


def rule(char="=", n=78):
    print(char * n)


def csv_path_from_config():
    """Read csv_path out of rsna.yaml -- this is what training ACTUALLY loads."""
    if not RSNA_YAML.is_file():
        return None
    text = RSNA_YAML.read_text()
    m = re.search(r"^\s*csv_path\s*:\s*['\"]?([^'\"\n#]+)", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def discover_csv():
    """Find a slice-level RSPECT CSV under DATA_ROOT, preferring train.csv."""
    if not DATA_ROOT.is_dir():
        return None, []

    candidates = sorted(DATA_ROOT.rglob("*.csv"))
    if not candidates:
        return None, []

    # Prefer an exact train.csv, else the first file with the RSPECT columns
    named = [p for p in candidates if p.name.lower() == "train.csv"]
    for p in named + candidates:
        try:
            head = pd.read_csv(p, nrows=0)
        except Exception:
            continue
        if RSPECT_MARKERS.issubset(set(head.columns)):
            return p, candidates

    return None, candidates


def resolve_csv(explicit):
    """Decide which CSV to inspect and warn if it isn't the one training uses."""
    rule("-")
    print("STEP 0: locating the CSV")
    rule("-")
    print(f"script location : {HERE}")
    print(f"repo root       : {REPO_ROOT}")
    print(f"expected data   : {DATA_ROOT}  (exists={DATA_ROOT.is_dir()})")

    configured = csv_path_from_config()
    print(f"rsna.yaml csv_path : {configured or '<not found>'}")
    if configured:
        print(f"   -> exists on this machine: {Path(configured).is_file()}")

    if explicit:
        chosen = Path(explicit)
        print(f"\nusing --csv override: {chosen}")
    elif configured and Path(configured).is_file():
        chosen = Path(configured)
        print(f"\nusing the path from rsna.yaml (same file training reads): {chosen}")
    else:
        found, candidates = discover_csv()
        if found is None:
            print("\nCould not locate an RSPECT slice-level CSV.")
            if candidates:
                print("CSVs seen under the data root:")
                for p in candidates[:20]:
                    print(f"   {p}")
            print("\nPass one explicitly:  python check_rsna_splits.py --csv /path/to/train.csv")
            return None
        chosen = found
        print(f"\nrsna.yaml path is not readable here; discovered instead: {chosen}")

    # The important cross-check
    if configured and Path(configured).resolve() != chosen.resolve():
        print()
        print("WARNING: this is NOT the file the training job loads.")
        print(f"  training reads : {configured}")
        print(f"  inspecting     : {chosen}")
        print("  Fix rsna.yaml's csv_path, or re-run with --csv pointing at the real one,")
        print("  otherwise the results below say nothing about the running job.")

    print()
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="RSPECT slice-level CSV. Default: read csv_path from rsna.yaml, "
                         "else auto-discover under ../../RSPECT")
    args = ap.parse_args()

    csv_path = resolve_csv(args.csv)
    if csv_path is None:
        return 1

    rule()
    print(f"Reading {csv_path}")
    rule()

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"FAIL: file not found: {csv_path}")
        return 1

    print(f"rows={len(df):,}  cols={len(df.columns)}")
    print(f"columns: {list(df.columns)}\n")

    # ---------------------------------------------------------------- step 1
    rule("-")
    print("STEP 1: does a split column exist?")
    rule("-")

    split_col = None
    if "Split" in df.columns:
        split_col = "Split"
    elif "split" in df.columns:
        split_col = "split"

    if split_col is None:
        print("RESULT: NO 'Split' or 'split' column found.")
        print()
        print("  RSNADataset2D has no `else` branch, so the filter is skipped entirely.")
        print("  train, valid and test will all be the SAME full dataframe.")
        print("  Your val AUROC is being measured on the training data.")
        print()
        print("  Fix: add a study-level 'Split' column with values train/valid/test")
        print("  to this CSV (group by StudyInstanceUID so no study spans splits),")
        print("  or add an explicit `raise` in the else branch so this fails loudly.")
        return 1

    print(f"RESULT: found column '{split_col}'")
    print(f"value counts (before any filtering):\n{df[split_col].value_counts(dropna=False).to_string()}\n")

    # ---------------------------------------------------------------- step 2
    rule("-")
    print("STEP 2: do the values match what DataModule requests?")
    rule("-")

    present = set(df[split_col].dropna().unique())
    missing = [s for s in REQUESTED_SPLITS if s not in present]

    if missing:
        print(f"RESULT: MISMATCH. DataModule asks for {REQUESTED_SPLITS}")
        print(f"        but the column only contains {sorted(present)}")
        print(f"        missing: {missing}")
        print()
        if "valid" in missing:
            near = [v for v in present if str(v).lower().startswith("val")]
            if near:
                print(f"  Looks like you used {near} where the code expects 'valid'.")
            print("  An empty val split means validation_step never runs, so")
            print("  step_outputs['val']['y'] stays empty and shared_epoch_end hits")
            print("  torch.cat([]) -> RuntimeError: expected a non-empty list.")
            print("  With num_sanity_val_steps=0 you only find out at end of epoch 1.")
    else:
        print(f"RESULT: OK, all of {REQUESTED_SPLITS} are present.")
    print()

    # ---------------------------------------------------------------- step 3
    rule("-")
    print("STEP 3: replay RSNADataset2D filtering (incl. negative_exam_for_pe == 0)")
    rule("-")

    work = df
    if "negative_exam_for_pe" in work.columns:
        before = len(work)
        work = work[work.negative_exam_for_pe == 0]
        print(f"positive-exam filter: {before:,} -> {len(work):,} rows "
              f"({len(work)/max(before,1):.1%} kept)\n")
    else:
        print("WARNING: no 'negative_exam_for_pe' column; dataset __init__ would raise.\n")

    subsets = {}
    for split in REQUESTED_SPLITS:
        sub = work[work[split_col] == split]
        subsets[split] = sub

        print(f"[{split}] rows={len(sub):,}")
        if len(sub) == 0:
            print("      EMPTY -> this dataloader yields nothing; epoch end will crash.\n")
            continue

        if LABEL_COL in sub.columns:
            vc = sub[LABEL_COL].value_counts()
            pos = int(vc.get(1, 0))
            neg = int(vc.get(0, 0))
            print(f"      {LABEL_COL}: pos={pos:,} neg={neg:,} prevalence={pos/len(sub):.4f}")
            if pos == 0 or neg == 0:
                print("      SINGLE CLASS -> get_auroc returns a hardcoded 0.0, not a real score.")
        if STUDY_COL in sub.columns:
            print(f"      unique studies={sub[STUDY_COL].nunique():,}")
        if SERIES_COL in sub.columns:
            print(f"      unique series ={sub[SERIES_COL].nunique():,}")
        print()

    # ---------------------------------------------------------------- step 4
    rule("-")
    print("STEP 4: study-level leakage between splits")
    rule("-")

    if STUDY_COL not in work.columns:
        print(f"SKIP: no {STUDY_COL} column.")
    else:
        studies = {s: set(sub[STUDY_COL].unique()) for s, sub in subsets.items() if len(sub)}
        names = list(studies)
        clean = True
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                shared = studies[a] & studies[b]
                if shared:
                    clean = False
                    print(f"LEAK: {a} and {b} share {len(shared):,} studies "
                          f"(e.g. {list(shared)[:3]})")
        if clean and names:
            print("RESULT: OK, no StudyInstanceUID appears in more than one split.")
        elif not names:
            print("RESULT: nothing to compare, splits are empty.")
    print()

    # ---------------------------------------------------------------- step 5
    rule("-")
    print("STEP 5: validation cost estimate")
    rule("-")

    n_val = len(subsets.get("valid", []))
    if n_val:
        bs = 4  # rsna.yaml batch_size
        print(f"valid rows={n_val:,} at batch_size={bs} -> {n_val // bs:,} forward passes per epoch")
        print("Each is a full resnetv2_101x3 forward. If that number is in the hundreds of")
        print("thousands, validation dominates wall-clock; consider limit_val_batches < 1.0")
        print("or a per-study subsample for the val set.")
    else:
        print("valid split is empty, nothing to estimate.")
    print()

    rule()
    print("Done.")
    rule()
    return 0


if __name__ == "__main__":
    sys.exit(main())
