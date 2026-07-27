#!/usr/bin/env python
"""
Add a study-level `Split` column to the RSPECT (RSNA-STR PE Detection) train.csv.

Why this is needed
------------------
radfusion3's RSNADataset2D / RSNADataset1D both assume the CSV already carries a
`Split` column -- upstream maintained a split-annotated copy on their cluster.
The stock Kaggle download has no such column (17 columns, none of them Split),
so the filter silently degrades to "use everything" and train == valid == test.

What this writes
----------------
A NEW csv (default: <input>_with_splits.csv) with one extra column, `Split`,
whose values are exactly "train" / "valid" / "test" -- the strings DataModule
requests. The input file is never modified, so this is safe to run while a
training job is reading the original.

Guarantees
----------
* Grouping is by StudyInstanceUID, so no study (and therefore no series, and no
  slice) ever spans two splits. This is the property that matters: slices within
  a study are near-duplicates, so a slice-level split would leak badly.
* Stratified on the study-level `negative_exam_for_pe` label, so PE prevalence is
  preserved across splits.
* Deterministic given --seed.

Usage
-----
    python make_rspect_splits.py                     # 70/15/15, seed 42
    python make_rspect_splits.py --train 0.8 --valid 0.1 --test 0.1
    python make_rspect_splits.py --csv /path/train.csv --out /path/out.csv

Then point image/radfusion3/configs/dataset/rsna.yaml at the new file and
re-run Custom/check_rsna_splits.py to confirm.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
LABEL_COL = "pe_present_on_image"
EXAM_NEG_COL = "negative_exam_for_pe"
SPLIT_COL = "Split"

SPLIT_NAMES = ["train", "valid", "test"]  # exact strings DataModule asks for

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
RSNA_YAML = REPO_ROOT / "image" / "radfusion3" / "configs" / "dataset" / "rsna.yaml"


def rule(char="=", n=78):
    print(char * n)


def csv_path_from_config():
    if not RSNA_YAML.is_file():
        return None
    m = re.search(
        r"^\s*csv_path\s*:\s*['\"]?([^'\"\n#]+)", RSNA_YAML.read_text(), re.MULTILINE
    )
    return m.group(1).strip() if m else None


def assign_splits(study_df, fracs, seed):
    """Stratified study-level assignment.

    study_df: one row per StudyInstanceUID, with a `stratum` column.
    Returns a Series of split names indexed like study_df.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series(index=study_df.index, dtype=object)

    for stratum, group in study_df.groupby("stratum", sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)

        # Largest-remainder allocation so the counts sum exactly to n and no
        # split is silently emptied by float rounding on small strata.
        raw = np.array([n * f for f in fracs])
        counts = np.floor(raw).astype(int)
        remainder = n - counts.sum()
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            counts[order[:remainder]] += 1

        start = 0
        for name, c in zip(SPLIT_NAMES, counts):
            out.loc[idx[start:start + c]] = name
            start += c

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="input RSPECT train.csv")
    ap.add_argument("--out", default=None, help="output path (default <input>_with_splits.csv)")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--valid", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting --out")
    args = ap.parse_args()

    fracs = [args.train, args.valid, args.test]
    if abs(sum(fracs) - 1.0) > 1e-6:
        print(f"FAIL: fractions must sum to 1.0, got {sum(fracs)}")
        return 1
    if any(f <= 0 for f in fracs):
        print("FAIL: every fraction must be > 0")
        return 1

    # ------------------------------------------------------------------ input
    csv_path = args.csv or csv_path_from_config()
    if not csv_path:
        print("FAIL: no --csv given and could not parse csv_path from rsna.yaml")
        return 1
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f"FAIL: not found: {csv_path}")
        return 1

    out_path = Path(args.out) if args.out else csv_path.with_name(
        csv_path.stem + "_with_splits" + csv_path.suffix
    )
    if out_path.exists() and not args.overwrite:
        print(f"FAIL: {out_path} exists. Pass --overwrite to replace it.")
        return 1
    if out_path.resolve() == csv_path.resolve():
        print("FAIL: refusing to overwrite the input CSV in place.")
        return 1

    rule()
    print(f"in  : {csv_path}")
    print(f"out : {out_path}")
    print(f"split: train={args.train} valid={args.valid} test={args.test}  seed={args.seed}")
    rule()

    df = pd.read_csv(csv_path)
    print(f"rows={len(df):,}  cols={len(df.columns)}")

    for col in (STUDY_COL, EXAM_NEG_COL):
        if col not in df.columns:
            print(f"FAIL: required column '{col}' missing.")
            return 1

    if SPLIT_COL in df.columns:
        print(f"NOTE: input already has a '{SPLIT_COL}' column; it will be replaced.")
        df = df.drop(columns=[SPLIT_COL])

    # ------------------------------------------------- build study-level table
    print()
    rule("-")
    print("Building study-level table")
    rule("-")

    per_study = df.groupby(STUDY_COL)[EXAM_NEG_COL].nunique()
    inconsistent = per_study[per_study > 1]
    if len(inconsistent):
        print(f"WARNING: {len(inconsistent)} studies have inconsistent {EXAM_NEG_COL};")
        print("         using the per-study max as the stratum.")

    studies = (
        df.groupby(STUDY_COL, sort=True)[EXAM_NEG_COL]
        .max()
        .rename("stratum")
        .reset_index()
    )
    print(f"unique studies: {len(studies):,}")
    print(f"  PE-positive exams (negative_exam_for_pe==0): "
          f"{(studies['stratum'] == 0).sum():,}")
    print(f"  PE-negative exams (negative_exam_for_pe==1): "
          f"{(studies['stratum'] == 1).sum():,}")

    # ------------------------------------------------------------- assignment
    studies[SPLIT_COL] = assign_splits(studies, fracs, args.seed)

    if studies[SPLIT_COL].isna().any():
        print("FAIL: some studies were left unassigned.")
        return 1

    out = df.merge(studies[[STUDY_COL, SPLIT_COL]], on=STUDY_COL, how="left")
    if out[SPLIT_COL].isna().any():
        print("FAIL: merge left unassigned slices.")
        return 1
    if len(out) != len(df):
        print(f"FAIL: merge changed row count {len(df):,} -> {len(out):,}")
        return 1

    # ----------------------------------------------------------- verification
    print()
    rule("-")
    print("Verification")
    rule("-")

    # 1. no study spans splits
    spanning = out.groupby(STUDY_COL)[SPLIT_COL].nunique()
    n_span = int((spanning > 1).sum())
    print(f"studies spanning >1 split : {n_span}  {'OK' if n_span == 0 else 'FAIL'}")
    if n_span:
        return 1

    # 2. series never span splits either (implied, but check explicitly)
    if SERIES_COL in out.columns:
        span_series = out.groupby(SERIES_COL)[SPLIT_COL].nunique()
        n_ss = int((span_series > 1).sum())
        print(f"series spanning >1 split  : {n_ss}  {'OK' if n_ss == 0 else 'FAIL'}")
        if n_ss:
            return 1

    # 3. all three names present
    present = set(out[SPLIT_COL].unique())
    missing = [s for s in SPLIT_NAMES if s not in present]
    print(f"split names present       : {sorted(present)}"
          f"{'  OK' if not missing else '  FAIL missing ' + str(missing)}")
    if missing:
        return 1

    # 4. per-split breakdown, both raw and post-filter (the dataset drops
    #    negative exams before training, so that is the count that matters)
    print()
    header = f"{'split':<8}{'studies':>10}{'slices':>14}{'pos-exam slices':>18}{'slice PE rate':>16}"
    print(header)
    print("-" * len(header))
    for name in SPLIT_NAMES:
        sub = out[out[SPLIT_COL] == name]
        pos_exam = sub[sub[EXAM_NEG_COL] == 0]
        rate = pos_exam[LABEL_COL].mean() if len(pos_exam) and LABEL_COL in sub.columns else float("nan")
        print(f"{name:<8}{sub[STUDY_COL].nunique():>10,}{len(sub):>14,}"
              f"{len(pos_exam):>18,}{rate:>16.4f}")

    total_pos = (out[EXAM_NEG_COL] == 0).sum()
    print("-" * len(header))
    print(f"{'TOTAL':<8}{out[STUDY_COL].nunique():>10,}{len(out):>14,}{total_pos:>18,}")

    # ---------------------------------------------------------------- write
    out.to_csv(out_path, index=False)
    print()
    rule()
    print(f"Wrote {out_path}  ({len(out):,} rows, {len(out.columns)} cols)")
    print()
    print("Next steps:")
    print(f"  1. set csv_path in {RSNA_YAML.relative_to(REPO_ROOT)} to:")
    print(f"       '{out_path}'")
    print("  2. python Custom/check_rsna_splits.py     # should now pass every step")
    rule()
    return 0


if __name__ == "__main__":
    sys.exit(main())
