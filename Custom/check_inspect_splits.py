#!/usr/bin/env python
"""
Verify the INSPECT / Stanford split plumbing in Dataset1D.

This is the path that produces the numbers we actually report, so it matters
more than the RSPECT one. Dataset1D (radfusion3/data/dataset_1d.py) does:

    self.df["patient_datetime"] = self.df["image_id"].str.replace(".nii.gz", "")
    if 'impression_id' not in self.df.columns:
        self.df['impression_id'] = self.df["patient_datetime"]
    self.df = self.df.drop_duplicates(subset=["patient_datetime"])

    if hasattr(cfg.dataset, 'label_csv') and cfg.dataset.label_csv:
        self.df = self.df.merge(pd.read_csv(cfg.dataset.label_csv),
                                on='impression_id', how='left')
    if hasattr(cfg.dataset, 'split_csv') and cfg.dataset.split_csv:
        self.df = self.df.merge(pd.read_csv(cfg.dataset.split_csv),
                                on='impression_id', how='left')

    if split != "all" and 'split' in self.df.columns:
        self.df = self.df[self.df["split"] == split]      # <-- lowercase, no else

Three ways this goes wrong silently:

 1. The split file names its column `Split` (capital S -- which is what this
    repo's own constants.py declares as SPLIT_COL). Then `'split' in columns`
    is False, the filter is skipped, and train == valid == test.

 2. The merge on impression_id fails (dtype mismatch, missing keys), leaving
    NaN splits. Rows silently vanish from every split.

 3. Splits are assigned per EXAM rather than per PATIENT. Since some INSPECT
    patients have more than one CTPA, the same patient can appear in train and
    test -- same anatomy, same comorbidities, correlated outcome.

Usage:
    python check_inspect_splits.py
    python check_inspect_splits.py --metadata M.csv --labels L.csv --splits S.csv
    python check_inspect_splits.py --target 12_month_mortality
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
STANFORD_YAML = (
    REPO_ROOT / "image" / "radfusion3" / "configs" / "dataset" / "stanford_featurized.yaml"
)

MERGE_KEY = "impression_id"


def rule(char="=", n=78):
    print(char * n)


def yaml_value(text, key):
    m = re.search(rf"^\s*{key}\s*:\s*['\"]?([^'\"\n#]+)", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def load_config():
    if not STANFORD_YAML.is_file():
        return {}
    text = STANFORD_YAML.read_text()
    return {
        k: yaml_value(text, k)
        for k in ("csv_path", "label_csv", "split_csv", "target")
    }


def read(path, what):
    if path is None:
        print(f"  {what:<10}: <not configured>")
        return None
    p = Path(path)
    if not p.is_file():
        print(f"  {what:<10}: MISSING -> {p}")
        return None
    df = pd.read_csv(p)
    print(f"  {what:<10}: {len(df):>8,} rows, {len(df.columns):>3} cols  {p}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=None, help="csv_path from stanford_featurized.yaml")
    ap.add_argument("--labels", default=None, help="label_csv")
    ap.add_argument("--splits", default=None, help="split_csv")
    ap.add_argument("--target", default=None, help="target column, e.g. 12_month_mortality")
    args = ap.parse_args()

    cfg = load_config()
    meta_path = args.metadata or cfg.get("csv_path")
    label_path = args.labels or cfg.get("label_csv")
    split_path = args.splits or cfg.get("split_csv")
    target = args.target or cfg.get("target")

    rule()
    print("STEP 0: inputs")
    rule()
    print(f"config: {STANFORD_YAML if STANFORD_YAML.is_file() else '<not found>'}")
    print(f"target: {target}")
    meta = read(meta_path, "metadata")
    labels = read(label_path, "labels")
    splits = read(split_path, "splits")

    if meta is None:
        print("\nFAIL: cannot proceed without the metadata CSV. Pass --metadata.")
        return 1

    # --------------------------------------------------------- replay Dataset1D
    print()
    rule("-")
    print("STEP 1: replay Dataset1D preprocessing")
    rule("-")

    df = meta.copy()
    if "image_id" not in df.columns:
        print("FAIL: metadata has no 'image_id' column.")
        return 1

    # NOTE: Dataset1D calls .str.replace(".nii.gz", "") without regex=False.
    # On pandas < 2.0 that is a REGEX replace, where '.' matches any character.
    df["patient_datetime"] = df["image_id"].str.replace(".nii.gz", "", regex=False)
    n_regex = (
        df["image_id"].str.replace(".nii.gz", "", regex=True) != df["patient_datetime"]
    ).sum()
    print(f"pandas {pd.__version__}; literal vs regex .replace differ on {n_regex:,} rows")
    if n_regex:
        print("  WARNING: on pandas<2.0 Dataset1D would mangle these IDs.")

    if MERGE_KEY not in df.columns:
        df[MERGE_KEY] = df["patient_datetime"]
        print(f"'{MERGE_KEY}' absent -> derived from patient_datetime")

    before = len(df)
    df = df.drop_duplicates(subset=["patient_datetime"])
    print(f"drop_duplicates(patient_datetime): {before:,} -> {len(df):,}")

    for name, extra in (("labels", labels), ("splits", splits)):
        if extra is None:
            continue
        if MERGE_KEY not in extra.columns:
            print(f"FAIL: {name} CSV has no '{MERGE_KEY}' column; "
                  f"columns are {list(extra.columns)[:8]}...")
            return 1
        lt, rt = df[MERGE_KEY].dtype, extra[MERGE_KEY].dtype
        if lt != rt:
            print(f"  WARNING: {name} merge dtype mismatch: {lt} vs {rt} "
                  "-> merge will match nothing")
        matched = df[MERGE_KEY].isin(set(extra[MERGE_KEY])).sum()
        print(f"  {name}: {matched:,}/{len(df):,} keys match ({matched/max(len(df),1):.1%})")
        df = df.merge(extra, on=MERGE_KEY, how="left")

    # ------------------------------------------------ the capitalization check
    print()
    rule("-")
    print("STEP 2: does Dataset1D's filter actually fire?")
    rule("-")

    has_lower = "split" in df.columns
    has_upper = "Split" in df.columns
    print(f"lowercase 'split' column present : {has_lower}")
    print(f"capital   'Split' column present : {has_upper}")

    if not has_lower:
        print()
        print("RESULT: FAIL -- Dataset1D checks for lowercase 'split' only.")
        if has_upper:
            print("  Your split file uses 'Split'. The filter is SKIPPED entirely,")
            print("  so train == valid == test on the INSPECT cohort, and every")
            print("  reported number is measured on training data.")
            print("  Fix: rename the column to 'split', or make Dataset1D accept both.")
        else:
            print("  No split column at all after the merge. Same consequence.")
        return 1

    print("RESULT: OK -- the filter will fire.")

    n_null = df["split"].isna().sum()
    if n_null:
        print(f"  WARNING: {n_null:,} rows have NaN split (failed merge); "
              "they belong to no split and are silently dropped.")

    print(f"  values: {df['split'].value_counts(dropna=False).to_dict()}")

    requested = ["train", "valid", "test"]
    missing = [s for s in requested if s not in set(df["split"].dropna())]
    if missing:
        print(f"  WARNING: DataModule requests {requested}; missing {missing}.")
        print("           An empty split crashes at epoch end (torch.cat of []).")

    # ------------------------------------------------------- patient-level leak
    print()
    rule("-")
    print("STEP 3: patient-level leakage (the multi-CTPA problem)")
    rule("-")

    pid = None
    for cand in ("patient_id", "person_id", "PatientID"):
        if cand in df.columns:
            pid = cand
            break
    if pid is None:
        # patient_datetime is "<patient_id>_<procedure_time>"
        if df["patient_datetime"].str.contains("_").all():
            df["_pid"] = df["patient_datetime"].str.split("_").str[0]
            pid = "_pid"
            print("derived patient id from patient_datetime prefix")
        else:
            print("SKIP: no patient id column and cannot derive one.")
            pid = None

    if pid:
        exams = df.groupby(pid)[MERGE_KEY].nunique()
        multi = int((exams > 1).sum())
        print(f"patients: {df[pid].nunique():,}   exams: {len(df):,}")
        print(f"patients with >1 CTPA: {multi:,} (max {int(exams.max())})")

        if multi == 0:
            print("RESULT: OK -- one exam per patient, exam-level split is patient-level.")
        else:
            spanning = df.groupby(pid)["split"].nunique()
            n_span = int((spanning > 1).sum())
            if n_span == 0:
                print("RESULT: OK -- no patient's exams are split across sets.")
            else:
                print(f"RESULT: LEAK -- {n_span:,} patients have exams in >1 split.")
                bad = spanning[spanning > 1].index[:5].tolist()
                for b in bad:
                    sub = df[df[pid] == b]
                    print(f"    {b}: {sub['split'].value_counts().to_dict()}")
                print("  Same patient in train and test: same anatomy, same")
                print("  comorbidities, correlated outcome. Split by patient instead.")

    # ------------------------------------------------------------ per-split view
    if target and target in df.columns:
        print()
        rule("-")
        print(f"STEP 4: per-split breakdown for target '{target}'")
        rule("-")
        hdr = f"{'split':<8}{'exams':>10}{'patients':>11}{'pos':>10}{'prevalence':>13}"
        print(hdr)
        print("-" * len(hdr))
        for name in requested:
            sub = df[df["split"] == name]
            if not len(sub):
                print(f"{name:<8}{'EMPTY':>10}")
                continue
            y = pd.to_numeric(sub[target], errors="coerce")
            pos = int((y == 1).sum())
            npat = sub[pid].nunique() if pid else -1
            print(f"{name:<8}{len(sub):>10,}{npat:>11,}{pos:>10,}"
                  f"{pos/max(len(sub),1):>13.4f}")
    elif target:
        print(f"\nNOTE: target '{target}' not found after merge; "
              "check the label CSV joined correctly.")

    print()
    rule()
    print("Done.")
    rule()
    return 0


if __name__ == "__main__":
    sys.exit(main())
