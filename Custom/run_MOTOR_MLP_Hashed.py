#!/usr/bin/env python
"""
run_MOTOR_MLP_Hashed.py

Same MOTOR + 2-stage MLP pipeline as run_MOTOR_MLP.py, but evaluated on
CLMBR's ORIGINAL hash-based split instead of the cohort file's patient-level
split.

Why this exists
---------------
`clmbr_create_batches` partitions patients by hashing them into percentile
buckets (seed 97; --val_start 80 / --test_start 85 give an 80/5/15 split).
Stock `femr/models/linear_probe.py` evaluates on exactly that partition and
never opens a cohort file. Our pipeline overrides it with the cohort's
patient-level split so results are comparable to the GBM and imaging arms.

Running both lets us answer, on identical representations, how much the split
source actually changes the reported AUROC -- which is the open question about
INSPECT's published MOTOR numbers.

Caveats when interpreting the output
------------------------------------
* The hash partition is 80/5/15, so the validation set is TINY (~5%) and its
  AUROC is correspondingly noisy. Model selection on it is less reliable.
* It is patient-level (FEMR hashes patient IDs), so it is not leaky -- it is
  simply a *different* partition of the same patients.
* Patients absent from the cohort file are NOT skipped here, since the cohort
  file is not consulted. Expect a slightly larger N than the cohort run.

This is a thin wrapper: all logic lives in run_MOTOR_MLP.py so the two cannot
drift apart. It only flips the --split-source default to "hash".

Usage
-----
    python Custom/run_MOTOR_MLP_Hashed.py --task 12_month_PH
    python Custom/run_MOTOR_MLP_Hashed.py --task all

Every flag from run_MOTOR_MLP.py is accepted and forwarded unchanged
(--epochs, --dropout, --hidden-dim, --lr, --batch-size, --force-batches, ...).
"""

import sys

import run_MOTOR_MLP as base


def main() -> int:
    # Force the hash split unless the caller explicitly asked for something else.
    if not any(a == "--split-source" or a.startswith("--split-source=") for a in sys.argv[1:]):
        sys.argv.extend(["--split-source", "hash"])

    print("=" * 70)
    print("MOTOR + 2-stage MLP  --  CLMBR HASH SPLIT (80/5/15, seed 97)")
    print("This is NOT the cohort patient-level split. For the comparable-to-")
    print("baselines number, use run_MOTOR_MLP.py instead.")
    print("=" * 70)

    return base.main() or 0


if __name__ == "__main__":
    sys.exit(main())
