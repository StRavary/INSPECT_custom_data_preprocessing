"""
x_build_sparse_matrix.py
-------------------------
Load X.npy, X_mask.npy, and y.npy from a Route B feature export directory
(app_feature_extraction.py's Export tab) and fuse them into a single
scipy.sparse matrix:

    [ X (NaN -> 0) | X_mask (1 = observed, 0 = missing) | y ]

Why X_mask is in there
-----------------------
X.npy uses NaN for "not measured" (labs) and a plain 0 for "not observed"
(diag:/drug:/proc:/obs:/visit: counts). scipy.sparse compresses structural
zeros, not NaN — a NaN-heavy array (typical for a labs-only extract) won't
sparsify at all until the NaNs become 0. But naively filling NaN -> 0
collapses two different things into the same stored value: "this lab came
back 0" and "this lab was never drawn." Appending X_mask as extra columns
keeps that distinction available to anything reading the fused matrix,
instead of silently discarding it the way a bare `np.nan_to_num(X)` would.

Usage
-----
    python Custom/x_build_sparse_matrix.py \
        /home/sravar/Documents/INSPECT/DATA_PROCESSED/exports/12_month_PH/visit_1eb8f0b1

    python Custom/x_build_sparse_matrix.py <export_dir> -o /path/to/output.npz

Reload the result with:
    import scipy.sparse
    fused = scipy.sparse.load_npz("fused_sparse.npz")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def build_sparse_matrix(export_dir: Path) -> tuple[sp.csr_matrix, dict]:
    """Load X.npy / X_mask.npy / y.npy from `export_dir` and fuse them.

    Returns
    -------
    fused : scipy.sparse.csr_matrix, shape (n_samples, 2*n_features + 1)
        Column layout: [0 : n_features)              = X, NaN filled with 0
                        [n_features : 2*n_features)   = X_mask (1=observed, 0=missing)
                        [2*n_features]                = y
    meta  : dict describing the column layout and basic stats, so a caller
        doesn't have to re-derive n_features from the fused shape by hand.
    """
    X      = np.load(export_dir / "X.npy")        # float32, NaN = not measured
    X_mask = np.load(export_dir / "X_mask.npy")    # uint8,   1 = observed, 0 = missing
    y      = np.load(export_dir / "y.npy")

    if X.shape != X_mask.shape:
        raise ValueError(
            f"X.npy {X.shape} and X_mask.npy {X_mask.shape} shapes don't match "
            "— are they from the same export?")
    if len(y) != X.shape[0]:
        raise ValueError(f"y.npy has {len(y):,} rows but X.npy has {X.shape[0]:,}")

    n_samples, n_features = X.shape
    n_nan = int(np.isnan(X).sum())
    total = X.size
    print(f"X.npy:      {X.shape}  ({n_nan:,}/{total:,} = {n_nan / total:.1%} NaN)")
    print(f"X_mask.npy: {X_mask.shape}  "
          f"({int(X_mask.sum()):,}/{total:,} = {X_mask.mean():.1%} observed)")
    print(f"y.npy:      {y.shape}  (prevalence {float(np.nanmean(y)):.4f})")

    # NaN -> 0 so the matrix actually sparsifies. See module docstring for
    # why X_mask is appended rather than just discarding the distinction.
    X_filled = np.nan_to_num(X, nan=0.0).astype(np.float32)

    X_sparse    = sp.csr_matrix(X_filled)
    mask_sparse = sp.csr_matrix(X_mask.astype(np.float32))
    y_sparse    = sp.csr_matrix(y.reshape(-1, 1).astype(np.float32))

    fused = sp.hstack([X_sparse, mask_sparse, y_sparse], format="csr")

    meta = {
        "n_samples":    n_samples,
        "n_features":   n_features,
        "value_cols":   [0, n_features],              # half-open [start, end)
        "mask_cols":    [n_features, 2 * n_features],  # half-open [start, end)
        "label_col":    2 * n_features,
        "density":      fused.nnz / (fused.shape[0] * fused.shape[1]),
    }
    return fused, meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fuse X.npy + X_mask.npy + y.npy into one sparse matrix.")
    ap.add_argument("export_dir", type=Path,
                     help="Directory containing X.npy, X_mask.npy, y.npy "
                          "(a Route B export directory).")
    ap.add_argument("-o", "--output", type=Path, default=None,
                     help="Output .npz path (default: <export_dir>/fused_sparse.npz)")
    args = ap.parse_args()

    fused, meta = build_sparse_matrix(args.export_dir)

    out_path = args.output or (args.export_dir / "fused_sparse.npz")
    sp.save_npz(out_path, fused)

    # Small sidecar so the column layout survives without re-deriving it
    # from n_features by hand every time this gets reloaded.
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nFused matrix: {fused.shape}  nnz={fused.nnz:,}  density={meta['density']:.2%}")
    print(f"  columns {meta['value_cols']}  = X values (NaN filled with 0)")
    print(f"  columns {meta['mask_cols']}  = X_mask (1=observed, 0=missing)")
    print(f"  column  {meta['label_col']}  = y")
    print(f"\nSaved -> {out_path}")
    print(f"Column layout -> {meta_path}")
    print(f"\nReload with:\n"
          f"  import scipy.sparse, json\n"
          f"  fused = scipy.sparse.load_npz('{out_path}')\n"
          f"  meta  = json.load(open('{meta_path}'))")


if __name__ == "__main__":
    main()
