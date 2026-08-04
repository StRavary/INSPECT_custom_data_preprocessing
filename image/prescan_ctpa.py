"""
prescan_ctpa.py — Test every NIfTI in the CTPA directory and quarantine corrupt files.

Two scan modes:
  --fast (default): runs `gzip -t` on each file — tests CRC/integrity without
                    decompressing into RAM. ~2700 it/s. Catches truncation and
                    corrupt gzip streams. Won't catch valid gzip with garbled
                    NIfTI pixel data.
  --deep:           calls nibabel get_fdata() and validates shape + finite values.
                    Uses multiprocessing (--workers N). ~10–20 min for 23k files.
                    Matches exactly what run_featurize.py does at runtime.

Moves bad files to CTPA_corrupt/ sibling folder so run_featurize.py never sees them.
Re-download the quarantined files later and move them back to CTPA/ to re-featurize.

Usage:
    python prescan_ctpa.py [--dry_run]           # fast CRC scan, no moves
    python prescan_ctpa.py                       # fast CRC scan, quarantine bad
    python prescan_ctpa.py --deep --workers 16   # full data validation
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctpa_dir", default="/data/scratch/INSPECT/INSPECT_CTPA/full/CTPA")
    p.add_argument("--quarantine_dir", default="/data/scratch/INSPECT/INSPECT_CTPA/full/CTPA_corrupt")
    p.add_argument("--dry_run", action="store_true", help="Report only, don't move files.")
    p.add_argument("--deep", action="store_true", help="Load full pixel data (slow but thorough).")
    p.add_argument("--workers", type=int, default=8, help="Workers for --deep mode.")
    return p.parse_args()


# --- Fast mode: gzip CRC check ---

def check_fast(path: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["gzip", "-t", str(path)],
        capture_output=True,
    )
    if result.returncode != 0:
        return True, result.stderr.decode().strip() or "gzip CRC error"
    return False, ""


# --- Deep mode: full nibabel load (runs in worker process) ---

def check_deep(path_str: str) -> tuple[str, bool, str]:
    path = Path(path_str)
    try:
        img = nib.load(path_str)
        shape = img.shape
        if len(shape) < 3:
            return path.name, True, f"unexpected ndim={len(shape)}"
        if shape[0] < 2 or shape[1] < 2 or shape[2] < 1:
            return path.name, True, f"degenerate shape {shape}"

        data = img.get_fdata(dtype=np.float32)

        if not np.any(np.isfinite(data)):
            return path.name, True, "all non-finite values"
        if data.shape[2] != shape[2]:
            return path.name, True, f"shape mismatch: header {shape} vs data {data.shape}"

        return path.name, False, ""
    except Exception as e:
        return path.name, True, str(e)


def main():
    args = parse_args()
    ctpa_dir = Path(args.ctpa_dir)
    quarantine_dir = Path(args.quarantine_dir)

    if not ctpa_dir.exists():
        print(f"[ERROR] CTPA dir not found: {ctpa_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(ctpa_dir.glob("*.nii.gz"))
    print(f"Found {len(files):,} NIfTI files in {ctpa_dir}")
    mode = "DEEP (get_fdata)" if args.deep else "FAST (gzip -t CRC)"
    print(f"Mode: {mode}")

    if not args.dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    corrupt = []  # list of (Path, reason)

    if args.deep:
        path_strs = [str(f) for f in files]
        with Pool(args.workers) as pool:
            results = list(tqdm(
                pool.imap(check_deep, path_strs, chunksize=16),
                total=len(path_strs),
                desc=f"Scanning (deep, {args.workers} workers)",
            ))
        name_to_path = {f.name: f for f in files}
        for name, bad, reason in results:
            if bad:
                corrupt.append((name_to_path[name], reason))
    else:
        for path in tqdm(files, desc="Scanning (fast)"):
            bad, reason = check_fast(path)
            if bad:
                corrupt.append((path, reason))

    # Move corrupt files
    if not args.dry_run:
        for path, _ in corrupt:
            dest = quarantine_dir / path.name
            shutil.move(str(path), str(dest))

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Results:")
    print(f"  Total scanned  : {len(files):,}")
    print(f"  Healthy        : {len(files) - len(corrupt):,}")
    print(f"  Corrupt/moved  : {len(corrupt):,}")

    if corrupt:
        log_path = quarantine_dir.parent / "corrupt_ctpa_ids.txt"
        with open(log_path, "w") as f:
            for path, reason in corrupt:
                f.write(f"{path.name}\t{reason}\n")
        print(f"\nCorrupt IDs written to: {log_path}")
        print("\nFirst 10 corrupt files:")
        for path, reason in corrupt[:10]:
            print(f"  {path.name}: {reason}")


if __name__ == "__main__":
    main()
