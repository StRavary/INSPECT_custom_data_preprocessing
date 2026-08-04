"""
prescan_ctpa.py — Test every NIfTI in the CTPA directory and quarantine corrupt files.

Corrupt = fails to load header OR has an unexpected shape (not 3D/4D).
Moves bad files to CTPA_corrupt/ sibling folder so run_featurize.py never touches them.
Re-download the quarantined files later and move them back to CTPA/ to re-featurize.

Usage:
    python prescan_ctpa.py [--ctpa_dir /data/scratch/INSPECT/INSPECT_CTPA/full/CTPA]
                           [--quarantine_dir /data/scratch/INSPECT/INSPECT_CTPA/full/CTPA_corrupt]
                           [--dry_run]
"""

import argparse
import shutil
import sys
from pathlib import Path

import nibabel as nib
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ctpa_dir",
        default="/data/scratch/INSPECT/INSPECT_CTPA/full/CTPA",
    )
    p.add_argument(
        "--quarantine_dir",
        default="/data/scratch/INSPECT/INSPECT_CTPA/full/CTPA_corrupt",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Report only — don't move any files.",
    )
    return p.parse_args()


def is_corrupt(path: Path) -> tuple[bool, str]:
    """Return (corrupt, reason). Only loads header, not full volume."""
    try:
        img = nib.load(str(path))
        shape = img.shape
        if len(shape) < 3:
            return True, f"unexpected ndim={len(shape)}"
        # Quick sanity: at least 2×2 spatial and at least 1 slice
        if shape[0] < 2 or shape[1] < 2 or shape[2] < 1:
            return True, f"degenerate shape {shape}"
        return False, ""
    except Exception as e:
        return True, str(e)


def main():
    args = parse_args()
    ctpa_dir = Path(args.ctpa_dir)
    quarantine_dir = Path(args.quarantine_dir)

    if not ctpa_dir.exists():
        print(f"[ERROR] CTPA dir not found: {ctpa_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(ctpa_dir.glob("*.nii.gz"))
    print(f"Found {len(files):,} NIfTI files in {ctpa_dir}")

    if not args.dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    corrupt = []
    for path in tqdm(files, desc="Scanning"):
        bad, reason = is_corrupt(path)
        if bad:
            corrupt.append((path, reason))
            if not args.dry_run:
                dest = quarantine_dir / path.name
                shutil.move(str(path), str(dest))

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Results:")
    print(f"  Total scanned : {len(files):,}")
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
