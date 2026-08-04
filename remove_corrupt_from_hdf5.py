"""
remove_corrupt_from_hdf5.py — Remove zero-feature (corrupt/missing) scans from both HDF5 files.

Reads corrupt_ctpa_ids.txt (produced by prescan_ctpa.py) and any scan whose
features are all-zero (missing files that featurize silently zeroed out), then
deletes those keys from both features.hdf5 and features_resnext_uncompressed.hdf5.
"""

from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

CORRUPT_LOG  = "/data/scratch/INSPECT/INSPECT_CTPA/full/corrupt_ctpa_ids.txt"
HDF5_COMPRESSED   = "/data/processed/INSPECT/CNN_embeddings/features.hdf5"
HDF5_UNCOMPRESSED = "/data/processed/INSPECT/CNN_embeddings/features_resnext_uncompressed.hdf5"

# --- 1. Build set of known-corrupt IDs from prescan log ---
corrupt_ids = set()
log = Path(CORRUPT_LOG)
if log.exists():
    for line in log.read_text().splitlines():
        fname = line.split('\t')[0].strip()
        corrupt_ids.add(fname.replace('.nii.gz', ''))
    print(f"Corrupt IDs from prescan log : {len(corrupt_ids)}")
else:
    print(f"[WARN] Corrupt log not found at {CORRUPT_LOG}")

# --- 2. Scan HDF5 for all-zero feature scans (missing files) ---
zero_ids = set()
print("Scanning for all-zero features (missing files)...")
with h5py.File(HDF5_COMPRESSED, 'r') as f:
    for key in tqdm(f.keys()):
        arr = f[key][:]
        # Exclude the positional encoding column (last column) before checking
        features_only = arr[:, :-1]
        if not np.any(features_only != 0):
            zero_ids.add(key)

print(f"All-zero feature scans found  : {len(zero_ids)}")

to_remove = corrupt_ids | zero_ids
print(f"Total keys to remove          : {len(to_remove)}")

if not to_remove:
    print("Nothing to remove. Exiting.")
    exit(0)

# --- 3. Remove from both HDF5 files ---
for hdf5_path in [HDF5_COMPRESSED, HDF5_UNCOMPRESSED]:
    p = Path(hdf5_path)
    if not p.exists():
        print(f"[SKIP] {hdf5_path} not found")
        continue
    removed = 0
    with h5py.File(hdf5_path, 'a') as f:
        for key in to_remove:
            if key in f:
                del f[key]
                removed += 1
    print(f"Removed {removed} keys from {p.name}")

print("Done.")
