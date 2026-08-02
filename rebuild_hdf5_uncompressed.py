"""
Rebuild features.hdf5 without compression for faster random access during training.
Reads the existing compressed HDF5 and writes an uncompressed copy.
"""

import h5py
import tqdm
import os

SRC = "/data/processed/INSPECT/CNN_embeddings/features.hdf5"
DST = "/data/processed/INSPECT/CNN_embeddings/features_uncompressed.hdf5"

print(f"Source: {SRC}")
print(f"Destination: {DST}")

with h5py.File(SRC, "r") as src, h5py.File(DST, "w") as dst:
    keys = list(src.keys())
    print(f"Total scans: {len(keys)}")

    for key in tqdm.tqdm(keys, desc="Rebuilding"):
        data = src[key][:]
        dst.create_dataset(key, data=data, chunks=True)  # no compression

print(f"Done. Written to {DST}")
print(f"Size: {os.path.getsize(DST) / 1e9:.1f} GB")
