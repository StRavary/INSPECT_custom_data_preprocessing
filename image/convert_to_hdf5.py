import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_npy_to_hdf5(input_dir, output_path, metadata_path):
    sep = '\t' if str(metadata_path).endswith('.tsv') else ','
    df_metadata = pd.read_csv(metadata_path, sep=sep)

    # Build set of valid image_ids from metadata (strip .nii.gz)
    valid_image_ids = set(df_metadata['image_id'].str.replace('.nii.gz', '', regex=False))

    # Each scan is a subdirectory; slices are slice_NNNN.npy files inside
    scan_dirs = sorted([d for d in Path(input_dir).iterdir() if d.is_dir()])

    written = 0
    skipped = 0

    with h5py.File(output_path, 'w') as hdf5_file:
        for scan_dir in tqdm(scan_dirs, desc="Building HDF5"):
            image_id = scan_dir.name

            # Skip dirs not in metadata (e.g. wandb, classify_* output dirs)
            if image_id not in valid_image_ids:
                skipped += 1
                continue

            slice_files = sorted(scan_dir.glob('slice_*.npy'))
            if not slice_files:
                print(f"[WARN] No slice files for {image_id}, skipping")
                skipped += 1
                continue

            try:
                slices = [np.load(f) for f in slice_files]
                features = np.stack(slices, axis=0).astype(np.float32)  # [num_slices, feature_dim]
                hdf5_file.create_dataset(image_id, data=features, dtype='float32')
                written += 1
            except Exception as e:
                print(f"[ERROR] {image_id}: {e}")
                skipped += 1
                continue

    print(f"HDF5 complete: {written} scans written, {skipped} skipped.")


if __name__ == "__main__":
    input_dir = "/data/processed/INSPECT/CNN_embeddings"
    output_path = "/data/processed/INSPECT/CNN_embeddings/features.hdf5"
    metadata_path = "/home/users/steven/INSPECT/DATA_RAW/LABELS/series_metadata_20250611.tsv"

    convert_npy_to_hdf5(input_dir, output_path, metadata_path)
    print(f"HDF5 file created at: {output_path}")
