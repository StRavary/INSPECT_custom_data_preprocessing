import os
import h5py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR      = "/data/processed/INSPECT/CNN_embeddings"
METADATA_PATH = "/home/users/steven/INSPECT/DATA_RAW/LABELS/series_metadata_20250611.tsv"


def convert_npy_to_hdf5(input_dir, output_path, metadata_path):
    sep = '\t' if str(metadata_path).endswith('.tsv') else ','
    df_metadata = pd.read_csv(metadata_path, sep=sep)

    # Build set of valid image_ids from metadata (strip .nii.gz)
    valid_image_ids = set(df_metadata['image_id'].str.replace('.nii.gz', '', regex=False))

    # Each scan is a subdirectory named by image_id; slices are slice_NNNN.npy inside
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name", default="resnext_window",
        help="Tag for this featurization run. "
             "Input .npy dir: {BASE_DIR}/{run_name}_*/ (most recent timestamped dir). "
             "Output HDF5: {BASE_DIR}/{run_name}_uncompressed.hdf5"
    )
    parser.add_argument("--input_dir",  default=None, help="Override input directory")
    parser.add_argument("--output_path", default=None, help="Override output HDF5 path")
    parser.add_argument("--metadata_path", default=METADATA_PATH)
    args = parser.parse_args()

    # Resolve input dir: find the most recent timestamped subdir matching run_name
    if args.input_dir:
        input_dir = args.input_dir
    else:
        candidates = sorted(Path(BASE_DIR).glob(f"{args.run_name}_*"))
        candidates = [d for d in candidates if d.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"No directory matching '{args.run_name}_*' found in {BASE_DIR}. "
                "Run run_featurize.py first, or pass --input_dir."
            )
        input_dir = str(candidates[-1])
        print(f"Input dir: {input_dir}")

    # Resolve output path
    output_path = args.output_path or os.path.join(
        BASE_DIR, f"{args.run_name}_uncompressed.hdf5"
    )
    print(f"Output HDF5: {output_path}")

    convert_npy_to_hdf5(input_dir, output_path, args.metadata_path)
    print(f"HDF5 file created at: {output_path}")
