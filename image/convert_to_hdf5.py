import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def convert_npy_to_hdf5(input_dir, output_path, metadata_path):
    # Read metadata to get mapping between impression_id and image_id
    sep = '\t' if str(metadata_path).endswith('.tsv') else ','
    df_metadata = pd.read_csv(metadata_path, sep=sep)

    # Create HDF5 file
    with h5py.File(output_path, 'w') as hdf5_file:
        # Get all NPY files
        npy_files = list(Path(input_dir).glob('*.npy'))

        for npy_file in npy_files:
            try:
                # Get image_id from filename
                image_id = npy_file.stem

                # Get impression_id from metadata
                impression_id = df_metadata[df_metadata['image_id'] == f"{image_id}.nii.gz"]['impression_id'].iloc[0]

                # Load features
                features = np.load(npy_file)

                # Ensure features are 2D (num_slices x feature_dim)
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)

                # Save to HDF5 using the image_id as key (without .nii.gz)
                hdf5_file.create_dataset(image_id, data=features, dtype='float32')
                print(f"Added {image_id} to HDF5 file with shape {features.shape}")
            except Exception as e:
                print(f"Error processing {npy_file}: {str(e)}")
                continue

if __name__ == "__main__":
    input_dir = "/data/processed/INSPECT/CNN_embeddings"
    output_path = "/data/processed/INSPECT/CNN_embeddings/features.hdf5"
    metadata_path = "/home/users/steven/INSPECT/DATA_RAW/LABELS/series_metadata_20250611.tsv"

    convert_npy_to_hdf5(input_dir, output_path, metadata_path)
    print(f"HDF5 file created at: {output_path}")
