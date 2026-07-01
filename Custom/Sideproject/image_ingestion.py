import os
import pandas as pd
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd

class CTPAImageLoader:
    """
    High-Throughput 3D Image Ingestion Pipeline using MONAI.
    Lazy-loads CTPA volumetric arrays directly from disk into PyTorch Tensors.
    """
    def __init__(self, raw_image_dir: str, crosswalk_path: str, file_extension: str = ".nii.gz"):
        self.raw_image_dir = raw_image_dir
        self.crosswalk_path = crosswalk_path
        self.file_extension = file_extension
        
        # Phase 1 minimal transforms: Just load the file efficiently.
        # (Phase 2 will add Resample, Normalize, etc. here)
        self.transforms = Compose([
            LoadImaged(keys=["image"], reader="ITKReader"),
            EnsureChannelFirstd(keys=["image"]),
        ])

    def build_dataset(self):
        print(f"Loading Crosswalk mapping from {self.crosswalk_path}...")
        df_crosswalk = pd.read_csv(self.crosswalk_path)
        
        data_dicts = []
        missing_count = 0
        
        print("Mapping patient IDs to physical 3D image paths...")
        for _, row in df_crosswalk.iterrows():
            person_id = row['person_id']
            image_id = row['image_id']
            
            # Construct the expected file path.
            # MONAI LoadImaged can handle both a single .nii.gz file OR a directory of .dcm files!
            # We assume NIfTI format (.nii.gz) as default for modern public datasets.
            expected_path = os.path.join(self.raw_image_dir, f"{image_id}{self.file_extension}")
            
            if os.path.exists(expected_path):
                data_dicts.append({
                    "image": expected_path,
                    "person_id": person_id
                })
            else:
                missing_count += 1
                
        print(f"Successfully mapped {len(data_dicts)} images.")
        if missing_count > 0:
            print(f"Warning: {missing_count} images mapped in crosswalk were not found on disk.")
            
        # Wrap into MONAI's lazy-loading Dataset
        return Dataset(data=data_dicts, transform=self.transforms)

    def get_dataloader(self, batch_size=4, num_workers=4):
        dataset = self.build_dataset()
        print(f"Spawning DataLoader with {num_workers} background workers...")
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=torch.cuda.is_available()
        )

if __name__ == "__main__":
    # WARNING: You must download the images from the Stanford Azure portal first!
    IMAGE_DIR = os.path.expanduser("../DATA_RAW/IMAGES")
    CROSSWALK_PATH = os.path.expanduser("../DATA_PROCESSED/image_ehr_crosswalk_20250418.csv")
    
    loader = CTPAImageLoader(raw_image_dir=IMAGE_DIR, crosswalk_path=CROSSWALK_PATH)
    
    # We won't actually fetch a batch unless the images exist.
    if os.path.exists(IMAGE_DIR):
        dataloader = loader.get_dataloader(batch_size=2)
        print("\n--- Pipeline Test ---")
        try:
            batch = next(iter(dataloader))
            images = batch["image"]
            print(f"Successfully loaded batch!")
            print(f"Image Tensor Shape: {images.shape} (Batch, Channel, Depth, Height, Width)")
            print(f"Patient IDs in batch: {batch['person_id']}")
        except Exception as e:
            print(f"Error loading batch: {e}")
    else:
        print(f"\n[!] Raw image directory not found at {IMAGE_DIR}.")
        print("[!] Please download the imaging dataset from Stanford AIMI before running the DataLoader.")
