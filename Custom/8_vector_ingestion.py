import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalVectorLoader(Dataset):
    """
    High-Speed Multimodal Dataset.
    Loads compressed [50-dim] CTPA image vectors and merges them 
    with corresponding EHR tabular/temporal data.
    """
    def __init__(self, vector_dir: str, crosswalk_path: str, ehr_df: pd.DataFrame = None):
        """
        Args:
            vector_dir: Path to the compressed `.pt` ctpa vectors.
            crosswalk_path: Path to the crosswalk CSV mapping person_id to image_id.
            ehr_df: Optional PyArrow-backed pandas DataFrame containing EHR features for each person_id.
        """
        self.vector_dir = vector_dir
        self.crosswalk_path = crosswalk_path
        self.ehr_df = ehr_df
        
        print(f"Loading Crosswalk mapping from {self.crosswalk_path}...")
        df_crosswalk = pd.read_csv(self.crosswalk_path)
        
        self.data = []
        missing_count = 0
        
        print("Validating vector files on disk...")
        for _, row in df_crosswalk.iterrows():
            person_id = row['person_id']
            image_id = row['image_id']
            
            # The compressed vectors have "_ctpa_vector.pt" appended
            expected_filename = f"{image_id}_ctpa_vector.pt"
            expected_path = os.path.join(self.vector_dir, expected_filename)
            
            if os.path.exists(expected_path):
                # We do NOT load the vector here. We just store the path to keep RAM free.
                # It gets lazy-loaded during __getitem__ by the background workers.
                entry = {
                    "vector_path": expected_path,
                    "person_id": person_id,
                    "image_id": image_id
                }
                self.data.append(entry)
            else:
                missing_count += 1
                
        print(f"Successfully mapped {len(self.data)} valid patient-image pairs.")
        if missing_count > 0:
            print(f"Warning: {missing_count} vectors mapped in crosswalk were not found on disk.")

        # If an EHR dataframe is provided, index it by person_id for O(1) lookups
        if self.ehr_df is not None:
            self.ehr_df.set_index('person_id', inplace=True)
            print("EHR Tabular data indexed and ready for fusion.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        person_id = entry["person_id"]
        
        # 1. Load the pre-compressed Image Vector
        # Shape: [50] (or whatever PCA dimension you used)
        image_vector = torch.load(entry["vector_path"], map_location='cpu')
        
        output_dict = {
            "image_vector": image_vector,
            "person_id": person_id,
            "image_id": entry["image_id"]
        }
        
        # 2. Extract and append EHR Features (if provided)
        if self.ehr_df is not None:
            try:
                # Fetch row for this patient
                patient_ehr_data = self.ehr_df.loc[person_id]
                # Convert the specific feature columns to a tensor
                # Example: exclude ID columns and get values
                ehr_features = patient_ehr_data.drop(['image_id'], errors='ignore').values
                output_dict["ehr_vector"] = torch.tensor(ehr_features, dtype=torch.float32)
            except KeyError:
                # If patient is not found in EHR dataset, return zeros or handle missing data
                pass
                
        return output_dict

def get_dataloader(vector_dir: str, crosswalk_path: str, ehr_df: pd.DataFrame = None, batch_size=32, num_workers=4):
    """Helper function to spawn the multi-threaded DataLoader."""
    dataset = MultimodalVectorLoader(vector_dir, crosswalk_path, ehr_df)
    
    print(f"Spawning DataLoader with {num_workers} background workers and batch size {batch_size}...")
    # Because these vectors are tiny (50 dims), we can easily push huge batch sizes!
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available() # Speeds up transfer to GPU
    )

if __name__ == "__main__":
    # Example usage / Sanity check
    COMPRESSED_VECTOR_DIR = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/ctpa_vectors_compressed"
    CROSSWALK_PATH = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/image_ehr_crosswalk_20250418.csv"
    
    loader = get_dataloader(
        vector_dir=COMPRESSED_VECTOR_DIR, 
        crosswalk_path=CROSSWALK_PATH,
        batch_size=256 # Can be massive now!
    )
    
    print("\n--- Multimodal Ingestion Test ---")
    try:
        batch = next(iter(loader))
        images = batch["image_vector"]
        patients = batch["person_id"]
        
        print(f"Successfully loaded batch!")
        print(f"Image Vector Tensor Shape: {images.shape} (Batch, Features)")
        print(f"First 5 Patient IDs in batch: {patients[:5].tolist()}")
        print("\nYou are now ready to feed `batch['image_vector']` and `batch['ehr_vector']` into your PyTorch Network!")
    except Exception as e:
        print(f"Error loading batch: {e}")
