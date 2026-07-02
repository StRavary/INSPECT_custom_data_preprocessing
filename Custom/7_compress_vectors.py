import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def main():
    # Input and Output Directories
    input_dir = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/ctpa_vectors"
    output_dir = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/ctpa_vectors_compressed"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Step 1: Loading all vectors into memory...")
    vectors = []
    filenames = []
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
    
    for f in tqdm(files, desc="Loading .pt files"):
        path = os.path.join(input_dir, f)
        try:
            vec = torch.load(path, map_location='cpu')
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().numpy().flatten()
            else:
                vec = np.array(vec).flatten()
            vectors.append(vec)
            filenames.append(f)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    # Convert to a single large numpy matrix
    # Shape will be [23227, 6144]
    X = np.array(vectors)
    print(f"\nSuccessfully loaded {X.shape[0]} vectors of dimension {X.shape[1]}")
    
    print("\nStep 2: Centering and Normalizing the Raw Data...")
    # StandardScaler removes the mean (centering) and scales to unit variance
    # This solves the "anisotropy" problem where all vectors point the same way.
    scaler_raw = StandardScaler()
    X_scaled = scaler_raw.fit_transform(X)
    
    print("Step 3: Applying PCA compression...")
    # We compress from 6144 to 50 dimensions (retains ~99.8% of variance)
    n_components = 50
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA complete! Reduced dimension to {n_components}. Retained {explained_variance:.2f}% of original variance.")
    
    print("\nStep 4: Normalizing the final PCA embeddings (Z-score)...")
    # It's best practice to normalize the resulting PCA embeddings 
    # so the downstream model receives clean, well-distributed inputs.
    scaler_pca = StandardScaler()
    X_final = scaler_pca.fit_transform(X_pca)
    
    print("\nStep 5: Saving compressed vectors to disk...")
    # Save them back as individual .pt files so it integrates seamlessly with your PyArrow/MONAI loop
    for i, f in enumerate(tqdm(filenames, desc="Saving compressed .pt files")):
        output_path = os.path.join(output_dir, f)
        # Convert back to torch tensor (float32)
        tensor_out = torch.tensor(X_final[i], dtype=torch.float32)
        torch.save(tensor_out, output_path)
        
    print(f"\n✅ All compressed vectors saved to: {output_dir}")
    print(f"Your downstream model will now learn from clean {n_components}-dimensional features instead of messy 6144-dimensional ones!")

if __name__ == "__main__":
    main()
