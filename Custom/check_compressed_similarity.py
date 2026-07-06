import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

def main():
    compressed_dir = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/ctpa_vectors_compressed"
    files = [f for f in os.listdir(compressed_dir) if f.endswith('.pt')]
    
    # Subsample 2000 files to match the original analysis scale and keep computation fast
    random.seed(42)
    sample_files = random.sample(files, min(2000, len(files)))
    
    vectors = []
    for f in sample_files:
        path = os.path.join(compressed_dir, f)
        vec = torch.load(path, map_location='cpu').numpy()
        vectors.append(vec)
        
    X = np.array(vectors)
    print(f"Loaded {X.shape[0]} compressed vectors of dimension {X.shape[1]}")
    
    # Compute pairwise cosine similarity
    print("Computing pairwise cosine similarities...")
    sim_matrix = cosine_similarity(X)
    
    # Extract upper triangle to avoid self-similarity (1.0 on diagonal) and duplicates
    upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[upper_tri_indices]
    
    print("\nCosine Similarity stats for PCA'd [50-dim] vectors:")
    print(f"  Mean: {np.mean(pairwise_sims):.4f}")
    print(f"  Std:  {np.std(pairwise_sims):.4f}")
    print(f"  Min:  {np.min(pairwise_sims):.4f}")
    print(f"  Max:  {np.max(pairwise_sims):.4f}")

if __name__ == "__main__":
    main()
