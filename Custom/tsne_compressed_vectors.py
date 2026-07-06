import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

def main():
    # 1. Define paths
    compressed_dir = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/ctpa_vectors_compressed"
    cohort_path = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv"
    output_plot = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/tsne_compressed_pe_colored.png"
    
    # 2. Load Ground Truth Labels from Master Cohort
    print(f"Loading labels from {cohort_path}...")
    df = pd.read_csv(cohort_path)
    
    # Map image_id -> PE Status (True/False)
    # Note: 'pe_positive_nlp' is the ground truth column
    label_map = dict(zip(df['image_id'], df['pe_positive_nlp']))
    
    print(f"Found {len(label_map)} labeled image IDs in the master cohort.")

    # 3. Load the Compressed 50-dim Vectors
    print("Loading compressed vectors from disk...")
    files = [f for f in os.listdir(compressed_dir) if f.endswith('.pt')]
    
    vectors = []
    labels = []
    
    for f in tqdm(files, desc="Loading compressed .pt"):
        # Extract the image_id from the filename (e.g., "PE12345_ctpa_vector.pt" -> "PE12345")
        image_id = f.split("_ctpa_vector.pt")[0]
        
        # Only load it if we have a label for it
        if image_id in label_map:
            path = os.path.join(compressed_dir, f)
            try:
                vec = torch.load(path, map_location='cpu').numpy()
                vectors.append(vec)
                # Convert string 'True'/'False' or bools to integer 1/0
                is_pe = label_map[image_id]
                labels.append(1 if str(is_pe).lower() == 'true' else 0)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
    X = np.array(vectors)
    y = np.array(labels)
    
    print(f"\nSuccessfully loaded {X.shape[0]} compressed vectors.")
    print(f"Distribution: {np.sum(y)} PE Positive | {len(y) - np.sum(y)} PE Negative")

    # 4. Run t-SNE on the compressed [50-dim] data
    print("\nRunning t-SNE (this will be extremely fast on 50 dimensions!)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(X)
    
    # 5. Plot and save
    print("Plotting results...")
    plt.figure(figsize=(12, 10))
    
    # Scatter Negative Cases (Blue)
    plt.scatter(
        tsne_results[y == 0, 0], 
        tsne_results[y == 0, 1], 
        c='cornflowerblue', 
        label='PE Negative', 
        alpha=0.4, 
        s=10
    )
    
    # Scatter Positive Cases (Red) on top
    plt.scatter(
        tsne_results[y == 1, 0], 
        tsne_results[y == 1, 1], 
        c='crimson', 
        label='PE Positive', 
        alpha=0.6, 
        s=15
    )
    
    plt.title('t-SNE of Compressed 50-Dim CTPA Vectors (Pre-Fine-Tuning)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Done! Plot saved to: {output_plot}")

if __name__ == "__main__":
    main()
