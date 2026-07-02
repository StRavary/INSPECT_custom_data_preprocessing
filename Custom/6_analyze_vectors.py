import os
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    data_dir = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/ctpa_vectors"
    output_dir = "/home/steven/Documents/Internship_INSPECT/DATA_PROCESSED/Ablation study ResnetV2-101x3 vectors/analysis_results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading vectors...")
    vectors = []
    filenames = []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    
    # Load vectors (limit to a reasonable number to avoid memory issues, or load all if not too big)
    max_files = 23227
    if len(files) > max_files:
        print(f"Found {len(files)} vectors. Subsampling {max_files} for analysis...")
        np.random.seed(42)
        files = np.random.choice(files, max_files, replace=False)
        
    for f in tqdm(files):
        path = os.path.join(data_dir, f)
        try:
            # Vectors might be stored with grad or on GPU, move to cpu and detach
            vec = torch.load(path, map_location='cpu')
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().numpy().flatten()
            else:
                vec = np.array(vec).flatten()
            vectors.append(vec)
            filenames.append(f)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    vectors = np.array(vectors)
    print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
    
    # 1. Cosine Similarity Analysis
    print("Computing pairwise cosine similarities...")
    # Calculate for a subset to avoid huge distance matrices
    subset_size = min(1000, vectors.shape[0])
    idx = np.random.choice(vectors.shape[0], subset_size, replace=False)
    sub_vectors = vectors[idx]
    
    sim_matrix = cosine_similarity(sub_vectors)
    # Get upper triangle excluding diagonal
    upper_tri = sim_matrix[np.triu_indices(subset_size, k=1)]
    
    print(f"Cosine Similarity stats:")
    print(f"  Mean: {np.mean(upper_tri):.4f}")
    print(f"  Std:  {np.std(upper_tri):.4f}")
    print(f"  Min:  {np.min(upper_tri):.4f}")
    print(f"  Max:  {np.max(upper_tri):.4f}")
    
    # Save cosine similarity histogram
    plt.figure(figsize=(10, 6))
    plt.hist(upper_tri, bins=50, alpha=0.75, color='blue')
    plt.title('Pairwise Cosine Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cosine_similarity_hist.png'))
    plt.close()
    
    # 2. PCA Analysis
    print("Running PCA...")
    # Standardize data roughly (zero mean)
    vectors_centered = vectors - np.mean(vectors, axis=0)
    
    n_components = min(50, vectors.shape[1], vectors.shape[0])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(vectors_centered)
    
    explained_variance = pca.explained_variance_ratio_
    cum_explained_variance = np.cumsum(explained_variance)
    
    print(f"PCA top 5 explained variance: {explained_variance[:5]}")
    print(f"PCA cumulative explained variance (top {n_components}): {cum_explained_variance[-1]:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cum_explained_variance) + 1), cum_explained_variance, marker='o', linestyle='-', color='b')
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    plt.close()
    
    # 3. K-Means Clustering (for coloring t-SNE)
    print("Running K-Means clustering (K=5)...")
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vectors)
    
    # 4. t-SNE Visualization
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='tab10', alpha=0.6, s=15)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title('t-SNE Visualization of CTPA Vectors')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    plt.close()
    
    print(f"Analysis complete. Results and plots saved to {output_dir}")

if __name__ == "__main__":
    main()
