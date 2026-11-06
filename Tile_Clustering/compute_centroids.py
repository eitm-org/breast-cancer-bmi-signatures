import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import get_vectors_from_h5

# Path to the tile-level vectors (tile_vectors_90_ROI); placeholder, set with --tiles-base
TILES_BASE = "/path/to/tile_vectors_90_ROI"


def build_h5_path(tile_id: str, tissue_type: str) -> str:
    """Construct H5 path for a tile ID based on tissue type."""
    tissue_dir = "Positive" if tissue_type == "tumor" else "Negative"
    return f"{TILES_BASE}/{tile_id}/{tile_id}_01/{tissue_dir}/{tile_id}_01.h5"


def load_vectors_for_list(tile_list_file: str, tissue_type: str) -> pd.DataFrame:
    """Load all vectors for tiles listed in a file for given tissue type."""
    if not os.path.exists(tile_list_file):
        raise FileNotFoundError(f"Tile list not found: {tile_list_file}")
    with open(tile_list_file, "r") as f:
        tiles = [line.strip() for line in f if line.strip()]
    print(f"  Found {len(tiles)} tiles in {tile_list_file}")

    df_all = pd.DataFrame()
    for i, tile in enumerate(tiles, 1):
        h5_path = build_h5_path(tile, tissue_type)
        try:
            df = get_vectors_from_h5(h5_path)
            df_all = pd.concat([df_all, df], ignore_index=True)
        except Exception as e:
            print(f"    Error loading {h5_path}: {e}")
        if i % 25 == 0 or i == 1:
            print(f"    [{i}/{len(tiles)}] Loaded; total vectors: {len(df_all)}")
    print(f"  ✓ Total vectors for {tissue_type}: {len(df_all)}")
    return df_all


def compute_centroids_kmeans_pca(df: pd.DataFrame, n_components: int, random_state: int = 42) -> np.ndarray:
    """Standardize, reduce to n_components PCA, cluster k=2, return centroids (2 x n_components)."""
    X = df.drop(columns=["ID"]).values.astype(np.float32)
    print(f"    Feature matrix: {X.shape}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    Xp = pca.fit_transform(Xs)
    print(f"    PCA({n_components}) complete. Variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=10)
    kmeans.fit(Xp)
    centers = kmeans.cluster_centers_  # shape (2, n_components)
    return centers


def save_centroids_csv(centroids: np.ndarray, out_path: str):
    """Save (2, 50) centroid array to CSV with column names 0..49."""
    cols = [str(i) for i in range(centroids.shape[1])]
    df = pd.DataFrame(centroids, columns=cols)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved centroids: {out_path} (shape {centroids.shape})")


def main():
    global TILES_BASE
    parser = argparse.ArgumentParser(description="Compute KMeans (k=2) centroids in PCA space for tumor and stroma TOI")
    parser.add_argument("--tiles-base", type=str, default=TILES_BASE, help="Folder with tile-level vectors (tile_vectors_90_ROI)")
    parser.add_argument("--tumor-toi", type=str, default="tumor_toi.txt", help="Path to tumor tiles-of-interest list")
    parser.add_argument("--stroma-toi", type=str, default="stroma_toi.txt", help="Path to stroma tiles-of-interest list")
    parser.add_argument("--pca-components", type=int, default=50, help="Number of PCA components for centroid computation (e.g., 50 or 200)")
    parser.add_argument("--output-dir", type=str, default="centroids", help="Directory to save centroid CSVs")
    args = parser.parse_args()
    TILES_BASE = args.tiles_base

    # Tumor
    print("\n[1/2] Tumor: loading vectors and computing centroids...")
    tumor_df = load_vectors_for_list(args.tumor_toi, tissue_type="tumor")
    tumor_centroids = compute_centroids_kmeans_pca(tumor_df, n_components=args.pca_components)
    tumor_out = os.path.join(args.output_dir, f"tumor_centroids_kmeans2_pca{args.pca_components}.csv")
    save_centroids_csv(tumor_centroids, tumor_out)

    # Stroma
    print("\n[2/2] Stroma: loading vectors and computing centroids...")
    stroma_df = load_vectors_for_list(args.stroma_toi, tissue_type="stroma")
    stroma_centroids = compute_centroids_kmeans_pca(stroma_df, n_components=args.pca_components)
    stroma_out = os.path.join(args.output_dir, f"stroma_centroids_kmeans2_pca{args.pca_components}.csv")
    save_centroids_csv(stroma_centroids, stroma_out)

    print("\nDone. Centroids saved:")
    print(f"  - {tumor_out}")
    print(f"  - {stroma_out}")


if __name__ == "__main__":
    main()
