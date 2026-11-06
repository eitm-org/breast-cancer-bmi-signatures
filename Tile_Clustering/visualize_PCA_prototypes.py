import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import argparse
import os
from utils import get_vectors_from_h5, visualize_cluster_samples

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PCA+GMM clustering with prototype centroid initialization (2D scatter + combined grids)')
parser.add_argument('--clusters', type=int, default=2, help='Number of GMM clusters (default: 2)')
parser.add_argument('--tile-type', type=str, default='stroma', choices=['stroma', 'tumor', 'all'],
                    help='Type of tiles: stroma, tumor, or all (default: stroma)')
parser.add_argument('--pca-components', type=int, default=200, help='PCA components used for clustering (default: 50)')
parser.add_argument('--covariance-type', type=str, default='full', choices=['full', 'tied', 'diag', 'spherical'],
                    help='GMM covariance type (default: full)')
parser.add_argument('--reg-covar', type=float, default=1e-5, help='GMM regularization added to the diagonal of covariance (default: 1e-5)')
parser.add_argument('--max-iter', type=int, default=200, help='Maximum GMM iterations (default: 200)')
parser.add_argument('--prototype-centroids', type=str, default=None,
                    help='Path to prototype centroids CSV (shape 2xD). If omitted, uses defaults per tile-type.')
parser.add_argument('--confidence-threshold', type=float, default=0.7,
                    help='Confidence threshold for high-certainty samples (default: 0.7)')
parser.add_argument('--show-uncertain', action='store_true', help='Show uncertain boundary points in gray on PCA plot')
parser.add_argument('--figsize', type=int, nargs=2, default=[14, 10], help='Figure size for PCA scatter (default: 14 10)')
parser.add_argument('--dpi', type=int, default=150, help='DPI for PCA scatter output (default: 150)')
parser.add_argument('--output', type=str, default='pca_visualization.png', help='Output path for PCA scatter image')
parser.add_argument('--grid-output', type=str, default='cluster_samples.png', help='Output path for high-confidence cluster grid image')
parser.add_argument('--uncertain-grid-output', type=str, default='uncertain_samples.png', help='Output path for uncertain boundary tiles grid image')
parser.add_argument('--num-samples', type=int, default=100, help='Samples per cluster in grid (default: 50)')
parser.add_argument('--tiles-path', type=str, default='/path/to/tile_vectors_90_ROI',
                    help='Folder with tile-level vectors (tile_vectors_90_ROI)')
parser.add_argument('--base-image-path', type=str, default='/path/to/tile_images',
                    help='Base path to tile images for grid rendering')
parser.add_argument('--density', type=str, default='hex', choices=['hex', 'scatter'],
                    help="Visualization mode for PCA plot: 'hex' for density hexbin, 'scatter' for points (default: hex)")
parser.add_argument('--gridsize', type=int, default=100, help='Hexbin grid size for density visualization (default: 100)')
args = parser.parse_args()

CLUSTER_COUNT = args.clusters
TILES_PATH = args.tiles_path

# --------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------
print("[1/7] Loading tile list...")
tile_list_file = "all_tiles.txt"
if not os.path.exists(tile_list_file):
    raise FileNotFoundError(f"Tile list not found: {tile_list_file}")
with open(tile_list_file, "r") as f:
    tiles_of_interest = [line.strip() for line in f if line.strip()]
print(f"  Found {len(tiles_of_interest)} tiles")

print("\n[2/7] Loading tile vectors...")
tile_df = pd.DataFrame()

def build_h5_path(tile_id: str) -> str:
    if args.tile_type == 'stroma':
        tissue_dir = 'Negative'
        return f"{TILES_PATH}/{tile_id}/{tile_id}_01/{tissue_dir}/{tile_id}_01.h5"
    if args.tile_type == 'tumor':
        tissue_dir = 'Positive'
        return f"{TILES_PATH}/{tile_id}/{tile_id}_01/{tissue_dir}/{tile_id}_01.h5"
    # all: try Negative then Positive
    for tissue_dir in ['Negative', 'Positive']:
        candidate = f"{TILES_PATH}/{tile_id}/{tile_id}_01/{tissue_dir}/{tile_id}_01.h5"
        if os.path.exists(candidate):
            return candidate
    return None

for i, tile in enumerate(tiles_of_interest, 1):
    h5_path = build_h5_path(tile)
    if not h5_path:
        continue
    if i % 25 == 0 or i == 1:
        print(f"  [{i}/{len(tiles_of_interest)}] Reading: {h5_path}")
    try:
        df = get_vectors_from_h5(h5_path)
        tile_df = pd.concat([tile_df, df], ignore_index=True)
    except Exception as e:
        print(f"    Error loading {tile}: {e}")
        continue

print(f"\n✓ All files loaded: {len(tile_df)} total tile vectors")
if len(tile_df) == 0:
    print("Error: No tile vectors loaded!")
    raise SystemExit(1)

# --------------------------------------------------------------------
# PCA-high-D for clustering
# --------------------------------------------------------------------
print("\n[3/7] Extracting and standardizing features...")
X = tile_df.drop(columns=['ID']).values.astype(np.float32)
print(f"  Feature matrix shape: {X.shape}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("  Standardization complete")

print(f"\n[4/7] Applying PCA ({args.pca_components} components) for clustering...")
pca_hd = PCA(n_components=args.pca_components)
X_pca_hd = pca_hd.fit_transform(X_scaled)
D = X_pca_hd.shape[1]
print(f"  Reduced shape (clustering space): {X_pca_hd.shape}")
print(f"  Variance explained ({args.pca_components}D): {pca_hd.explained_variance_ratio_.sum():.4f}")

# --------------------------------------------------------------------
# Load prototype centroids (2xD) for means_init
# --------------------------------------------------------------------
def default_centroid_path(tile_type: str, pca_components: int) -> str:
    """Return centroid CSV path matching tile type and PCA dimensionality."""
    if tile_type == 'tumor':
        return os.path.join('centroids', f'tumor_centroids_kmeans2_pca{pca_components}.csv')
    if tile_type == 'stroma':
        return os.path.join('centroids', f'stroma_centroids_kmeans2_pca{pca_components}.csv')
    return None

centroid_path = args.prototype_centroids or default_centroid_path(args.tile_type, args.pca_components)
centroids = None
if centroid_path and os.path.exists(centroid_path):
    try:
        cdf = pd.read_csv(centroid_path)
        centroids = cdf.values.astype(np.float64)
        if centroids.shape != (CLUSTER_COUNT, D):
            print(f"  ⚠ Centroid dim mismatch: got {centroids.shape}, expected ({CLUSTER_COUNT}, {D}). Ignoring means_init.")
            centroids = None
        else:
            print(f"  ✓ Loaded centroids from {centroid_path} (shape {centroids.shape})")
    except Exception as e:
        print(f"  ⚠ Failed to load centroids: {e}. Proceeding without means_init.")
else:
    if centroid_path:
        print(f"  ⚠ Centroid file not found: {centroid_path}. Proceeding without means_init.")
    else:
        print("  ℹ No centroid file provided for 'all' tile-type. Proceeding without means_init.")

# --------------------------------------------------------------------
# GMM clustering with means_init (if available)
# --------------------------------------------------------------------
print(f"\n[5/7] Running GMM clustering (k={CLUSTER_COUNT}, covariance={args.covariance_type})...")
gmm = GaussianMixture(
    n_components=CLUSTER_COUNT,
    covariance_type=args.covariance_type,
    means_init=centroids,
    reg_covar=args.reg_covar,
    max_iter=args.max_iter,
    random_state=42,
    init_params='kmeans'
)

gmm.fit(X_pca_hd)
probs = gmm.predict_proba(X_pca_hd)
cluster_labels = probs.argmax(axis=1)
confidence = probs.max(axis=1)

print("  GMM complete")

tile_df['kmeans_cluster'] = cluster_labels
tile_df['confidence'] = confidence

# Confidence analysis
print("\n  Confidence analysis:")
print(f"    Mean confidence: {confidence.mean():.3f}")
print(f"    Median confidence: {np.median(confidence):.3f}")
for threshold in [0.7, 0.8, 0.9, 0.95]:
    pct = (confidence >= threshold).mean() * 100
    print(f"    Samples with confidence ≥ {threshold}: {pct:.1f}%")

high_conf_mask = confidence >= args.confidence_threshold
print(f"\n  Using confidence threshold: {args.confidence_threshold}")
print(f"    High-confidence samples: {high_conf_mask.sum()} ({high_conf_mask.mean():.1%})")
print(f"    Boundary/uncertain: {(~high_conf_mask).sum()} ({(~high_conf_mask).mean():.1%})")

# Save CSV with confidence
out_csv = f"{args.tile_type}_cluster_assignments.csv"
tile_df[['ID', 'kmeans_cluster', 'confidence']].to_csv(out_csv, index=False)
print(f"  ✓ Saved: {out_csv} (with confidence)")

# --------------------------------------------------------------------
# PCA to 2D for visualization (density hexbin)
# --------------------------------------------------------------------
print("\n[6/7] Creating 2D PCA scatter...")
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=tuple(args.figsize))
# Explicit distinct colors for up to 8 clusters
cluster_colors = ['#FF0000', '#FF8C00', '#00B050', '#7030A0', '#FFC000', '#00B0F0', '#C00000', '#92D050']
colors = [cluster_colors[i % len(cluster_colors)] for i in range(CLUSTER_COUNT)]

def make_fade_cmap(hex_color: str):
    rgb = to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(
        'fade_'+hex_color,
        [(0.0, (*rgb, 0.0)), (1.0, (*rgb, 1.0))], N=256
    )

# Density visualization: cores first, optional uncertain points
if args.density == 'hex':
    handles, labels = [], []
    for i in range(CLUSTER_COUNT):
        mask = (cluster_labels == i) & high_conf_mask
        if not np.any(mask):
            continue
        cmap_i = make_fade_cmap(colors[i])
        ax.hexbin(
            X_pca_2d[mask, 0], X_pca_2d[mask, 1],
            gridsize=args.gridsize, cmap=cmap_i, bins='log', mincnt=1, linewidths=0
        )
        handles.append(Patch(facecolor=colors[i], edgecolor='none', label=f'Cluster {i} (core)'))
        labels.append(f'Cluster {i} (core)')
    if args.show_uncertain and (~high_conf_mask).any():
        ax.scatter(X_pca_2d[~high_conf_mask, 0], X_pca_2d[~high_conf_mask, 1],
                   c='lightgray', s=2, alpha=0.3, edgecolors='none', label='Uncertain')
        handles.append(Patch(facecolor='lightgray', edgecolor='none', label='Uncertain'))
        labels.append('Uncertain')
    if handles:
        ax.legend(handles=handles, labels=labels, loc='best', framealpha=0.9)
else:
    # Scatter mode
    for i in range(CLUSTER_COUNT):
        mask = (cluster_labels == i) & high_conf_mask
        ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], c=colors[i], s=8, alpha=0.5, edgecolors='none', label=f'Cluster {i} (core)')
    if args.show_uncertain and (~high_conf_mask).any():
        ax.scatter(X_pca_2d[~high_conf_mask, 0], X_pca_2d[~high_conf_mask, 1], c='lightgray', s=4, alpha=0.3, edgecolors='none', label='Uncertain')
    ax.legend(loc='best', framealpha=0.9, markerscale=2)

ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)")
tissue_label = args.tile_type.capitalize() if args.tile_type != 'all' else 'All'
ax.set_title(f"PCA Scatter of {tissue_label} Tiles\nGMM k={CLUSTER_COUNT} in {D}D PCA space (means_init={'yes' if centroids is not None else 'no'})")
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
print(f"  ✓ PCA scatter saved to: {args.output}")

# --------------------------------------------------------------------
# Combined grids: high-confidence cores and uncertain boundary
# --------------------------------------------------------------------
print("\n[7/7] Rendering cluster grids...")
tile_df_high = tile_df[tile_df['confidence'] >= args.confidence_threshold]
tile_df_unc = tile_df[tile_df['confidence'] < args.confidence_threshold]

if len(tile_df_high) > 0:
    print("  Creating high-confidence core grid...")
    visualize_cluster_samples(tile_df_high, args.base_image_path, args.grid_output, num_samples=args.num_samples, tissue_type=args.tile_type)
    print(f"  ✓ High-confidence grid saved to: {args.grid_output} (n={len(tile_df_high)})")
else:
    print(f"  ⚠ No high-confidence samples with threshold {args.confidence_threshold}; skipping core grid.")

if len(tile_df_unc) > 0:
    print("  Creating uncertain boundary tiles grid...")
    visualize_cluster_samples(tile_df_unc, args.base_image_path, args.uncertain_grid_output, num_samples=args.num_samples, tissue_type=args.tile_type)
    print(f"  ✓ Uncertain boundary grid saved to: {args.uncertain_grid_output} (n={len(tile_df_unc)})")
else:
    print(f"  ⚠ No uncertain samples below threshold {args.confidence_threshold}.")

print("\nDone.")
