"""Library of file access functions."""

import pandas as pd
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


def parse_tile_id(tile_id: str) -> tuple:
    """
    Parse tile ID to extract tumor ID and tile base name.
    
    Args:
        tile_id: Tile ID string (e.g., 'EITB-000AP_L1_R034580_C063840')
        
    Returns:
        Tuple of (tumor_id, tumor_base, tile_id)
        - tumor_id: e.g., 'EITB-000AP'
        - tumor_base: e.g., 'EITB-000AP_01' (used in directory structure)
        - tile_id: original tile ID
        
    Example:
        'EITB-000AP_L1_R034580_C063840' -> ('EITB-000AP', 'EITB-000AP_01', 'EITB-000AP_L1_R034580_C063840')
        Note: Directory uses _01 but tile ID contains _L1
    """
    parts = tile_id.split('_')
    if len(parts) >= 2:
        # Extract tumor ID (e.g., EITB-000AP from 'EITB-000AP_L1_R...')
        tumor_id = parts[0]  # e.g., EITB-000AP
        
        # The directory structure uses _01, not _L1
        tumor_base = f"{tumor_id}_01"  # e.g., EITB-000AP_01
        
        return tumor_id, tumor_base, tile_id
    return None, None, tile_id


def get_tiff_path(tile_id: str, base_image_path: str) -> str:
    """
    Generate the local file path for a TIFF image.
    
    Args:
        tile_id: Tile ID string (e.g., 'EITB-000AP_L1_R034580_C063840')
        base_image_path: Base path to the images directory
                        (e.g., '/path/to/tile_images')
        
    Returns:
        Full path to the TIFF file
        
    Example:
        get_tiff_path('EITB-000A7_L1_R042560_C036100', '/path/to/tile_images')
        -> '/path/to/tile_images/EITB-000A7/EITB-000A7_01/Normalized/EITB-000A7_L1_R042560_C036100.tiff'
    """
    tumor_id, tumor_base, tile_name = parse_tile_id(tile_id)
    
    if not tumor_id or not tumor_base:
        return None
    
    # Construct path: base_path/tumor_id/tumor_base/Normalized/tile_id.tiff
    tiff_path = f"{base_image_path}/{tumor_id}/{tumor_base}/Normalized/{tile_name}.tiff"
    
    return tiff_path


def get_vectors_from_h5(h5_filepath: str) -> pd.DataFrame:
    """Read a h5 file, extract vectors and IDs, and create a pandas dataframe."""

    with h5py.File(h5_filepath, "r") as f1:
        if 'ids' not in f1.keys():
            raise ValueError("The vector h5 file must have an 'ids' key.")
        if 'vectors' not in f1.keys():
            raise ValueError("The vector h5 file must have a 'vectors' key.")

        id_arr = f1['ids'][:]
        vec_arr = f1['vectors'][:]

        if len(id_arr) != len(vec_arr):
            raise ValueError("The number of IDs in file does not match the number of vectors in file.")

    id_list = [idval.decode('utf-8') for idval in id_arr]
    id_df = pd.DataFrame(id_list, columns=['ID'])

    n_cols = vec_arr.shape[1]
    cols = [str(val) for val in range(n_cols)]
    vec_df = pd.DataFrame(vec_arr, columns=cols)

    res_df = pd.concat([id_df, vec_df], axis=1)
    return res_df


# write a function that takes in either stroma or tumor as an argument and a 50 dimensional vector. if stroma read 2 csvs stroma_dense and stroma_notDense. 
# if tumor read 2 csvs  tumor_dense and tumor_notDense. then compare the passed vector with the two reference vectors that we4re read. 
# return the name of the file (minus the .csv) as a label for that class. this will be used to name the clusters output by the clustering_scripts.py

def classify_vector(tissue_type: str, vector: np.ndarray) -> str:
    """
    Classify a 50-dimensional vector by comparing it to reference vectors.
    
    Args:
        tissue_type: Either "stroma" or "tumor"
        vector: A 50-dimensional numpy array
        
    Returns:
        The name of the reference class (e.g., "stroma_dense", "tumor_notDense") 
        that the input vector is most similar to
        
    Raises:
        ValueError: If tissue_type is not "stroma" or "tumor", or if vector is not 50-dimensional
    """
    if tissue_type not in ("stroma", "tumor"):
        raise ValueError("tissue_type must be either 'stroma' or 'tumor'")
    
    # Allow vectors with at least 50 dims (e.g., PCA=100); use first 50 to match references
    if len(vector) < 50:
        raise ValueError(f"vector must be at least 50-dimensional, got {len(vector)}")
    if len(vector) > 50:
        vector = vector[:50]
    
    # Determine which CSV files to read
    if tissue_type == "stroma":
        file1 = "stroma_dense.csv"
        file2 = "stroma_notDense.csv"
    else:  # tumor
        file1 = "tumor_Dense.csv"
        file2 = "tumor_notDense.csv"
    
    # Read reference vectors from CSV files (first row only)
    ref_vec1 = pd.read_csv(file1).iloc[0].values
    ref_vec2 = pd.read_csv(file2).iloc[0].values
    
    # Reshape vectors for cosine similarity calculation
    input_vec_reshaped = vector.reshape(1, -1)
    ref_vec1_reshaped = ref_vec1.reshape(1, -1)
    ref_vec2_reshaped = ref_vec2.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity1 = cosine_similarity(input_vec_reshaped, ref_vec1_reshaped)[0][0]
    similarity2 = cosine_similarity(input_vec_reshaped, ref_vec2_reshaped)[0][0]
    
    # Return "Dense" or "NotDense" based on which reference vector is more similar
    if similarity1 > similarity2:
        return "Dense"
    else:
        return "NotDense"


def classify_clusters(tissue_type: str, vectors: list) -> dict:
    """
    Assign string labels based on cluster IDs only (no references).

    Args:
        tissue_type: Either "stroma" or "tumor" (kept for API compatibility)
        vectors: A list of cluster centroids (only length is used)

    Returns:
        A dictionary mapping cluster index to its string ID, e.g., {0: "0", 1: "1"}.
    """
    num_clusters = len(vectors)
    return {i: str(i) for i in range(num_clusters)}


def visualize_cluster_samples(dataframe: pd.DataFrame, base_image_path: str, output_path: str, num_samples: int = 12, tissue_type: str = "tissue") -> None:
    """
    Create visualization grids of actual tile images from each cluster.
    
    Args:
        dataframe: DataFrame with 'kmeans_cluster' column and ID column
        base_image_path: Base path to the TIFF images 
                        (e.g., '/path/to/tile_images')
        output_path: Path to save the visualization PNG
        num_samples: Number of samples to show per cluster (default 12)
        tissue_type: Type of tissue ("tumor" or "stroma") for labeling (default: "tissue")
    """
    from PIL import Image
    import math
    
    # Get unique clusters
    clusters = sorted(dataframe['kmeans_cluster'].unique())
    num_clusters = len(clusters)
    
    # Calculate near-square grid dimensions based on requested samples
    cols = int(math.ceil(math.sqrt(num_samples)))
    rows = int(math.ceil(num_samples / cols))
    tile_size = 256  # Higher per-tile resolution to reduce blur
    DPI = 200        # Higher DPI for crisper saved images

    # Compute figure size to closely match pixel dimensions and avoid downscaling
    width_px = cols * tile_size
    height_px = num_clusters * rows * tile_size
    fig_width_in = max(1.0, width_px / DPI)
    fig_height_in = max(1.0, height_px / DPI)
    fig, axes = plt.subplots(num_clusters, 1, figsize=(fig_width_in, fig_height_in))
    if num_clusters == 1:
        axes = [axes]
    
    # Process each cluster
    for cluster_idx, cluster_id in enumerate(clusters):
        cluster_data = dataframe[dataframe['kmeans_cluster'] == cluster_id]
        cluster_samples = cluster_data.sample(n=min(num_samples, len(cluster_data)), random_state=42)
        
        # Load and resize images
        cluster_images = []
        for tile_id in cluster_samples['ID'].values[:num_samples]:
            tiff_path = get_tiff_path(tile_id, base_image_path)
            if tiff_path and os.path.exists(tiff_path):
                try:
                    img = Image.open(tiff_path).convert('RGB')
                    img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    cluster_images.append(np.array(img))
                except Exception as e:
                    print(f"Error loading {tiff_path}: {e}")
                    cluster_images.append(np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 128)
            else:
                if tiff_path:
                    print(f"File not found: {tiff_path}")
                cluster_images.append(np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 128)
        
        # Pad if fewer images than expected
        while len(cluster_images) < rows * cols:
            cluster_images.append(np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255)
        
        # Create grid for this cluster
        cluster_grid = np.zeros((tile_size * rows, tile_size * cols, 3), dtype=np.uint8)
        for i, img in enumerate(cluster_images[:rows * cols]):
            row = i // cols
            col = i % cols
            cluster_grid[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = img
        
        # Display grid
        axes[cluster_idx].imshow(cluster_grid, interpolation='nearest')
        axes[cluster_idx].set_title(f'{tissue_type.capitalize()} Cluster {cluster_id} (n={len(cluster_data)})', 
                        fontsize=14, fontweight='bold')
        axes[cluster_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cluster visualization to {output_path}")


def save_classified_tiles_to_h5(dataframe: pd.DataFrame, base_output_path: str) -> None:
    """
    Save classified tiles (Dense and NotDense) to separate h5 files for each sample.
    
    Args:
        dataframe: DataFrame with 'classification' and 'ID' columns and original vector dimensions
        base_output_path: Base path where to save files
                         Files will be saved to: base_output_path/{sample_name}/{sample_name}_01/{Dense or NotDense}/{sample_name}_01.h5
    """
    
    # Extract unique samples from tile IDs
    samples = set()
    for tile_id in dataframe['ID'].values:
        sample_name, _, _ = parse_tile_id(tile_id)
        if sample_name:
            samples.add(sample_name)
    
    # Process each sample separately
    for sample_name in sorted(samples):
        # Filter data for this sample
        sample_mask = dataframe['ID'].str.startswith(sample_name)
        sample_df = dataframe[sample_mask]

        # Group by kmeans_cluster for this sample
        for cluster_id in sorted(sample_df['kmeans_cluster'].unique()):
            cluster_df = sample_df[sample_df['kmeans_cluster'] == cluster_id]

            if len(cluster_df) == 0:
                print(f"No tiles found for sample '{sample_name}' in cluster {cluster_id}")
                continue

            # Extract IDs and vectors (all numeric columns are vectors - original full-dimensional)
            ids = cluster_df['ID'].values
            vector_cols = [col for col in cluster_df.columns if col not in ['ID', 'kmeans_cluster', 'classification']]
            vectors = cluster_df[vector_cols].values

            # Create output directory if it doesn't exist
            output_dir = f"{base_output_path}/{sample_name}/{sample_name}_01/cluster_{cluster_id}"
            os.makedirs(output_dir, exist_ok=True)

            # Create h5 file with same structure as input files
            output_file = f"{output_dir}/{sample_name}_01.h5"

            with h5py.File(output_file, 'w') as f:
                # Store IDs as encoded strings
                id_bytes = [id_str.encode('utf-8') for id_str in ids]
                f.create_dataset('ids', data=id_bytes)
                # Store vectors as float array (original full-dimensional)
                f.create_dataset('vectors', data=vectors.astype(np.float32))

            print(f"Saved {len(cluster_df)} tiles for sample '{sample_name}' to {output_file} (cluster_{cluster_id})")