# Tile-level cellular-density clustering (tumor / stroma HD and LD)

This folder contains the code that splits tumor and stromal tiles into high- and low-cellular-density groups (manuscript Methods 4.7 and Figure 1D). Its output defines the four subregions used elsewhere in this repository:

| Name in code and data | Name in manuscript |
|---|---|
| `TumorDense` | tumor HD |
| `TumorNotDense` | tumor LD |
| `StromaDense` | stroma HD |
| `StromaNotDense` | stroma LD |

The code comes from the internal repository `Tile-Clustering-Pipeline` (commit `60f9d49`, 2025-12-22). Only file paths were changed; they are now command-line options.

## Pipeline

Run the three scripts from inside `Tile_Clustering/`: they read `all_tiles.txt`, `tumor_toi.txt`, `stroma_toi.txt` and `centroids/` from the current folder.

```
tile vectors (1536-dim)          compute_centroids.py            visualize_PCA_prototypes.py          reorganize_h5_by_confidence.py
of the initialization slides ──► 2 starting centroids ──────────► GMM on all tiles of 304 slides ───► tiles with confidence >= 0.7,
                                 (centroids/*_pca50.csv)          {tumor,stroma}_cluster_assignments   split into Dense / NotDense
                                                                                                                │
                                                                    CanvOI slide encoder (proprietary, not included)
                                                                                                                ▼
                                                                    slide-level vectors per subregion (2000-dim), used by
                                                                    AUC_Boxplot_Notebooks and Distance_Metrics_Fig_4
```

### Step 1: starting centroids (`compute_centroids.py`)

Reads the tiles of the initialization slides (`tumor_toi.txt`: 4 slides; `stroma_toi.txt`: 7 slides). It standardizes the tile vectors, reduces them with PCA (50 components), and runs k-means (k = 2, `random_state=42`, `n_init=10`). The two cluster centres are saved to `centroids/`.

```bash
python compute_centroids.py --tiles-base /path/to/tile_vectors_90_ROI --pca-components 50
```

The centroid files produced for the manuscript are included (`centroids/tumor_centroids_kmeans2_pca50.csv`, `centroids/stroma_centroids_kmeans2_pca50.csv`), so this step can be skipped.

### Step 2: cluster all tiles (`visualize_PCA_prototypes.py`)

Reads all tiles of one tissue type for the 304 slides in `all_tiles.txt`. It fits its own standardization and PCA (50 components) on these tiles, then fits a two-component Gaussian mixture model (`reg_covar=1e-5`, `max_iter=200`, `random_state=42`). The GMM starting means are the step 1 centroids. Each tile is assigned to its most probable component; `confidence` is that probability.

Run with 50 PCA components and a confidence threshold of 0.7, as in the manuscript. The script's default is 200 components, so pass `--pca-components 50` explicitly:

```bash
python visualize_PCA_prototypes.py --tile-type tumor  --pca-components 50 --confidence-threshold 0.7 \
    --tiles-path /path/to/tile_vectors_90_ROI --output tumor_pca.png
python visualize_PCA_prototypes.py --tile-type stroma --pca-components 50 --confidence-threshold 0.7 \
    --tiles-path /path/to/tile_vectors_90_ROI --output stroma_pca.png
```

Output: `tumor_cluster_assignments.csv` and `stroma_cluster_assignments.csv`, with columns `ID` (tile), `kmeans_cluster` (the GMM component, 0 or 1) and `confidence`. The script also draws a 2-D PCA plot and grids of example tiles.

### Step 3: split tiles into subregions (`reorganize_h5_by_confidence.py`)

Keeps tiles with `confidence >= 0.7` and writes one H5 file per slide and subregion to `<dest>/<SLIDE>/<SLIDE>_01/<TYPE>/<SLIDE>_01.h5`. The components were mapped to subregions by visual inspection of example tiles (Supplementary Figures S3 and S4):

| Tissue | Component 0 | Component 1 |
|---|---|---|
| tumor | `TumorDense` | `TumorNotDense` |
| stroma | `StromaNotDense` | `StromaDense` |

```bash
python reorganize_h5_by_confidence.py --tissue tumor  --confidence-threshold 0.7 \
    --source /path/to/tile_vectors_90_ROI --dest /path/to/tile_vectors_90_ROI_cluster
python reorganize_h5_by_confidence.py --tissue stroma --confidence-threshold 0.7 \
    --source /path/to/tile_vectors_90_ROI --dest /path/to/tile_vectors_90_ROI_cluster
```

### Step 4: slide-level vectors (not included)

The tiles of each slide and subregion are aggregated into one 2000-dimensional slide vector with the CanvOI slide encoder, which is proprietary (see Code availability in the manuscript). The resulting slide vectors are provided in the dataset, for 291 (`TumorDense`), 304 (`TumorNotDense`), 304 (`StromaDense`) and 299 (`StromaNotDense`) slides.

## Inputs

- **Tile vectors**: `tile_vectors_90_ROI/<SLIDE>/<SLIDE>_01/Positive/<SLIDE>_01.h5` (tumor tiles) and `.../Negative/<SLIDE>_01.h5` (stroma tiles). Each file has an `ids` array (tile IDs such as `EITB-0000B_L1_R002280_C030020`) and a `vectors` array (1536 values per tile). These are provided in the Zenodo dataset ([10.5281/zenodo.23992256](https://doi.org/10.5281/zenodo.23992256)).
- **Slide lists**: `all_tiles.txt` (the 304 slides, one ID per line), `tumor_toi.txt` and `stroma_toi.txt` (initialization slides).
- **Tile images** (optional): used only to draw the example-tile grids (`--base-image-path`). They are not distributed; without them the grids show grey placeholders.

## Files

| File | Role |
|---|---|
| `compute_centroids.py` | Step 1 |
| `visualize_PCA_prototypes.py` | Step 2 |
| `reorganize_h5_by_confidence.py` | Step 3 |
| `utils.py` | H5 reading, tile-ID parsing and grid drawing used by the three scripts; also contains helper functions from earlier exploration that these scripts do not call |
| `all_tiles.txt`, `tumor_toi.txt`, `stroma_toi.txt` | Slide lists |
| `centroids/*_pca50.csv` | Starting centroids used for the manuscript |
| `requirements.txt` | Python packages, pinned to the versions used for the manuscript run (Python 3.10.12) |

## Notes

- Step 2 holds all tiles of one tissue type in memory (about 1.4 million tumor or 1.5 million stromal tiles × 1536 values), so it needs a large-memory machine.
- The step 1 centroids are expressed in the PCA computed in step 1 on the initialization slides. Step 2 computes a separate standardization and PCA on all tiles and uses those centroids as the GMM starting means.
- The clustering uses only the tile vectors. BMI is not used.
