# breast-cancer-bmi-signatures

Code for the manuscript *Deep Learning Identifies BMI-Associated Morphologic Features in Breast Cancer Tissue*: analysis of histopathology slide vectors of breast cancer patients in relation to BMI (high vs. low). The repo has three folders: `Tile_Clustering` defines the high- and low-cellular-density subregions, and the two analysis folders read the slide vectors and BMI values. The data are available on Zenodo ([10.5281/zenodo.23992256](https://doi.org/10.5281/zenodo.23992256)) and are **not stored here**; every data path in the code is a placeholder (`/path/to/...`) that you replace with your own location.

| Folder | What it does | Main output |
|--------|--------------|-------------|
| [Tile_Clustering](Tile_Clustering/) | Splits tumor and stromal tiles into high- and low-cellular-density groups with PCA + a Gaussian mixture model (GMM). | Per-tile cluster assignments, and tile vectors split into `TumorDense`, `TumorNotDense`, `StromaDense`, `StromaNotDense` |
| [Distance_Metrics_Fig_4](Distance_Metrics_Fig_4/) | Computes pairwise cosine distances between patients per region and BMI group, and plots the density of the average distance (heterogeneity, Fig. 4). | `aod_*.csv` and `fig_*.png` files, then density plots in a notebook |
| [AUC_Boxplot_Notebooks](AUC_Boxplot_Notebooks/) | Predicts high vs. low BMI from the vectors with 7 ML models and 5×2 cross-validation (5 folds, 2 random seeds), and plots the AUC per model and region (Fig. 2). Step 4 compares patient cohorts (Fig. 3, Table 3). | Boxplot PNGs and summary tables |

## How the Dense / NotDense vectors are made

1. Tiles are clustered at the tile level (`Tile_Clustering`), which splits the tumor and stromal tiles of each slide into dense and not-dense groups.
2. The tiles in each group (`TumorDense`, `TumorNotDense`, `StromaDense`, `StromaNotDense`) are aggregated with the CanvOI slide encoder into one slide-level vector per slide for each subgroup. The slide encoder is proprietary and is not included.
3. Those slide-level vectors are provided in the dataset as the `.h5` files that the two analysis folders read. They are the inputs used to plot the distance metrics (`Distance_Metrics_Fig_4`) and to train the ML models (`AUC_Boxplot_Notebooks`).

## Folders

### Tile_Clustering
`compute_centroids.py`, `visualize_PCA_prototypes.py` and `reorganize_h5_by_confidence.py` run in that order on the tile vectors. See [Tile_Clustering/README.md](Tile_Clustering/README.md).

### Distance_Metrics_Fig_4
`inter_heter_metrics.py` is run once per region and BMI group, then `plot_densities_ntbook.ipynb` draws the 3x2 low-vs-high BMI density figure. Set the paths in `constants.py` first. See [Distance_Metrics_Fig_4/README.md](Distance_Metrics_Fig_4/README.md).

### AUC_Boxplot_Notebooks
Three notebooks run in order: `step1_stratify` (folds), `step2_run_ml_models` (fit models), `step3_plot_auc_boxplot` (plot). A fourth notebook, `step4_single_model_multiple_cohorts_Vfinal`, reads aggregated AUC values from `aggregated_auc_results.xlsx` and compares cohorts and regions (Fig. 3, Table 3). Set the paths in the first cell of each notebook. See [AUC_Boxplot_Notebooks/README.md](AUC_Boxplot_Notebooks/README.md).

## Setup

Each folder has its own `requirements.txt` with pinned versions; install them separately. Use Python 3.12 for `AUC_Boxplot_Notebooks` and `Distance_Metrics_Fig_4` (their pinned numpy/scipy versions have no wheels for 3.13 or later). `Tile_Clustering` is pinned to the versions used for the manuscript run, which used Python 3.10.

```bash
cd <folder>
pip install -r requirements.txt
```

## Inputs shared by the folders

- Slide vectors: `.h5` files with an `ids` array (e.g. `EITB-0006G_01`) and a `vectors` array, one file per region (WSI, Tumor, Stroma, and the Dense / NotDense sub-regions).
- Tile vectors (`Tile_Clustering` only): one `.h5` file per slide and region, with 1536 values per tile.
- Clinical table: `large_group.csv` with `block_name` and `BMI` for the 304 patients. `inter_heter_metrics.py` can also filter by `Age`, `ER` and `HER2`, but these columns are not part of the dataset.
- BMI labels: high is BMI >= 30, low is BMI < 25; patients in between are dropped for the classification and are not used in the Fig. 4 plot.

## Citation

If you use this code, please cite the manuscript *Deep Learning Identifies BMI-Associated Morphologic Features in Breast Cancer Tissue* and the dataset (Zenodo, [10.5281/zenodo.23992256](https://doi.org/10.5281/zenodo.23992256)). Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub shows it under "Cite this repository".

## Status

This repository is the archived code for the manuscript and is not under active development. The version cited in the manuscript is tagged as a release.

## License

The code in this repository is released under the [MIT License](LICENSE). Copyright (c) 2026 Ellison Institute, LLC.
