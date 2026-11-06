# AUC boxplot – notebooks version

Reproduces the boxplot "All Patients Classification Performance" (manuscript Figure 2): cross-validated AUC for 7 ML models across 7 regions of interest (ROIs), for all patients (`large_group`, 20x).

The pipeline has three notebooks, run in order: **stratify** the patients into folds, **fit** the models, **plot** the AUCs. A fourth notebook (step 4) compares patient cohorts and regions for one model, using the aggregated AUC values in `aggregated_auc_results.xlsx` (manuscript Figure 3 and Table 3).

## Files in this folder

| File | Role |
|------|------|
| `step1_stratify.ipynb` | Step 1. Labels patients high/low BMI and creates the stratified 5-fold partitions ("datapacks"). |
| `step2_run_ml_models.ipynb` | Step 2. For every seed, fold and ROI, fits the 7 models on the training folds and tests on the held-out fold. |
| `step3_plot_auc_boxplot.ipynb` | Step 3. Draws boxplots for AUC results using all patients for different tissue regions and ML methods. |
| `step4_single_model_multiple_cohorts_Vfinal.ipynb` | Step 4. Draws a boxplot of AUC results for a single method (log) and multiple cohorts and regions. |
| `aggregated_auc_results.xlsx` | A spreadsheet that aggregates all AUC results from cross-validation experiments. |
| `common_mlops_lib.py` | Model fitting and metrics, imported by step 2 (it uses `build_ml_models3`; the other functions in the file are unused here). |
| `commonfilelib.py` | Reads a vector `.h5` file into a dataframe (`get_vectors_from_h5`), imported by step 2. The other functions are unused here. |
| `requirements.txt` | Pinned Python packages, including `ipykernel` (tested on Python 3.12). |

No data is stored in this folder. The data paths are set in the first code cell of each notebook and currently hold **sample paths** (`/path/to/BMI_DATA/...`, marked `SAMPLE PATH`). Replace them with your own before running.

## Data layout (sample)

```
/path/to/BMI_DATA/
├── large_group.csv        # input: one row per patient (block_name, BMI, ...)
├── VECTORS/               # input: slide vectors
│   └── Demographics_321/20x/CANVOI_v1_1/...
├── datapack/              # created by step 1
├── results/               # created by step 2
└── output/                # created by step 3 (plot + summary CSV)
```

The vector files used for each ROI (relative to `VECTORS/Demographics_321/20x/CANVOI_v1_1/`):

| ROI in code | Label in plot | Vector file |
|-------------|---------------|-------------|
| `WSI` | combined tumor/stroma | `collective_slide_vectors_non_ROI/full_vectors_compressed.h5` |
| `Tumor` | tumor | `collective_slide_vectors_90_ROI/Positive/full_vectors_compressed.h5` |
| `NonTumor` | stroma | `collective_slide_vectors_90_ROI/Negative/full_vectors_compressed.h5` |
| `TumorDense` | tumor HD | `collective_slide_vectors_90_ROI_cluster/TumorDense/full_vectors_compressed.h5` |
| `TumorNotDense` | tumor LD | `collective_slide_vectors_90_ROI_cluster/TumorNotDense/full_vectors_compressed.h5` |
| `StromaDense` | stroma HD | `collective_slide_vectors_90_ROI_cluster/StromaDense/full_vectors_compressed.h5` |
| `StromaNotDense` | stroma LD | `collective_slide_vectors_90_ROI_cluster/StromaNotDense/full_vectors_compressed.h5` |

Each file has an `ids` array and a 2000-dimension `vectors` array. HD = high density, LD = low density. The HD/LD tiles are defined by the tile-level clustering in [Tile_Clustering](../Tile_Clustering/); the slide vectors are then computed with the CanvOI slide encoder (proprietary) and are provided in the dataset.

## Run

Use Python 3.12.

```bash
cd AUC_Boxplot_Notebooks
pip install -r requirements.txt
jupyter lab        # or open the notebooks in VS Code
```

Open the notebooks with `AUC_Boxplot_Notebooks` as the working directory (step 2 imports the two `.py` helper files from it), edit the sample paths in the first code cell, and run all cells in this order.

### Step 1 – `step1_stratify.ipynb`

Settings: `DATA_DIR` (folder with `large_group.csv`, sample `/path/to/BMI_DATA`) and `OUTPUT_DIR` (sample `/path/to/BMI_DATA/datapack`). Also `GROUPS`, `SEEDS` (11 and 35317), `N_SPLITS` (5).

1. Reads `large_group.csv` (304 patients).
2. Labels each patient from BMI: **high** if BMI >= 30, **low** if BMI < 25. Patients in between (109) are dropped, leaving 195 (102 high, 93 low).
3. Splits the 195 patients into 5 stratified folds (`StratifiedKFold`, shuffled), once for each seed: `11` (saved as `s0011`) and `35317`.
4. Writes `large_group_bmi_s<seed>_part<k>_datapack.csv` (columns `index`, `block_name`, `status`) to `OUTPUT_DIR`.

### Step 2 – `step2_run_ml_models.ipynb`

Settings: `VECTORS_ROOT` (sample `/path/to/BMI_DATA/VECTORS`), `DATAPACK_DIR` (sample `/path/to/BMI_DATA/datapack`), `RESULTS_DIR` (sample `/path/to/BMI_DATA/results`). Also `GROUP`, `SEEDS`, `PARTITIONS`, `MAGNITUDE`, `ROIS` (set a subset such as `["TumorDense", "StromaDense"]` to run fewer ROIs).

For each seed (`0011`, `35317`), fold `k` (0-4) and ROI (7):

1. Test set = fold `k`; training set = the other 4 folds.
2. Patient IDs are `block_name + "_01"`; they are joined to the ROI vectors (patients without a vector in that ROI are dropped).
3. `BMI_High` = 1 for high, 0 for low. The 2000 vector values are the features.
4. Seven models are fitted with default scikit-learn settings (no grid search): `log` (logistic regression), `knn`, `mlp`, `svm`, `dtr` (decision tree), `xgb` (XGBoost), `rfo` (random forest). Model random seed is 15.
5. Writes `RESULTS_DIR/20x/large_group_s<seed>_<ROI>/tst_part<k>.csv` with AUC, accuracy, MCC, F1, specificity, sensitivity and the confusion matrix counts (`c00`, `c01`, `c10`, `c11`) for each model.

Takes about 1.5 minutes.

### Step 3 – `step3_plot_auc_boxplot.ipynb`

Settings: `RESULTS_DIR` (sample `/path/to/BMI_DATA/results`) and `OUTPUT_DIR` (sample `/path/to/BMI_DATA/output`). Also `GROUP`, `MAGNITUDE`, `ROIS`.

1. Loads all `tst_part*.csv` files for each ROI: 2 seeds x 5 folds = **10 AUC values per model and ROI**.
2. Draws one box per model/ROI with the 10 points on top, shows it in the notebook and saves `boxplot_auc_large_group_all_ROI_comparison.png` to `OUTPUT_DIR`.
3. Writes `summary_all_ROIs_large_group.csv` (mean AUC with 95% CI, and accuracy, sensitivity, specificity, F1 and MCC computed from the confusion counts pooled over the 10 folds).

### Step 4 - `step4_single_model_multiple_cohorts_Vfinal.ipynb`

Settings: This notebook needs a spreadsheet with aggregated AUC results for multiple cohorts, regions, and methods. We 
provide a sample spreadsheet (`aggregated_auc_results.xlsx`) as a template. The spreadsheet has multiple pages, one for 
each combination of cohort and region. Each page has a column describing the ML method, and 10 columns for 5x2 
cross-validation AUC values. In the example, the columns names use the format `s<seed>_part<fold>`, e.g., `s0011_part0` 
for seed 11, fold 0. 

1. Loads the spreadsheet and extracts the AUC values for each method, cohort, and region.
2. Generates boxplots for combinations of cohort and region using a logistic regression classifier. 
3. Computes a table with mean AUC, confidence intervals, median, min, and max values for each combination of 
cohort and region. It also shows p-values comparing the mean AUC values with a random classifier (AUC = 0.5) 
using a one-sample t-test.
4. Computes a table with p-values comparing p-values between different regions (stroma vs tumor) and stroma vs combined tumor/stroma for 
each of the target cohorts. 