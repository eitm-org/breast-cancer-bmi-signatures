
"""Utility functions for generating and saving population distance distribution figures."""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind


names = {
        "sh_th": "stroma-tumor (high BMI)",
        "sh_ch": "stroma-combined (high BMI)",
        "th_ch": "tumor-combined (high BMI)",
        "sl_tl": "stroma-tumor (low BMI)",
        "sl_cl": "stroma-combined (low BMI)",
        "tl_cl": "tumor-combined (low BMI)",
        "sh_sl": "high-low BMI (stroma)",
        "th_tl": "high-low BMI (tumor)"
    }

def mann_whitney_u_test(array1: np.array, array2: np.array) -> tuple:
    """Perform Mann-Whitney U test between two arrays."""
    u_stat, p_value = mannwhitneyu(array1, array2, alternative='two-sided')
    return u_stat, p_value

def make_multipartite_figures(distributions: dict, output_dir: str) -> None:
    """Generate and save figures for distance distributions."""
    dist = distributions.copy()

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    sns.kdeplot(dist["sh_th"], label='stroma-tumor', color='red', fill=True)
    sns.kdeplot(dist["sh_ch"], label='stroma-combined', color='blue', fill=True)
    sns.kdeplot(dist["th_ch"], label='tumor-combined', color='green', fill=True)
    plt.title('Multipartite Distance Distributions for High BMI')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    sns.kdeplot(dist["sl_tl"], label='stroma-tumor', color='red', fill=True)
    sns.kdeplot(dist["sl_cl"], label='stroma-combined', color='blue', fill=True)
    sns.kdeplot(dist["tl_cl"], label='tumor-combined', color='green', fill=True)
    plt.title('Multipartite Distance Distributions for Low BMI')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    sns.kdeplot(dist["sh_sl"], label='high-low BMI (stroma)', color='orange', fill=True)
    sns.kdeplot(dist["th_tl"], label='high-low BMI (tumor)', color='purple', fill=True)
    plt.title('Multipartite Distance Distribution for Stroma and Tumor')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multipartite_distance_distributions.png'))
    plt.close()


def make_unipartite_figures(distributions: dict, output_dir: str) -> None:
    """Generate and save figures for patient distance distributions."""
    dist = distributions.copy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    sns.kdeplot(dist["sh_th"], label='stroma-tumor', color='red', fill=True)
    sns.kdeplot(dist["sh_ch"], label='stroma-combined', color='blue', fill=True)
    sns.kdeplot(dist["th_ch"], label='tumor-combined', color='green', fill=True)
    plt.title('Unipartite Distance Distributions for High BMI')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.kdeplot(dist["sl_tl"], label='stroma-tumor', color='red', fill=True)
    sns.kdeplot(dist["sl_cl"], label='stroma-combined', color='blue', fill=True)
    sns.kdeplot(dist["tl_cl"], label='tumor-combined', color='green', fill=True)
    plt.title('Unipartite Distance Distributions for Low BMI')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unipartite_distance_distributions.png'))
    plt.close()

def compute_multipartite_stats(distributions: dict, output_dir: str) -> None:
    """Compute and save statistics for multipartite distance distributions."""
    dist = distributions.copy()

    res = {"distribution": [], "mean": [], "median": [], "std": [], "max": [], "min": []}
    for key in dist.keys():
        res["distribution"].append(names[key])
        res["mean"].append(np.mean(dist[key]))
        res["median"].append(np.median(dist[key]))
        res["std"].append(np.std(dist[key]))
        res["max"].append(np.max(dist[key]))
        res["min"].append(np.min(dist[key]))

    res_df = pd.DataFrame(res)
    res_df.to_csv(os.path.join(output_dir, 'multipartite_distance_stats.csv'), index=False)

    pvalues = {"distribution1": [], "distribution2": [], "p_value": []}
    pvalues2 = {"distribution1": [], "distribution2": [], "p_value2": []}

    d_pairs = [("sh_th", "sh_ch"), ("sh_th", "th_ch"), ("sh_ch", "th_ch"),
               ("sl_tl", "sl_cl"), ("sl_tl", "tl_cl"), ("sl_cl", "tl_cl"),
               ("sh_sl", "th_tl")]

    for pair in d_pairs:
        d1, d2 = pair

        pvalues["distribution1"].append(names[d1])
        pvalues["distribution2"].append(names[d2])
        _, p_val = mann_whitney_u_test(dist[d1], dist[d2])
        pvalues["p_value"].append(p_val)

        pvalues2["distribution1"].append(names[d1])
        pvalues2["distribution2"].append(names[d2])
        _, p_val2 = ttest_ind(dist[d1], dist[d2], equal_var=False, nan_policy='omit')
        pvalues2["p_value2"].append(p_val2)

    pvalues_df = pd.DataFrame(pvalues)
    pvalues_df.to_csv(os.path.join(output_dir, 'multipartite_u_test_pvalues.csv'), index=False)
    pvalues2_df = pd.DataFrame(pvalues2)
    pvalues2_df.to_csv(os.path.join(output_dir, 'multipartite_t_test_pvalues.csv'), index=False)


def compute_unipartite_stats(distributions: dict, output_dir: str) -> None:
    """Compute and save statistics for unipartite distance distributions."""
    dist = distributions.copy()

    res = {"distribution": [], "mean": [], "median": [], "std": [], "max": [], "min": []}

    for key in dist.keys():
        res["distribution"].append(names[key])
        res["mean"].append(np.mean(dist[key]))
        res["median"].append(np.median(dist[key]))
        res["std"].append(np.std(dist[key]))
        res["max"].append(np.max(dist[key]))
        res["min"].append(np.min(dist[key]))

    res_df = pd.DataFrame(res)
    res_df.to_csv(os.path.join(output_dir, 'unipartite_distance_stats.csv'), index=False)

    pvalues = {"distribution1": [], "distribution2": [], "p_value": []}

    d1, d2 = "sh_th", "sh_ch"
    pvalues["distribution1"].append(names[d1])
    pvalues["distribution2"].append(names[d2])
    _, p_val = mann_whitney_u_test(dist[d1], dist[d2])
    pvalues["p_value"].append(p_val)

    d1, d2 = "sh_th", "th_ch"
    pvalues["distribution1"].append(names[d1])
    pvalues["distribution2"].append(names[d2])
    _, p_val = mann_whitney_u_test(dist[d1], dist[d2])
    pvalues["p_value"].append(p_val)

    d1, d2 = "sh_ch", "th_ch"
    pvalues["distribution1"].append(names[d1])
    pvalues["distribution2"].append(names[d2])
    _, p_val = mann_whitney_u_test(dist[d1], dist[d2])
    pvalues["p_value"].append(p_val)

    d1, d2 = "sl_tl", "sl_cl"
    pvalues["distribution1"].append(names[d1])
    pvalues["distribution2"].append(names[d2])
    _, p_val = mann_whitney_u_test(dist[d1], dist[d2])
    pvalues["p_value"].append(p_val)

    d1, d2 = "sl_tl", "tl_cl"
    pvalues["distribution1"].append(names[d1])
    pvalues["distribution2"].append(names[d2])
    _, p_val = mann_whitney_u_test(dist[d1], dist[d2])
    pvalues["p_value"].append(p_val)

    d1, d2 = "sl_cl", "tl_cl"
    pvalues["distribution1"].append(names[d1])
    pvalues["distribution2"].append(names[d2])
    _, p_val = mann_whitney_u_test(dist[d1], dist[d2])
    pvalues["p_value"].append(p_val)

    pvalues_df = pd.DataFrame(pvalues)
    pvalues_df.to_csv(os.path.join(output_dir, 'unipartite_distance_pvalues.csv'), index=False)


def compute_p_val_matrix(distributions: dict, output_dir: str) -> None:
    """Compute and save p-value matrix for all distance distributions."""
    dist = distributions.copy()
    keys = list(dist.keys())
    keys = [key for key in keys if 'c' not in key]
    print("Performing t-tests for:", keys)
    n = len(keys)

    pval_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i > j:
                _, p_val = ttest_ind(dist[keys[i]], dist[keys[j]], equal_var=False, nan_policy='omit')
                pval_matrix[i, j] = np.round(p_val, decimals=4)
            elif i == j:
                pval_matrix[i, j] = 1.0
            else:
                pval_matrix[i, j] = np.nan

    pval_df = pd.DataFrame(pval_matrix, index=[names[k] for k in keys], columns=[names[k] for k in keys])

    pval_df[pval_df < 0.001] = '<0.001'
    pval_df.to_csv(os.path.join(output_dir, 'distance_pvalue_matrix.csv'))


def make_aod_figures(aod_data: dict, output_dir: str) -> None:
    """Generate and save figures for AOD distributions."""
    if len(aod_data.keys()) != 8:
        raise ValueError("There should be 8 marginals in the marginals dictionary.")

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    plt.figure(figsize=(10, 10))
    for k, token in enumerate(aod_data):

        mean_samples = aod_data[token]
        plt.subplot(4, 2, k+1)
        sns.kdeplot(mean_samples, label=token, color=colors[k], fill=True)
        # sns.kdeplot(dist["sh_ch"], label='stroma-combined', color='blue', fill=True)
        # sns.kdeplot(dist["th_ch"], label='tumor-combined', color='green', fill=True)
        # plt.title('Distributions of mean distances for {}'.format(token))
        if k in [6, 7]:
            plt.xlabel('Average Cosine Distance')
        if k in [0, 2, 4, 6]:
            plt.ylabel('Density')
        plt.xlim(0.1, 0.65)
        plt.ylim(0, 10.5)
        plt.legend(fontsize=13, loc='upper right')
        plt.grid(True)

    plt.suptitle('AoD density plots for all relevant combinations', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aod_density_plots.png'))
    plt.close()

def make_aod_same_group_figures(aod_data: dict, output_dir: str) -> None:
    """Generate and save figures for AoD distributions within the same group (high and low BMI)."""
    if len(aod_data.keys()) != 8:
        raise ValueError("There should be 8 marginals in the marginals dictionary.")

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    plt.figure(figsize=(10, 5))
    high_group_codes = ['th_sh', 'sh_th']
    low_group_codes = ['tl_sl', 'sl_tl']

    plt.subplot(1, 3, 1)
    samples1 = aod_data[high_group_codes[0]]
    samples2 = aod_data[high_group_codes[1]]
    sns.kdeplot(samples1, label=high_group_codes[0], color=colors[0], fill=True)
    sns.kdeplot(samples2, label=high_group_codes[1], color=colors[2], fill=True)
    plt.xlabel('Average Cosine Distance')
    plt.ylabel('Density')
    plt.xlim(0.1, 0.65)
    plt.ylim(0, 10.5)
    plt.title('High BMI')
    plt.legend(fontsize=13, loc='upper right')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    samples3 = aod_data[low_group_codes[0]]
    samples4 = aod_data[low_group_codes[1]]
    sns.kdeplot(samples3, label=low_group_codes[0], color=colors[5], fill=True)
    sns.kdeplot(samples4, label=low_group_codes[1], color=colors[7], fill=True)
    plt.xlabel('Average Cosine Distance')
    plt.ylabel('Density')
    plt.xlim(0.1, 0.65)
    plt.ylim(0, 10.5)
    plt.title('Low BMI')
    plt.legend(fontsize=13, loc='upper right')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    sns.kdeplot(samples1, label=high_group_codes[0], color=colors[0], fill=True)
    sns.kdeplot(samples2, label=high_group_codes[1], color=colors[2], fill=True)
    sns.kdeplot(samples3, label=low_group_codes[0], color=colors[5], fill=True)
    sns.kdeplot(samples4, label=low_group_codes[1], color=colors[7], fill=True)
    plt.xlabel('Average Cosine Distance')
    plt.ylabel('Density')
    plt.xlim(0.1, 0.65)
    plt.ylim(0, 10.5)
    plt.title('High BMI and Low BMI')
    plt.legend(fontsize=13, loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aod_group_density_plots.png'))
    plt.close()

def make_aod_same_region_figures(aod_data: dict, output_dir: str) -> None:
    """Generate and save figures for AoD distributions within the same region (stroma and tumor)."""
    if len(aod_data.keys()) != 8:
        raise ValueError("There should be 8 marginals in the marginals dictionary.")

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    plt.figure(figsize=(10, 5))
    stroma_region_codes = ['sh_sl', 'sl_sh']
    tumor_region_codes = ['th_tl', 'tl_th']

    plt.subplot(1, 3, 1)
    samples1 = aod_data[stroma_region_codes[0]]
    samples2 = aod_data[stroma_region_codes[1]]
    sns.kdeplot(samples1, label=stroma_region_codes[0], color=colors[3], fill=True)
    sns.kdeplot(samples2, label=stroma_region_codes[1], color=colors[6], fill=True)
    plt.xlabel('Average Cosine Distance')
    plt.ylabel('Density')
    plt.xlim(0.1, 0.65)
    plt.ylim(0, 10.5)
    plt.title('Stroma')
    plt.legend(fontsize=13, loc='upper right')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    samples3 = aod_data[tumor_region_codes[0]]
    samples4 = aod_data[tumor_region_codes[1]]
    sns.kdeplot(samples3, label=tumor_region_codes[0], color=colors[1], fill=True)
    sns.kdeplot(samples4, label=tumor_region_codes[1], color=colors[4], fill=True)
    plt.xlabel('Average Cosine Distance')
    plt.ylabel('Density')
    plt.xlim(0.1, 0.65)
    plt.ylim(0, 10.5)
    plt.title('Tumor')
    plt.legend(fontsize=13, loc='upper right')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    sns.kdeplot(samples1, label=stroma_region_codes[0], color=colors[3], fill=True)
    sns.kdeplot(samples2, label=stroma_region_codes[1], color=colors[6], fill=True)
    sns.kdeplot(samples3, label=tumor_region_codes[0], color=colors[1], fill=True)
    sns.kdeplot(samples4, label=tumor_region_codes[1], color=colors[4], fill=True)
    plt.xlabel('Average Cosine Distance')
    plt.ylabel('Density')
    plt.xlim(0.1, 0.65)
    plt.ylim(0, 10.5)
    plt.title('Stroma and Tumor')
    plt.legend(fontsize=13, loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aod_region_density_plots.png'))
    plt.close()


def is_symmetric(matrix):
    """ Checks if a given square matrix is symmetric. Returns True if symmetric, False otherwise."""

    # A symmetric matrix must be square
    if matrix.shape[0] != matrix.shape[1]:
        return False

    # Check if the matrix is equal to its transpose
    return np.array_equal(matrix, matrix.T)

def is_same(matrix1, matrix2):
    """ Check if the two matrices are the same. Returns True if they are the same, False otherwise."""

    # Check if the shapes are the same
    if matrix1.shape != matrix2.shape:
        return False

    # Check if all elements are the same
    return np.array_equal(matrix1, matrix2)

def is_square(matrix):
    """ Checks if a given matrix is square. Returns True if square, False otherwise."""
    return matrix.shape[0] == matrix.shape[1]


def generate_plots(matrix_data: np.ndarray, title: str ="Heatmap and Density"):
    """ Displays a heatmap for a given NxN matrix, a density plot for average of distances,
    and a boxplot using the average of distances.
    Args:
        matrix_data (np.ndarray): The NxN matrix data.
        title (str): The title for the heatmap plot.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    # The range of cosine distance values is from 0 to 2. However, in this case all observations are less than 1.0
    sns.heatmap(matrix_data, cmap="coolwarm", linewidths=0.0, vmin=0.0, vmax=1.0)
    plt.title("Heterogeneity")

    plt.subplot(1, 3, 2)
    ave_distances = np.mean(matrix_data, axis=1)
    sns.kdeplot(x=ave_distances, shade=True, color='olive', fill=True)
    plt.xlabel('Average Cosine Distance')
    plt.ylabel('Density')

    plt.subplot(1, 3, 3)
    sns.boxplot(y=ave_distances, color='skyblue')
    sns.stripplot(y=ave_distances, jitter=True, color="grey")

    plt.suptitle(title)

    plt.tight_layout()
    plt.show()

def generate_plots2(matrix_data: np.ndarray, title_parts: list[str], savedir=None):
    ave_distances = np.mean(matrix_data, axis=1)
    aod_median = np.round(np.median(ave_distances), decimals=2)

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nrows=1, ncols=3, figure=fig, width_ratios=[3, 1, 1])

    # Create subplots using the GridSpec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Plotting on the subplots
    sns.heatmap(matrix_data, cmap="coolwarm", linewidths=0.0, vmin=0.0, vmax=1.0, ax=ax1)
    ax1.set_title('Heatmap of distances')

    a = sns.kdeplot(x=ave_distances, color='navy', fill=True, ax=ax2)
    a.axvline(x=float(aod_median), color='grey', linestyle='--', label='median')
    # ax2.text(x=0.1, y=7.0, s=f"md = {aod_median}", color='black', fontsize=11, ha='left', va='bottom')
    ax2.set_xlabel('Average of Distances (AoD)')
    ax2.set_title('AoD Density Function')
    ax2.grid(visible=True, which='major', axis='both', linestyle='--', color='grey', linewidth=0.5)

    sns.boxplot(y=ave_distances, color='thistle', ax=ax3)
    sns.stripplot(y=ave_distances, jitter=True, color="grey", alpha=0.5, ax=ax3)
    ax3.set_title('AoD Boxplot')

    if len(title_parts) != 0:
        figure_title = ", ".join(title_parts)
        filename = "fig_" + "_".join(title_parts) + ".png"
    else:
        figure_title = "heatmap and density"
        filename = "fig_heatmap_and_density.png"

    print(f"The figure filename is {filename}")

    plt.suptitle(figure_title)
    plt.tight_layout()
    if savedir:
        filepath = os.path.join(savedir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"A figure with a heatmap, density plot and box plot has been saved to {filepath}")
    else:
        plt.show()


def generate_eigenvalues_figure(sorted_eigenvalues: np.ndarray,
                                title_parts: list[str], savedir=None, normalized=False):
    box_properties = dict(
        boxstyle='round,pad=1',  # Style of the box (e.g., 'square', 'round')
        facecolor='white',  # Background color of the box
        alpha=0.7,  # Transparency of the box
        edgecolor='black',  # Color of the box border
        linewidth=0.5  # Width of the box border
    )
    token = "normalized" if normalized else "plain"

    fiedler_eigenval = np.round(sorted_eigenvalues[1], decimals=2)

    if len(title_parts) != 0:
        figure_title = ", ".join(title_parts) + f" ({token})"
        filename = f"spec_{token[0]}_" + "_".join(title_parts) + ".png"
    else:
        figure_title = "Sorted Eigenvalues" + f" ({token})"
        filename = f"eiv_sorted_eigenvalues_{token[0]}.png"

    y_for_txt = 0.1
    x_for_txt = 0.7*len(sorted_eigenvalues)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, color='royalblue', marker='.')
    plt.text(x=x_for_txt, y=y_for_txt,  s=f'Fiedler: {fiedler_eigenval}', fontsize=12,  bbox=box_properties)

    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    plt.title(figure_title + " (Norm. spectra)")
    plt.grid(True)

    if savedir:
        filepath = os.path.join(savedir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved figures showing {token} eigenvalues to {filepath}")
    else:
        plt.show()


def get_extreme_pairs(dist_matrix: np.ndarray, num_pairs: int, extreme: str):
    """Return the pairs (row, col) and values of the K largest or smallest elements in dist_matrix,
    sorted in decreasing (max) or increasing (min) order.
    """

    if extreme not in ['min', 'max']:
        raise ValueError("extreme must be 'min' or 'max'")

    flat = dist_matrix.flatten()
    n_rows, n_cols = dist_matrix.shape

    if extreme == 'min':
        nonzero_idx = np.where(flat != 0)[0]
        sorted_nonzero_idx = nonzero_idx[np.argsort(flat[nonzero_idx])]
    else:
        sorted_nonzero_idx = np.argsort(-flat)

    values, rows, cols = [], [], []
    count = 0
    for idx in sorted_nonzero_idx:
        r, c = np.unravel_index(idx, dist_matrix.shape)
        if r < c:  # Only keep one direction
            values.append(dist_matrix[r, c])
            rows.append(r)
            cols.append(c)
            count += 1
            if count == num_pairs:
                break

    return np.array(values), np.array(rows), np.array(cols)