"""Generate inter-tumor or inter-stroma distance plots. Also, compute similar distance plots
for subgroups of patients based on ages, ER status, and HER2 status."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import datahandler as dh
import constants as con
import utilities as ut

SELECTED_COHORT: str = "all_pat"
BMI_SUBSETS: list[str] = ['all', 'ob', 'ow', 'hw', 'uw', 'high', 'mid', 'low']
REGIONS: list[str] = ["tumor", "stroma", "combined", "TumorDense", "TumorNotDense", "StromaDense", "StromaNotDense"]
AGES: list[str] = ["all", "25-50", "50-75", "75-100", "50-100"]
ER_VALS: list[str] = ["all", "pos", "neg"]
HER2_VALS: list[str] = ["all", "pos", "neg"]
SORT_MODES = ["a", "d"]
NUM_PAIRS: int = 10

def parse_arguments():
    """Parse command line arguments."""
    desc = "Compute inter-tumor or inter-stroma distance for designated patient groups."
    parser = argparse.ArgumentParser(description=desc)
    required = parser.add_argument_group('Requirements')

    parser.add_argument("-out", "--output_dir", type=str,
                          help="Path to a directory that will store output plots and data. If not used, results "
                               "are displayed on the screen but no data is saved.")

    parser.add_argument("-reg", "--region", type=str, choices=REGIONS, default=REGIONS[0],
                        help=f"Choose a region from {REGIONS}. Default: {REGIONS[0]}")

    parser.add_argument("-bmi", "--bmi_subset", type=str, choices=BMI_SUBSETS, default=BMI_SUBSETS[0],
                        help=f"Choose a BMI group from {BMI_SUBSETS}. Default: {BMI_SUBSETS[0]}")

    parser.add_argument('-age', '--age_subset', type=str, choices=AGES, default=AGES[0],
                        help=f"Choose an age group from {AGES}. Default: {AGES[0]}")

    parser.add_argument('-er', '--er_subset', type=str, choices=ER_VALS, default=ER_VALS[0],
                        help=f"Choose an ER status from {ER_VALS}. Default: {ER_VALS[0]}")

    parser.add_argument('-her2', '--her2_subset', type=str, choices=HER2_VALS, default=HER2_VALS[0],
                        help=f"Choose a HER2 status from {HER2_VALS}. Default: {HER2_VALS[0]}")

    parser.add_argument('-sort', '--data_sort', type=str, choices=SORT_MODES, default=None,
                        help="Sort the data table per BMI in ascending (a) or descending (d)  order. If not "
                             "specified, the data is not sorted.")

    args = parser.parse_args()

    return args


def main():
    """Run the main sequence of procedures."""
    # Parse command line arguments
    args = parse_arguments()

    print("Parsed arguments and their values:")
    for arg_name, arg_value in sorted(vars(args).items()):
        print(f"  {arg_name}: {arg_value}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load clinical data
    clinical_df = pd.read_csv(con.patient_cohort_data[SELECTED_COHORT])

    # Add BMI category column (high, mid, low)
    clinical_df['BMI_category'] = clinical_df.apply(lambda row: dh.get_bmi_category(row["BMI"]), axis=1)
    # Add BMI group column (ob, ow, hw, uw)
    clinical_df['BMI_group'] = clinical_df.apply(lambda row: dh.get_bmi_group(row["BMI"]), axis=1)

    print("Clinical data shape after adding bmi columns:", clinical_df.shape)

    # Rename the block_name column to ID for compatibility with vector data
    clinical_df.rename(columns={'block_name': 'ID'}, inplace=True)

    # Show columns in the clinical data table
    print("Columns in clinical data table:", clinical_df.columns.tolist())

    if args.bmi_subset == "uw":
        print("+++ Selected rows of the clinical data table for BMI group 'uw':")
        uw_patients = clinical_df[clinical_df['BMI_group'] == 'uw']
        print(uw_patients[['ID', 'BMI']])

    vectors_file_path = con.vector_data[con.region_codes[args.region]]
    vectors_table = dh.get_vectors_dataframe(vectors_file_path)
    vectors_table["ID"] = vectors_table["ID"].apply(lambda x: x.split("_")[0])

    print("Shape of the vector table:", vectors_table.shape)

    # Merge clinical and vector data tables.
    merged_table = pd.merge(clinical_df, vectors_table, on='ID', how='inner')

    # Sport the merged clinical-vector data table if requested
    if args.data_sort:
        if args.data_sort == "a":
            merged_table.sort_values(by="BMI", ascending=True, inplace=True)
        elif args.data_sort == "d":
            merged_table.sort_values(by="BMI", ascending=False, inplace=True)
        else:
            raise ValueError("Invalid data sort mode")

    print("Shape of the merged (clinical and vector) table:", merged_table.shape)

    if args.bmi_subset != "all":
        if args.bmi_subset in ['ob', 'ow', 'hw', 'uw']:
            merged_table = merged_table[merged_table['BMI_group'] == args.bmi_subset]
        elif args.bmi_subset in ['high', 'mid', 'low']:
            merged_table = merged_table[merged_table['BMI_category'] == args.bmi_subset]
        else:
            print("Error: Invalid BMI subset!")
            sys.exit(1)
        print("Shape of the merged table after filtering the bmi subset:", merged_table.shape)

    if args.age_subset != "all":
        age1, age2 = args.age_subset.split("-")
        age1 = int(age1)
        age2 = int(age2)
        merged_table = merged_table[(merged_table['Age'] >= age1) & (merged_table['Age'] < age2)]
        print("Shape of the merged table after filtering the age group:", merged_table.shape)

    if args.er_subset != "all":
        if args.er_subset == "pos":
            merged_table = merged_table[merged_table['ER'] == 'Positive']
        else:
            merged_table = merged_table[merged_table['ER'] == 'Negative']
        print("Shape of the merged table after selecting ER status:", merged_table.shape)

    if args.her2_subset != "all":
        if args.her2_subset == "pos":
            merged_table = merged_table[merged_table['HER2'] == 'Positive']
        else:
            merged_table = merged_table[merged_table['HER2'] == 'Negative']
        print("Shape of the merged table after selecting HER2 status:", merged_table.shape)

    print("Final shape of the merged table after all filtering:", merged_table.shape)

    if args.bmi_subset == "uw":
        print("+++ The filtered merged table for BMI group 'uw':")
        print(merged_table[['ID', 'BMI']])

# Get only the numeric column names

    numeric_columns = [col for col in merged_table.columns if isinstance(col, int)
                       or (isinstance(col, str) and col.isdigit())]

    # Extract the dataframe with only numbered columns
    vector_data = merged_table[numeric_columns].to_numpy()

    print("Shape of the vector data array:", vector_data.shape)

    cd_matrix = cdist(vector_data, vector_data, metric='cosine')

    if args.bmi_subset == "uw":
        print("+++ Cosine distance matrix for BMI group 'uw':")
        print(cd_matrix)

    # Find patients with the minimum and maximum distances
    max_vals, max_rows, max_cols = ut.get_extreme_pairs(dist_matrix=cd_matrix, num_pairs=NUM_PAIRS, extreme='max')
    max_indices = zip(max_rows, max_cols)
    min_vals, min_rows, min_cols = ut.get_extreme_pairs(dist_matrix=cd_matrix, num_pairs=NUM_PAIRS, extreme='min')
    min_indices = zip(min_rows, min_cols)

    print("\nMax values extracted from the distance matrix:", max_vals)
    for index_pair in max_indices:
        pat1_id = merged_table.iloc[index_pair[0]]['ID']
        pat2_id = merged_table.iloc[index_pair[1]]['ID']
        dist_val = cd_matrix[index_pair[0], index_pair[1]]
        print(f"Max distance pair: {pat1_id} and {pat2_id} with distance {dist_val:.4f}")

    print("\nMin values extracted from the distance matrix:", min_vals)
    for index_pair in min_indices:
        pat1_id = merged_table.iloc[index_pair[0]]['ID']
        pat2_id = merged_table.iloc[index_pair[1]]['ID']
        dist_val = cd_matrix[index_pair[0], index_pair[1]]
        print(f"Min distance pair: {pat1_id} and {pat2_id} with distance {dist_val:.4f}")
    print()


    # Compute the Average-of-Distance metric
    ave_distances = np.mean(cd_matrix, axis=1)
    title_parts = generate_title_parts(args)

    if args.output_dir:
        data_file_name = "aod_" + "_".join(title_parts) + ".csv"
        data_file_path = os.path.join(args.output_dir, data_file_name)
        column_name = "aod"
        np.savetxt(fname=data_file_path, X=ave_distances, delimiter=",", header=column_name, comments="")
        print("Saved the Average-of-Distance values to ", data_file_path)

    ut.generate_plots2(cd_matrix, title_parts=title_parts, savedir=args.output_dir)


def generate_title_parts(args) -> list[str]:
    """Generate a list of strings that can be used to name a file or a directory."""
    title_parts = []
    if args.region == "tumor":
        title_parts.append("tumor")
    elif args.region == "stroma":
        title_parts.append("stroma")
    elif args.region == "StromaDense":
        title_parts.append("StromaDense")
    elif args.region == "StromaNotDense":
        title_parts.append("StromaNotDense")
    elif args.region == "TumorDense":
        title_parts.append("TumorDense")
    elif args.region == "TumorNotDense":
        title_parts.append("TumorNotDense")
    else:
        title_parts.append("combined")

    if args.bmi_subset != "all":
        title_parts.append("bmi-" + args.bmi_subset)

    if args.age_subset != "all":
        title_parts.append("age-" + args.age_subset)

    if args.er_subset != "all":
        title_parts.append("erpos" if args.er_subset == "pos" else "erneg")

    if args.her2_subset != "all":
        title_parts.append("her2pos" if args.her2_subset == "pos" else "her2neg")

    return title_parts



if __name__ == "__main__":
    main()
