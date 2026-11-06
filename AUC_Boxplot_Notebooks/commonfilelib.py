#!/usr/bin/env python3
"""Library of file access functions."""

import h5py
import pandas as pd

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


def get_vectors_from_csv(csv_filepath: str) -> pd.DataFrame:
    """Read a csv file (vectorpack format), extract the vectors and IDs, and create a pandas dataframe."""

    dframe = pd.read_csv(csv_filepath)
    return dframe


def get_dimension_from_dframe(dframe: pd.DataFrame, df_includes_id=True) -> int:
    """Get the dimensions of vector embeddings from a dataframe that packs such vectors.
    By default, the dataframe includes one column called 'ID' and N columns representing a N-sized vector.
    The column names for vector elements are numbers from 0 to N-1 in string format.
    This function returns the value of N.
    If the df_includes_id argument is set to False, the function assumes that there is no ID column.
    """

    _, num_cols = dframe.shape
    n_value = num_cols - 1 if df_includes_id else num_cols
    return n_value
