"""Collection of functions and classes to handle data processing tasks."""

import pandas as pd
import h5py


def get_vectors_dataframe(h5_filepath: str) -> pd.DataFrame:
    """Read an h5 file, extract vectors and IDs, and create a pandas dataframe.
    Args:
        h5_filepath (str): Path to the h5 file containing 'ids' and 'vectors'
    Returns:
        A pandas dataframe where the first column is 'ID' and the remaining columns are vector elements.
        The column names for vector elements are string representations of their
        indices (e.g., '0', '1', ..., 'N-1').
    """

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


def get_bmi_category(bmi_value: float) -> str:
    """Return the BMI category based on BMI value.
    Args:
        bmi_value (float): The BMI value.
    Returns:
        A string representing the BMI category: 'high' (BMI >= 30), 'mid' (25 <= BMI < 30), or 'low' (BMI < 25).
    """
    if bmi_value >= 30:
        return "high"
    elif bmi_value < 25:
        return "low"
    else:
        return "mid"

def get_bmi_group(bmi_value: float) -> str:
    """Return the BMI group based on BMI value."""
    match bmi_value:
        case _ if bmi_value < 18.5:
            return "uw"
        case _ if 18.5 <= bmi_value < 25:
            return "hw"
        case _ if 25 <= bmi_value < 30:
            return "ow"
        case _ if bmi_value >= 30:
            return "ob"
        case _:
            raise ValueError("Invalid BMI value.")


