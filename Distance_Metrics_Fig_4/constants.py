# Placeholder: replace with the CANVOI_v1_1 folder created by unzipping slide_vectors.zip from the dataset.
VECTORS_ROOT = "/path/to/VECTORS/Demographics_321/20x/CANVOI_v1_1"

# One vector .h5 file per region (each has 'ids' and 'vectors').
vector_data = {
    "s": f"{VECTORS_ROOT}/collective_slide_vectors_90_ROI/Negative/full_vectors_compressed.h5",
    "t": f"{VECTORS_ROOT}/collective_slide_vectors_90_ROI/Positive/full_vectors_compressed.h5",
    "c": f"{VECTORS_ROOT}/collective_slide_vectors_non_ROI/full_vectors_compressed.h5",
    "sd": f"{VECTORS_ROOT}/collective_slide_vectors_90_ROI_cluster/StromaDense/full_vectors_compressed.h5",
    "sn": f"{VECTORS_ROOT}/collective_slide_vectors_90_ROI_cluster/StromaNotDense/full_vectors_compressed.h5",
    "td": f"{VECTORS_ROOT}/collective_slide_vectors_90_ROI_cluster/TumorDense/full_vectors_compressed.h5",
    "tn": f"{VECTORS_ROOT}/collective_slide_vectors_90_ROI_cluster/TumorNotDense/full_vectors_compressed.h5"
}

region_names = {
    "s": "stroma",
    "t": "tumor",
    "c": "combined",
    "sd": "StromaDense",
    "sn": "StromaNotDense",
    "td": "TumorDense",
    "tn": "TumorNotDense"
}

region_codes = {
    "stroma": 's',
    "tumor": 't',
    'combined': 'c',
    "StromaDense": 'sd', 
    "StromaNotDense":'sn',
    "TumorDense": 'td',
    "TumorNotDense": 'tn'

}

group_names = {
    'h': 'high_bmi',
    'l': 'low_bmi'
}

group_codes = {
    'high_bmi': 'h',
    'low_bmi': 'l'
}

patient_cohort_data = {
    # Placeholder: CSV with columns block_name and BMI (Age, ER, HER2 are needed only for the optional filters)
    "all_pat": "/path/to/large_group.csv"
}

