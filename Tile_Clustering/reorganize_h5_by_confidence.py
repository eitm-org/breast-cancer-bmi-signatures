#!/usr/bin/env python3
"""
Reorganize tile vectors into per-slide H5 files based on cluster assignments
and confidence threshold.

- Reads cluster assignments CSV with columns: ID, kmeans_cluster (or kmean_cluster), confidence
- Filters rows with confidence >= threshold
- Maps cluster -> Type per tissue:
    Tumor:  cluster 0 -> TumorDense, cluster 1 -> TumorNotDense
    Stroma: cluster 1 -> StromaDense, cluster 0 -> StromaNotDense
- Reads original source H5 per slide+tissue
- Writes one H5 per slide per Type to destination structure:
    <dest>/<SLIDE>/<SLIDE>_01/<TYPE>/<SLIDE>_01.h5

Example:
    python3 reorganize_h5_by_confidence.py \
      --tissue tumor \
      --assignments tumor_cluster_assignments.csv \
      --confidence-threshold 0.7 \
      --source /path/to/tile_vectors_90_ROI \
      --dest   /path/to/tile_vectors_90_ROI_cluster
"""

import os
import argparse
import h5py
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Set, Tuple

# Local utilities
from utils import parse_tile_id, get_vectors_from_h5


def map_type(tissue: str, cluster_id: int) -> str:
    """Map (tissue, cluster) -> Type directory name."""
    if tissue == 'tumor':
        return 'TumorDense' if cluster_id == 0 else 'TumorNotDense'
    elif tissue == 'stroma':
        return 'StromaDense' if cluster_id == 1 else 'StromaNotDense'
    else:
        raise ValueError("tissue must be 'tumor' or 'stroma'")


def tissue_dir_name(tissue: str) -> str:
    return 'Positive' if tissue == 'tumor' else 'Negative'


def load_assignments(path: str, conf_thresh: float) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Assignments CSV not found: {path}")
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # Handle kmeans_cluster vs kmean_cluster
    if 'kmeans_cluster' in df.columns:
        cluster_col = 'kmeans_cluster'
    elif 'kmean_cluster' in df.columns:
        cluster_col = 'kmean_cluster'
    else:
        raise ValueError("CSV missing 'kmeans_cluster' or 'kmean_cluster' column")
    if 'confidence' not in df.columns:
        raise ValueError("CSV missing 'confidence' column")
    if 'ID' not in df.columns:
        # Try case-insensitive
        id_col = cols.get('id', None)
        if not id_col:
            raise ValueError("CSV missing 'ID' column")
        df.rename(columns={id_col: 'ID'}, inplace=True)
    # Filter by threshold
    df_filt = df[df['confidence'] >= conf_thresh].copy()
    df_filt.rename(columns={cluster_col: 'cluster'}, inplace=True)
    return df_filt[['ID', 'cluster', 'confidence']]


def group_ids_by_slide_and_type(assign_df: pd.DataFrame, tissue: str) -> Dict[Tuple[str, str], Set[str]]:
    """Return mapping: (slide, type_name) -> set(tile_ids)."""
    buckets: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for _, row in assign_df.iterrows():
        tile_id = row['ID']
        cluster_id = int(row['cluster'])
        slide, slide_base, _ = parse_tile_id(tile_id)
        if not slide:
            continue
        type_name = map_type(tissue, cluster_id)
        buckets[(slide, type_name)].add(tile_id)
    return buckets


def read_source_h5(source_root: str, tissue: str, slide: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read ids and vectors from source H5 for given slide+tissue."""
    slide_base = f"{slide}_01"
    tdir = tissue_dir_name(tissue)
    h5_path = f"{source_root}/{slide}/{slide_base}/{tdir}/{slide_base}.h5"
    # Use helper to get a DataFrame, then convert to arrays
    df = get_vectors_from_h5(h5_path)
    ids = df['ID'].values
    vectors = df.drop(columns=['ID']).values.astype(np.float32)
    return ids, vectors


def write_h5(dest_root: str, slide: str, type_name: str, ids: np.ndarray, vectors: np.ndarray) -> str:
    """Write ids+vectors to destination H5, return path."""
    slide_base = f"{slide}_01"
    out_dir = f"{dest_root}/{slide}/{slide_base}/{type_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{slide_base}.h5"
    with h5py.File(out_path, 'w') as f:
        id_bytes = [str(x).encode('utf-8') for x in ids]
        f.create_dataset('ids', data=id_bytes)
        f.create_dataset('vectors', data=vectors.astype(np.float32))
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Reorganize h5 files by cluster and confidence')
    parser.add_argument('--tissue', type=str, required=True, choices=['tumor', 'stroma'], help='Tissue type')
    parser.add_argument('--assignments', type=str, required=False, help='Assignments CSV path')
    parser.add_argument('--confidence-threshold', type=float, default=0.7, help='Minimum confidence to include')
    parser.add_argument('--source', type=str, default='/path/to/tile_vectors_90_ROI', help='Source root')
    parser.add_argument('--dest', type=str, default='/path/to/tile_vectors_90_ROI_cluster', help='Destination root')
    parser.add_argument('--dry-run', action='store_true', help='List actions without writing files')
    args = parser.parse_args()

    # Default assignments path if not provided
    if not args.assignments:
        args.assignments = f"{args.tissue}_cluster_assignments.csv"

    print("=== Reorganize H5 by Confidence ===")
    print(f"Tissue: {args.tissue}")
    print(f"Assignments: {args.assignments}")
    print(f"Threshold: {args.confidence_threshold}")
    print(f"Source: {args.source}")
    print(f"Dest:   {args.dest}")
    print(f"Mode:   {'DRY-RUN' if args.dry_run else 'WRITE'}")
    
    # Show cluster to folder mapping
    if args.tissue == 'tumor':
        print("\nCluster mapping:")
        print("  Cluster 0 → TumorDense")
        print("  Cluster 1 → TumorNotDense")
    else:  # stroma
        print("\nCluster mapping:")
        print("  Cluster 0 → StromaNotDense")
        print("  Cluster 1 → StromaDense")

    # Load assignments filtered by confidence
    assign_df = load_assignments(args.assignments, args.confidence_threshold)
    if len(assign_df) == 0:
        print("No rows above threshold; nothing to do.")
        return
    print(f"Loaded {len(assign_df)} assignments above threshold.")

    # Group tile IDs by (slide, type)
    buckets = group_ids_by_slide_and_type(assign_df, args.tissue)
    print(f"Found {len(buckets)} (slide,type) buckets.")

    total_written = 0
    processed_slides = defaultdict(set)  # Track which types were created per slide
    for (slide, type_name), id_set in buckets.items():
        print(f"-> Processing {slide} / {type_name} (n={len(id_set)})")
        try:
            src_ids, src_vecs = read_source_h5(args.source, args.tissue, slide)
        except Exception as e:
            print(f"   ✗ Failed to read source for {slide}: {e}")
            continue
        # Build mask for selected IDs
        id_set_str = set(id_set)
        mask = np.array([id_ in id_set_str for id_ in src_ids])
        sel_ids = src_ids[mask]
        sel_vecs = src_vecs[mask]
        if len(sel_ids) == 0:
            print("   ⚠ No matching IDs found in source; skipping.")
            continue
        # Compute the destination path (same logic as write_h5)
        slide_base = f"{slide}_01"
        out_path_dest = f"{args.dest}/{slide}/{slide_base}/{type_name}/{slide_base}.h5"
        if args.dry_run:
            print(f"   DRY-RUN: Would write {len(sel_ids)} tiles to {out_path_dest}")
        else:
            try:
                out_path = write_h5(args.dest, slide, type_name, sel_ids, sel_vecs)
                print(f"   ✓ Wrote {len(sel_ids)} tiles -> {out_path}")
                processed_slides[slide].add(type_name)
                total_written += len(sel_ids)
            except Exception as e:
                print(f"   ✗ Failed to write for {slide}/{type_name}: {e}")
    
    # Report slides with incomplete folder pairs
    print(f"\nDone. Total tiles written: {total_written}")
    
    incomplete_slides = []
    for slide in processed_slides:
        if len(processed_slides[slide]) < 2:
            incomplete_slides.append((slide, processed_slides[slide]))
    
    if incomplete_slides:
        print(f"\n⚠ {len(incomplete_slides)} slides with incomplete folder pairs:")
        for slide, types_created in sorted(incomplete_slides):
            missing = set(map_type(args.tissue, i) for i in [0, 1]) - types_created
            print(f"  - {slide}: only has {', '.join(sorted(types_created))} (missing {', '.join(sorted(missing))})")
    else:
        print(f"\n✓ All {len(processed_slides)} slides have both folder types created.")


if __name__ == '__main__':
    main()
