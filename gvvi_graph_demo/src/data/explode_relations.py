"""
Phase 1a: Explode relations CSV files into positive edges parquet.

The SVI.csv, WVI.csv, DVI.csv files store relations in a compact format:
- Each row = one greenery grid cell
- img_id = comma-separated list of view IDs that see this grid
- pixel_wvi = comma-separated list of corresponding pixel counts

This script explodes them into a clean edge table.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def explode_relations(base_dir: str, view_type: str, chunk_size: int = 5000):
    """
    Explode one relations CSV into edge table.

    Args:
        base_dir: Path to GVVI_GNN_Demo实施包
        view_type: 'SVI', 'WVI', or 'DVI'
        chunk_size: Rows per chunk for memory management

    Returns:
        DataFrame with columns: view_type, view_id, greenery_id, visible, pixel_count
    """
    csv_path = os.path.join(base_dir, f"relations/{view_type}.csv")
    print(f"\n{'='*60}")
    print(f"Exploding {view_type}.csv")
    print(f"File: {csv_path}")

    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"Size: {file_size_mb:.1f} MB")

    all_edges = []
    total_rows = 0
    total_edges = 0
    total_positive_edges = 0
    length_mismatch_errors = 0
    total_pixel_mismatch_count = 0
    total_pixel_ok_count = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        total_rows += len(chunk)

        for _, row in chunk.iterrows():
            greenery_id = f"{int(row['grid_idx_x'])}_{int(row['grid_idx_y'])}"
            total_pixel = row['total_pixel']

            img_id_str = row['img_id']
            pixel_str = row['pixel_wvi']

            # Check for NaN/empty
            if pd.isna(img_id_str) or img_id_str == '' or img_id_str is None:
                # This greenery grid has no visible views of this type
                continue

            # Split and remove trailing empty entries
            img_ids = [x.strip() for x in img_id_str.split(',') if x.strip()]
            pixel_counts = [x.strip() for x in pixel_str.split(',') if x.strip()]

            # Verify lengths match
            if len(img_ids) != len(pixel_counts):
                length_mismatch_errors += 1
                print(f"ERROR: Length mismatch for greenery {greenery_id}: "
                      f"{len(img_ids)} img_ids vs {len(pixel_counts)} pixel_counts")
                continue

            if len(img_ids) == 0:
                continue

            # Convert types
            img_ids_int = [int(x) for x in img_ids]
            pixel_counts_int = [int(x) for x in pixel_counts]

            # Verify total_pixel matches sum of pixel_counts
            sum_pixels = sum(pixel_counts_int)
            if sum_pixels != total_pixel:
                total_pixel_mismatch_count += 1
            else:
                total_pixel_ok_count += 1

            # Create edges
            for vid, pc in zip(img_ids_int, pixel_counts_int):
                all_edges.append({
                    'view_type': view_type,
                    'view_id': vid,
                    'greenery_id': greenery_id,
                    'visible': 1,
                    'pixel_count': pc
                })
                total_positive_edges += 1

        total_edges = len(all_edges)
        print(f"  Processed {total_rows} rows, {total_positive_edges} positive edges so far...")

    edges_df = pd.DataFrame(all_edges)

    print(f"\n{view_type} Explosion Summary:")
    print(f"  Total grid rows processed: {total_rows}")
    print(f"  Total positive edges: {total_positive_edges}")
    print(f"  Length mismatches: {length_mismatch_errors}")
    print(f"  total_pixel matches: {total_pixel_ok_count}, mismatches: {total_pixel_mismatch_count}")

    if length_mismatch_errors > 0:
        raise RuntimeError(f"Found {length_mismatch_errors} length mismatches in {view_type}.csv!")

    return edges_df


def verify_viewpoint_connections(edges_df: pd.DataFrame, viewpoint_df: pd.DataFrame,
                                  view_type: str, id_col: str):
    """
    Verify that all view_ids in edges exist in the viewpoint table.
    """
    edge_view_ids = set(edges_df['view_id'].unique())
    viewpoint_ids = set(viewpoint_df[id_col].unique())

    missing = edge_view_ids - viewpoint_ids
    extra = viewpoint_ids - edge_view_ids

    print(f"\n{view_type} Viewpoint Connection Check:")
    print(f"  Unique view_ids in edges: {len(edge_view_ids)}")
    print(f"  Unique IDs in viewpoint table: {len(viewpoint_ids)}")
    print(f"  Missing from viewpoint table: {len(missing)}")
    print(f"  Viewpoints with no positive edges: {len(extra)}")

    if len(missing) > 0:
        raise RuntimeError(f"Found {len(missing)} view_ids in {view_type} edges "
                          f"not present in viewpoint table! First 10: {list(missing)[:10]}")

    print(f"  ✓ All {len(edge_view_ids)} view_ids successfully connect to viewpoint table")
    return extra  # view_ids with no positive edges (zero-views)


def main():
    base_dir = os.environ.get('GVVI_BASE_DIR',
        "/Users/wangfugui/线上交流_forLI/05_3D_CIM_概念示例/GVVI_GNN_Demo实施包")

    output_dir = os.path.join(base_dir, "gvvi_graph_demo/processed")
    os.makedirs(output_dir, exist_ok=True)

    all_edges = []
    zero_view_ids = {}  # view_type -> set of view_ids with no positive edges

    # Process each view type
    for view_type in ['SVI', 'WVI', 'DVI']:
        edges = explode_relations(base_dir, view_type)

        # Load viewpoint data and verify connections
        if view_type == 'SVI':
            vp = pd.read_csv(os.path.join(base_dir, "viewpoints/SVI_position_road.csv"))
            id_col = 'id'
        elif view_type == 'WVI':
            vp = pd.read_csv(os.path.join(base_dir, "viewpoints/window_view_location.csv"))
            id_col = 'ID'
        else:  # DVI
            vp = pd.read_csv(os.path.join(base_dir, "viewpoints/DVI_position_60.csv"))
            id_col = 'id'

        extra = verify_viewpoint_connections(edges, vp, view_type, id_col)
        zero_view_ids[view_type] = extra

        all_edges.append(edges)

    # Combine all edges
    combined = pd.concat(all_edges, ignore_index=True)

    # Save as parquet
    output_path = os.path.join(output_dir, "positive_edges.parquet")
    combined.to_parquet(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved {len(combined)} positive edges to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")

    # Summary statistics
    print(f"\nFinal Summary:")
    for vt in ['SVI', 'WVI', 'DVI']:
        vt_edges = combined[combined['view_type'] == vt]
        print(f"  {vt}: {len(vt_edges)} edges, "
              f"{vt_edges['view_id'].nunique()} unique views, "
              f"{vt_edges['greenery_id'].nunique()} unique greenery grids")

    # Save zero-view IDs (viewpoints with no positive edges)
    import json
    zero_views = {vt: list(ids) for vt, ids in zero_view_ids.items()}
    with open(os.path.join(output_dir, "zero_view_ids.json"), 'w') as f:
        json.dump({vt: len(ids) for vt, ids in zero_views.items()}, f)

    total_zero = sum(len(v) for v in zero_view_ids.values())
    print(f"\nTotal view IDs with zero positive edges: {total_zero}")
    for vt in ['SVI', 'WVI', 'DVI']:
        print(f"  {vt}: {len(zero_view_ids[vt])}")

    return combined


if __name__ == '__main__':
    main()
