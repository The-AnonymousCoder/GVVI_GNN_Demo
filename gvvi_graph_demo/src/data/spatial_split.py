"""
Phase 1c: Create spatial splits for train/val/test.

Splits by greenery grid spatial blocks to prevent data leakage.
Observations are assigned to blocks based on their spatial location.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from pathlib import Path
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_greenery_centers(base_dir):
    """Load greenery grid centers."""
    greenery = pd.read_csv(os.path.join(base_dir, "greenery/df_grid_ply.csv"))
    greenery['greenery_id'] = greenery.apply(
        lambda r: f"{int(r['grid_idx_x'])}_{int(r['grid_idx_y'])}", axis=1)
    greenery['center_x'] = (greenery['grid_coords_min_x'] + greenery['grid_coords_max_x']) / 2
    greenery['center_y'] = (greenery['grid_coords_min_y'] + greenery['grid_coords_max_y']) / 2
    return greenery


def spatial_cluster_split(greenery, n_blocks=5):
    """
    Cluster greenery grids by spatial location into n_blocks.

    Uses KMeans on spatial coordinates to create contiguous blocks.
    """
    coords = greenery[['center_x', 'center_y']].values

    # Normalize coordinates for better clustering
    coords_mean = coords.mean(axis=0)
    coords_std = coords.std(axis=0)
    coords_norm = (coords - coords_mean) / coords_std

    kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
    greenery = greenery.copy()
    greenery['block_id'] = kmeans.fit_predict(coords_norm)

    print(f"\nSpatial Clustering:")
    for bid in range(n_blocks):
        block = greenery[greenery['block_id'] == bid]
        print(f"  Block {bid}: {len(block)} grids, "
              f"x=[{block['center_x'].min():.0f}, {block['center_x'].max():.0f}], "
              f"y=[{block['center_y'].min():.0f}, {block['center_y'].max():.0f}]")

    return greenery


def assign_views_to_blocks(viewpoints, greenery):
    """
    Assign each viewpoint to a spatial block based on nearest greenery grid.
    """
    from scipy.spatial import KDTree

    greenery_coords = greenery[['center_x', 'center_y']].values
    tree = KDTree(greenery_coords)

    viewpoints = viewpoints.copy()
    view_coords = viewpoints[['x', 'y']].values

    # Find nearest greenery for each viewpoint
    distances, indices = tree.query(view_coords, k=1)
    viewpoints['block_id'] = greenery.iloc[indices]['block_id'].values
    viewpoints['nearest_greenery_dist'] = distances

    return viewpoints


def load_viewpoints(base_dir):
    """Load all viewpoint data with unified columns."""
    viewpoints = []

    svi = pd.read_csv(os.path.join(base_dir, "viewpoints/SVI_position_road.csv"))
    svi['view_type'] = 'SVI'
    svi = svi.rename(columns={'id': 'view_id', 'X': 'x', 'Y': 'y', 'Z': 'z'})
    viewpoints.append(svi[['view_type', 'view_id', 'x', 'y', 'z']])

    wvi = pd.read_csv(os.path.join(base_dir, "viewpoints/window_view_location.csv"))
    wvi['view_type'] = 'WVI'
    wvi = wvi.rename(columns={'ID': 'view_id', 'X': 'x', 'Y': 'y', 'Z': 'z'})
    viewpoints.append(wvi[['view_type', 'view_id', 'x', 'y', 'z']])

    dvi = pd.read_csv(os.path.join(base_dir, "viewpoints/DVI_position_60.csv"))
    dvi['view_type'] = 'DVI'
    dvi = dvi.rename(columns={'id': 'view_id'})
    viewpoints.append(dvi[['view_type', 'view_id', 'x', 'y', 'z']])

    all_vp = pd.concat(viewpoints, ignore_index=True)
    return all_vp


def assign_splits(greenery, n_blocks=5):
    """
    Assign train/val/test splits.

    Strategy: Use 3 blocks for train, 1 for val, 1 for test.
    Blocks are ordered by size and assigned to maximize spatial separation.
    """
    blocks = sorted(greenery['block_id'].unique())

    # Assign blocks to splits ensuring coverage of different areas
    # We use a deterministic assignment based on block sizes
    block_sizes = {b: len(greenery[greenery['block_id'] == b]) for b in blocks}
    blocks_by_size = sorted(blocks, key=lambda b: block_sizes[b], reverse=True)

    # Train gets largest 3 blocks (more training data)
    # Val gets 4th
    # Test gets 5th (smallest)
    split_assignment = {}
    for i, b in enumerate(blocks_by_size):
        if i < 3:
            split_assignment[b] = 'train'
        elif i == 3:
            split_assignment[b] = 'val'
        else:
            split_assignment[b] = 'test'

    # If fewer than 5 blocks, adjust
    if n_blocks < 5:
        split_assignment = {}
        if n_blocks == 3:
            split_assignment[blocks_by_size[0]] = 'train'
            split_assignment[blocks_by_size[1]] = 'val'
            split_assignment[blocks_by_size[2]] = 'test'

    greenery['split'] = greenery['block_id'].map(split_assignment)

    print(f"\nSplit Assignment:")
    for split_name in ['train', 'val', 'test']:
        split_data = greenery[greenery['split'] == split_name]
        blocks_in_split = split_data['block_id'].unique()
        print(f"  {split_name}: {len(split_data)} grids, "
              f"blocks {list(blocks_in_split)}")

    return greenery, split_assignment


def save_spatial_split_geojson(greenery, output_path):
    """Save spatial block boundaries as GeoJSON for visualization."""
    features = []
    for block_id in sorted(greenery['block_id'].unique()):
        block = greenery[greenery['block_id'] == block_id]
        split = block['split'].iloc[0]

        # Create polygon from min/max bounds
        min_x = block['center_x'].min() - 5
        max_x = block['center_x'].max() + 5
        min_y = block['center_y'].min() - 5
        max_y = block['center_y'].max() + 5

        coords = [[
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y]
        ]]

        features.append({
            'type': 'Feature',
            'properties': {
                'block_id': int(block_id),
                'split': split,
                'n_greenery': len(block),
                'x_range': f"{min_x:.0f}-{max_x:.0f}",
                'y_range': f"{min_y:.0f}-{max_y:.0f}"
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': coords
            }
        })

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    print(f"\nSaved spatial splits to {output_path}")


def main():
    base_dir = os.environ.get('GVVI_BASE_DIR',
        "/Users/wangfugui/线上交流_forLI/05_3D_CIM_概念示例/GVVI_GNN_Demo实施包")

    output_dir = os.path.join(base_dir, "gvvi_graph_demo/processed")
    os.makedirs(output_dir, exist_ok=True)

    # Load greenery data
    greenery = load_greenery_centers(base_dir)

    # Cluster into spatial blocks
    greenery = spatial_cluster_split(greenery, n_blocks=5)

    # Assign splits
    greenery, split_assignment = assign_splits(greenery)

    # Load and assign viewpoints
    viewpoints = load_viewpoints(base_dir)
    viewpoints = assign_views_to_blocks(viewpoints, greenery)

    # Save results
    greenery[['greenery_id', 'grid_idx_x', 'grid_idx_y', 'center_x', 'center_y',
              'block_id', 'split']].to_parquet(
        os.path.join(output_dir, "greenery_splits.parquet"), index=False)

    viewpoints[['view_type', 'view_id', 'x', 'y', 'z', 'block_id',
                 'nearest_greenery_dist']].to_parquet(
        os.path.join(output_dir, "viewpoint_splits.parquet"), index=False)

    # Save GeoJSON
    save_spatial_split_geojson(greenery,
        os.path.join(output_dir, "spatial_split.geojson"))

    # Save split assignment
    import json
    with open(os.path.join(output_dir, "split_assignment.json"), 'w') as f:
        json.dump({str(k): v for k, v in split_assignment.items()}, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Spatial Split Summary:")
    for split_name in ['train', 'val', 'test']:
        vp_split = viewpoints[viewpoints['block_id'].isin(
            greenery[greenery['split'] == split_name]['block_id'].unique())]
        print(f"  {split_name}: {len(vp_split)} viewpoints across "
              f"{vp_split['view_type'].nunique()} types")

    return greenery, viewpoints, split_assignment


if __name__ == '__main__':
    main()
