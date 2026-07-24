"""
Feature engineering for the GVVI Graph Demo.

Builds node features (observation, greenery) and edge features
for the heterogeneous graph, strictly using only cheap features
that can be obtained without exact rendering.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def encode_heading_sin_cos(heading_deg):
    """Encode heading as sin/cos to avoid 0°/360° discontinuity."""
    rad = np.radians(heading_deg)
    return np.sin(rad), np.cos(rad)


def build_observation_features(base_dir):
    """
    Build observation node features.

    Features:
    - view_type one-hot (SVI, WVI, DVI)
    - x, y, z (normalized per block later)
    - heading_sin, heading_cos
    - Approximate height above lowest Z
    """
    processed_dir = os.path.join(base_dir, "gvvi_graph_demo/processed")

    # Load viewpoint data
    viewpoints = []

    svi = pd.read_csv(os.path.join(base_dir, "viewpoints/SVI_position_road.csv"))
    svi['view_type'] = 'SVI'
    svi = svi.rename(columns={'id': 'view_id', 'X': 'x', 'Y': 'y', 'Z': 'z',
                               'Heading': 'heading'})
    viewpoints.append(svi[['view_type', 'view_id', 'x', 'y', 'z', 'heading']])

    wvi = pd.read_csv(os.path.join(base_dir, "viewpoints/window_view_location.csv"))
    wvi['view_type'] = 'WVI'
    wvi = wvi.rename(columns={'ID': 'view_id', 'X': 'x', 'Y': 'y', 'Z': 'z',
                               'Heading': 'heading'})
    viewpoints.append(wvi[['view_type', 'view_id', 'x', 'y', 'z', 'heading']])

    dvi = pd.read_csv(os.path.join(base_dir, "viewpoints/DVI_position_60.csv"))
    dvi['view_type'] = 'DVI'
    dvi = dvi.rename(columns={'id': 'view_id'})
    # DVI might have different heading column name
    if 'heading' not in dvi.columns:
        dvi = dvi.rename(columns={'heading_id': 'heading'})
    viewpoints.append(dvi[['view_type', 'view_id', 'x', 'y', 'z', 'heading']])

    obs = pd.concat(viewpoints, ignore_index=True)

    # One-hot encode view type
    obs['is_svi'] = (obs['view_type'] == 'SVI').astype(float)
    obs['is_wvi'] = (obs['view_type'] == 'WVI').astype(float)
    obs['is_dvi'] = (obs['view_type'] == 'DVI').astype(float)

    # Encode heading
    heading_sin_cos = obs['heading'].apply(encode_heading_sin_cos)
    obs['heading_sin'] = heading_sin_cos.apply(lambda x: x[0])
    obs['heading_cos'] = heading_sin_cos.apply(lambda x: x[1])

    # Approximate height above ground (relative to lowest Z in dataset)
    obs['rel_height'] = obs['z'] - obs['z'].min()

    # Create unique node IDs
    obs['node_id'] = obs.apply(lambda r: f"{r['view_type']}_{r['view_id']}", axis=1)

    # Select feature columns
    feature_cols = ['node_id', 'view_type', 'view_id', 'x', 'y', 'z',
                    'heading', 'heading_sin', 'heading_cos',
                    'is_svi', 'is_wvi', 'is_dvi', 'rel_height']

    obs_features = obs[feature_cols].copy()
    obs_features.set_index('node_id', inplace=True)

    print(f"Built observation features: {len(obs_features)} nodes")
    return obs_features


def build_greenery_features(base_dir):
    """
    Build greenery node features.

    Features:
    - center_x, center_y, center_z
    - width, height, area
    - R, G, B (mean color - proxy for vegetation density)
    """
    greenery = pd.read_csv(os.path.join(base_dir, "greenery/df_grid_ply.csv"))
    greenery['greenery_id'] = greenery.apply(
        lambda r: f"{int(r['grid_idx_x'])}_{int(r['grid_idx_y'])}", axis=1)

    greenery['center_x'] = (greenery['grid_coords_min_x'] + greenery['grid_coords_max_x']) / 2
    greenery['center_y'] = (greenery['grid_coords_min_y'] + greenery['grid_coords_max_y']) / 2
    greenery['center_z'] = 0.0  # approximate

    greenery['width'] = greenery['grid_coords_max_x'] - greenery['grid_coords_min_x']
    greenery['height'] = greenery['grid_coords_max_y'] - greenery['grid_coords_min_y']
    greenery['area'] = greenery['width'] * greenery['height']

    # Normalize colors to [0, 1]
    greenery['R_norm'] = greenery['R'] / 255.0
    greenery['G_norm'] = greenery['G'] / 255.0
    greenery['B_norm'] = greenery['B'] / 255.0

    # Greenness index (simple proxy)
    greenery['greenness'] = (2 * greenery['G_norm'] - greenery['R_norm'] - greenery['B_norm']).clip(0, 1)

    feature_cols = ['greenery_id', 'grid_idx_x', 'grid_idx_y',
                    'center_x', 'center_y', 'center_z',
                    'width', 'height', 'area',
                    'R_norm', 'G_norm', 'B_norm', 'greenness',
                    'raster_value']

    greenery_features = greenery[feature_cols].copy()
    greenery_features.set_index('greenery_id', inplace=True)

    print(f"Built greenery features: {len(greenery_features)} nodes")
    return greenery_features


def build_edge_features(candidate_edges_df):
    """
    Build edge features for candidate edges.

    Features:
    - horizontal_distance
    - euclidean_distance_3d
    - vertical_distance
    - azimuth_difference_sin, azimuth_difference_cos
    - log_distance
    - distance_inverse
    """
    edges = candidate_edges_df.copy()

    # Log transform distances
    edges['log_horizontal_distance'] = np.log1p(edges['horizontal_distance'])
    edges['log_euclidean_distance_3d'] = np.log1p(edges['euclidean_distance_3d'])

    # Inverse distance
    edges['inv_horizontal_distance'] = 1.0 / (edges['horizontal_distance'] + 1.0)

    # Azimuth difference sin/cos
    az_rad = np.radians(edges['azimuth_difference_deg'])
    edges['azimuth_diff_sin'] = np.sin(az_rad)
    edges['azimuth_diff_cos'] = np.cos(az_rad)

    return edges


def normalize_features(features_df, feature_cols, stats=None, fit=True):
    """
    Z-score normalize features. If fit=True, compute and return stats.
    If fit=False, use provided stats.
    """
    if fit:
        stats = {}
        for col in feature_cols:
            mean = features_df[col].mean()
            std = features_df[col].std()
            if std < 1e-10:
                std = 1.0
            stats[col] = {'mean': mean, 'std': std}
            features_df[col] = (features_df[col] - mean) / std
        return features_df, stats
    else:
        for col in feature_cols:
            if col in stats:
                mean, std = stats[col]['mean'], stats[col]['std']
                features_df[col] = (features_df[col] - mean) / std
        return features_df


def prepare_model_data(base_dir):
    """
    Prepare all data needed for model training.

    Returns:
        obs_features: Observation node features
        grn_features: Greenery node features
        candidate_edges: Candidate edges with features
        positive_edges: Positive edges (labels)
        split_info: Train/val/test split information
    """
    processed_dir = os.path.join(base_dir, "gvvi_graph_demo/processed")

    # Load data
    candidates = pd.read_parquet(os.path.join(processed_dir, "candidate_edges.parquet"))
    positives = pd.read_parquet(os.path.join(processed_dir, "positive_edges.parquet"))

    # Build features
    obs_features = build_observation_features(base_dir)
    grn_features = build_greenery_features(base_dir)

    # Build edge features
    candidate_edges = build_edge_features(candidates)

    # Load split info
    greenery_splits = pd.read_parquet(os.path.join(processed_dir, "greenery_splits.parquet"))
    viewpoint_splits = pd.read_parquet(os.path.join(processed_dir, "viewpoint_splits.parquet"))

    # Add split info to edges
    # Edges belong to split based on greenery grid's block
    greenery_split_map = greenery_splits.set_index('greenery_id')['split'].to_dict()
    viewpoint_split_map = {}
    for _, row in viewpoint_splits.iterrows():
        key = f"{row['view_type']}_{row['view_id']}"
        # Need block_id -> split mapping
        block_split_map = greenery_splits.set_index('block_id')['split'].to_dict()
        viewpoint_split_map[key] = block_split_map.get(row['block_id'], 'unknown')

    # For candidate edges, assign split based on greenery grid
    candidate_edges['greenery_split'] = candidate_edges['greenery_id'].map(greenery_split_map)
    candidate_edges['view_node_id'] = candidate_edges.apply(
        lambda r: f"{r['view_type']}_{r['view_id']}", axis=1)
    candidate_edges['view_split'] = candidate_edges['view_node_id'].map(viewpoint_split_map)

    # Edge split: use greenery split (conservative)
    # An edge is in train only if greenery is in train block
    candidate_edges['split'] = candidate_edges['greenery_split']

    # Label edges: merge with positive edges
    pos_key_set = set()
    for _, row in positives.iterrows():
        pos_key_set.add((row['view_type'], row['view_id'], row['greenery_id']))

    candidate_edges['is_visible'] = candidate_edges.apply(
        lambda r: (r['view_type'], r['view_id'], r['greenery_id']) in pos_key_set,
        axis=1
    ).astype(int)

    # Merge pixel count for visible edges
    positives_keyed = positives.set_index(['view_type', 'view_id', 'greenery_id'])
    candidate_edges = candidate_edges.join(
        positives_keyed['pixel_count'],
        on=['view_type', 'view_id', 'greenery_id'],
        how='left'
    )
    candidate_edges['pixel_count'] = candidate_edges['pixel_count'].fillna(0).astype(int)

    print(f"\nPrepared model data:")
    print(f"  Observation nodes: {len(obs_features)}")
    print(f"  Greenery nodes: {len(grn_features)}")
    print(f"  Candidate edges: {len(candidate_edges)}")
    print(f"  Positive edges: {candidate_edges['is_visible'].sum()}")
    print(f"  Negative edges: {(candidate_edges['is_visible'] == 0).sum()}")
    for split_name in ['train', 'val', 'test']:
        split_edges = candidate_edges[candidate_edges['split'] == split_name]
        print(f"  {split_name}: {len(split_edges)} edges, "
              f"{split_edges['is_visible'].sum()} positive")

    return obs_features, grn_features, candidate_edges, positives


if __name__ == '__main__':
    base_dir = os.environ.get('GVVI_BASE_DIR',
        "/Users/wangfugui/线上交流_forLI/05_3D_CIM_概念示例/GVVI_GNN_Demo实施包")
    prepare_model_data(base_dir)
