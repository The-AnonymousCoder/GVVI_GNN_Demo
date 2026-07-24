"""Build cheap spatial features for each greenery grid node."""
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def _build_viewpoint_tree(vp_df, x_col, y_col):
    """Build KDTree from viewpoint positions in EPSG:2326."""
    coords = vp_df[[x_col, y_col]].values.astype(np.float64)
    tree = KDTree(coords)
    return tree, coords


def _distance_features(grid_centers, vp_coords, vp_tree, radii):
    """Compute distance-based features for each radius.

    Returns a dict with keys like 'svi_count_50', 'svi_nearest_50', etc.
    """
    feats = {}
    for r in radii:
        indices = vp_tree.query_radius(grid_centers, r=r)
        counts = np.array([len(idx) for idx in indices], dtype=np.float32)
        nearest = np.full(len(grid_centers), r, dtype=np.float32)
        mean_dist = np.full(len(grid_centers), r, dtype=np.float32)
        std_dist = np.zeros(len(grid_centers), dtype=np.float32)

        for i, idx in enumerate(indices):
            if len(idx) > 0:
                dists = np.linalg.norm(vp_coords[idx] - grid_centers[i], axis=1)
                nearest[i] = dists.min()
                mean_dist[i] = dists.mean()
                if len(idx) > 1:
                    std_dist[i] = dists.std()
        feats[f"count_{r}"] = counts
        feats[f"nearest_{r}"] = nearest
        feats[f"mean_dist_{r}"] = mean_dist
        feats[f"std_dist_{r}"] = std_dist
    return feats


def _height_features(grid_centers, vp_coords, vp_z, vp_tree, radius, grid_z=0.0):
    """Compute height features within a fixed radius."""
    indices = vp_tree.query_radius(grid_centers, r=radius)
    n = len(grid_centers)
    mean_z = np.zeros(n, dtype=np.float32)
    max_z = np.zeros(n, dtype=np.float32)
    min_z = np.zeros(n, dtype=np.float32)
    height_diff = np.zeros(n, dtype=np.float32)

    for i, idx in enumerate(indices):
        if len(idx) > 0:
            zs = vp_z[idx]
            mean_z[i] = zs.mean()
            max_z[i] = zs.max()
            min_z[i] = zs.min()
            height_diff[i] = grid_z - zs.mean()
        else:
            mean_z[i] = 0.0
            max_z[i] = 0.0
            min_z[i] = 0.0
            height_diff[i] = 0.0

    return {
        "mean_z": mean_z,
        "max_z": max_z,
        "min_z": min_z,
        "height_diff": height_diff,
    }


def _heading_features(grid_centers, vp_coords, vp_heading, vp_tree, radius):
    """Compute mean sin/cos of heading within radius."""
    indices = vp_tree.query_radius(grid_centers, r=radius)
    n = len(grid_centers)
    mean_sin = np.zeros(n, dtype=np.float32)
    mean_cos = np.zeros(n, dtype=np.float32)

    for i, idx in enumerate(indices):
        if len(idx) > 0:
            headings_rad = np.deg2rad(vp_heading[idx])
            mean_sin[i] = np.sin(headings_rad).mean()
            mean_cos[i] = np.cos(headings_rad).mean()

    return {"mean_sin_heading": mean_sin, "mean_cos_heading": mean_cos}


def build_grid_features(data_root, grid_df, cfg):
    """Build the full feature matrix for all grid nodes.

    Returns:
        features: (N, D) float32 array
        feature_names: list of D strings
    """
    print("=" * 60)
    print("BUILD: Grid Node Features")
    print("=" * 60)

    radii = cfg["features"]["radii"]
    h_radius = cfg["features"]["height_radius"]
    heading_radius = cfg["features"]["heading_radius"]
    density_radius = cfg["features"]["density_radius"]

    # Compute grid centers
    grid_centers_x = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    grid_centers_y = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    grid_centers = np.stack([grid_centers_x, grid_centers_y], axis=1).astype(np.float64)
    n_nodes = len(grid_df)

    # Viewpoint types
    vp_configs = [
        ("svi", "SVI_position_road.csv", "X", "Y", "Z", "Heading"),
        ("wvi", "window_view_location.csv", "X", "Y", "Z", "Heading"),
        ("dvi", "DVI_position_60.csv", "x", "y", "z", "heading"),
    ]

    feature_blocks = []
    feature_names = []

    for prefix, csv_name, x_col, y_col, z_col, h_col in vp_configs:
        vp_df = pd.read_csv(f"{data_root}/viewpoints/{csv_name}")
        vp_tree, vp_coords = _build_viewpoint_tree(vp_df, x_col, y_col)
        print(f"  {prefix}: {len(vp_df)} viewpoints")

        # Distance features
        dist_feats = _distance_features(grid_centers, vp_coords, vp_tree, radii)
        for r in radii:
            for stat in ["count", "nearest", "mean_dist", "std_dist"]:
                key = f"{prefix}_{stat}_{r}"
                feature_blocks.append(dist_feats[f"{stat}_{r}"])
                feature_names.append(key)

        # Height features
        vp_z = vp_df[z_col].values.astype(np.float32)
        h_feats = _height_features(grid_centers, vp_coords, vp_z, vp_tree, h_radius)
        for stat in ["mean_z", "max_z", "min_z", "height_diff"]:
            key = f"{prefix}_{stat}"
            feature_blocks.append(h_feats[stat])
            feature_names.append(key)

        # Heading features
        vp_heading = vp_df[h_col].values.astype(np.float32)
        head_feats = _heading_features(grid_centers, vp_coords, vp_heading, vp_tree, heading_radius)
        for stat in ["mean_sin_heading", "mean_cos_heading"]:
            key = f"{prefix}_{stat}"
            feature_blocks.append(head_feats[stat])
            feature_names.append(key)

    # Grid self features
    x_min, x_max = grid_centers_x.min(), grid_centers_x.max()
    y_min, y_max = grid_centers_y.min(), grid_centers_y.max()
    x_local = (grid_centers_x - x_min) / (x_max - x_min)
    y_local = (grid_centers_y - y_min) / (y_max - y_min)
    feature_blocks.append(x_local.astype(np.float32))
    feature_names.append("x_local")
    feature_blocks.append(y_local.astype(np.float32))
    feature_names.append("y_local")

    # Row/col normalized
    norm_row = grid_df["grid_idx_x"].values.astype(np.float32) / 244.0
    norm_col = grid_df["grid_idx_y"].values.astype(np.float32) / 246.0
    feature_blocks.append(norm_row)
    feature_names.append("norm_row")
    feature_blocks.append(norm_col)
    feature_names.append("norm_col")

    # OBJ tile encoding: parse "Tile_+005_+016" -> (5, 16)
    tile_names = grid_df["ori_ply_name"].values
    tile_x = np.zeros(n_nodes, dtype=np.float32)
    tile_y = np.zeros(n_nodes, dtype=np.float32)
    for i, tn in enumerate(tile_names):
        if isinstance(tn, str) and tn.startswith("Tile_"):
            parts = tn.replace("Tile_", "").split("_")
            if len(parts) >= 2:
                tile_x[i] = float(parts[0])
                tile_y[i] = float(parts[1])
    feature_blocks.append(tile_x)
    feature_names.append("tile_x")
    feature_blocks.append(tile_y)
    feature_names.append("tile_y")

    # Neighbor density
    grid_tree = KDTree(grid_centers)
    density_indices = grid_tree.query_radius(grid_centers, r=density_radius)
    density = np.array([len(idx) - 1 for idx in density_indices], dtype=np.float32)
    feature_blocks.append(density)
    feature_names.append("neighbor_density")

    # Log-transform count features (skewed distributions)
    count_cols = [k for k in feature_names if "count" in k]
    for i, name in enumerate(feature_names):
        if name in count_cols:
            feature_blocks[i] = np.log1p(feature_blocks[i])

    X = np.stack(feature_blocks, axis=1).astype(np.float32)

    # Check for NaN/Inf
    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()
    print(f"\nFeature matrix: {X.shape}")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    if has_nan or has_inf:
        print("WARNING: NaN/Inf detected in features, applying fix...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Feature names ({len(feature_names)}): {feature_names}")
    print("Features built successfully.\n")

    return X, feature_names
