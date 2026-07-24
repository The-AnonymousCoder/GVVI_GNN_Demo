"""Build V2 features with directional alignment, building/road/height context."""
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import time


def _bearing(x1, y1, x2, y2):
    """Compute bearing from (x1,y1) to (x2,y2) in degrees [0, 360)."""
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dx, dy))
    return np.mod(angle, 360.0)


def _wrap_delta(delta):
    """Wrap angle difference to [-180, 180]."""
    return np.mod(delta + 180.0, 360.0) - 180.0


def _build_tree(vp_df, x_col, y_col):
    coords = vp_df[[x_col, y_col]].values.astype(np.float64)
    return KDTree(coords), coords


def build_grid_features_v2(data_root, grid_df, cfg):
    """Build full V2 feature matrix with directional alignment features."""
    print("=" * 60)
    print("BUILD: V2 Grid Node Features")
    print("=" * 60)
    t0 = time.time()

    radii = cfg["features"]["radii"]
    h_radius = cfg["features"]["height_radius"]
    heading_radius = cfg["features"]["heading_radius"]
    density_radius = cfg["features"]["density_radius"]
    fov_half = cfg["features"]["fov_half_angle"]
    dist_scales = cfg["features"]["distance_scales"]

    grid_centers_x = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    grid_centers_y = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    grid_centers = np.stack([grid_centers_x, grid_centers_y], axis=1).astype(np.float64)
    n_nodes = len(grid_df)

    feature_blocks = []
    feature_names = []

    # Viewpoint configs
    vp_configs = [
        ("svi", "SVI_position_road.csv", "X", "Y", "Z", "Heading"),
        ("wvi", "window_view_location.csv", "X", "Y", "Z", "Heading"),
        ("dvi", "DVI_position_60.csv", "x", "y", "z", "heading"),
    ]

    # ================================================================
    # BLOCK 1: Distance-based features (same as V1)
    # ================================================================
    for prefix, csv_name, x_col, y_col, z_col, h_col in vp_configs:
        vp_df = pd.read_csv(f"{data_root}/viewpoints/{csv_name}")
        vp_tree, vp_coords = _build_tree(vp_df, x_col, y_col)
        print(f"  {prefix}: {len(vp_df)} viewpoints")

        for r in radii:
            indices = vp_tree.query_radius(grid_centers, r=r)
            counts = np.array([len(idx) for idx in indices], dtype=np.float32)
            nearest = np.full(n_nodes, r, dtype=np.float32)
            mean_dist = np.full(n_nodes, r, dtype=np.float32)
            std_dist = np.zeros(n_nodes, dtype=np.float32)
            for i, idx in enumerate(indices):
                if len(idx) > 0:
                    dists = np.linalg.norm(vp_coords[idx] - grid_centers[i], axis=1)
                    nearest[i] = dists.min()
                    mean_dist[i] = dists.mean()
                    if len(idx) > 1:
                        std_dist[i] = dists.std()
            for stat, arr in [("count", counts), ("nearest", nearest), ("mean_dist", mean_dist), ("std_dist", std_dist)]:
                feature_names.append(f"{prefix}_{stat}_{r}")
                feature_blocks.append(arr)

        # Height features
        vp_z = vp_df[z_col].values.astype(np.float32)
        indices = vp_tree.query_radius(grid_centers, r=h_radius)
        mean_z, max_z, min_z, h_diff = np.zeros(n_nodes, dtype=np.float32), np.zeros(n_nodes, dtype=np.float32), np.zeros(n_nodes, dtype=np.float32), np.zeros(n_nodes, dtype=np.float32)
        for i, idx in enumerate(indices):
            if len(idx) > 0:
                zs = vp_z[idx]
                mean_z[i] = zs.mean(); max_z[i] = zs.max(); min_z[i] = zs.min(); h_diff[i] = -zs.mean()
        for stat, arr in [("mean_z", mean_z), ("max_z", max_z), ("min_z", min_z), ("height_diff", h_diff)]:
            feature_names.append(f"{prefix}_{stat}")
            feature_blocks.append(arr)

        # Heading features
        vp_heading = vp_df[h_col].values.astype(np.float32)
        indices = vp_tree.query_radius(grid_centers, r=heading_radius)
        mean_sin, mean_cos = np.zeros(n_nodes, dtype=np.float32), np.zeros(n_nodes, dtype=np.float32)
        for i, idx in enumerate(indices):
            if len(idx) > 0:
                h_rad = np.deg2rad(vp_heading[idx])
                mean_sin[i] = np.sin(h_rad).mean(); mean_cos[i] = np.cos(h_rad).mean()
        feature_names.append(f"{prefix}_mean_sin_heading"); feature_blocks.append(mean_sin)
        feature_names.append(f"{prefix}_mean_cos_heading"); feature_blocks.append(mean_cos)

        # Heading circular variance
        heading_var = np.zeros(n_nodes, dtype=np.float32)
        for i, idx in enumerate(indices):
            if len(idx) > 0:
                h_rad = np.deg2rad(vp_heading[idx])
                R = np.abs(np.mean(np.exp(1j * h_rad)))
                heading_var[i] = 1.0 - R
        feature_names.append(f"{prefix}_heading_var"); feature_blocks.append(heading_var)

    # ================================================================
    # BLOCK 2: Directional alignment features
    # ================================================================
    print("  Computing directional alignment features...")
    for prefix, csv_name, x_col, y_col, z_col, h_col in vp_configs:
        vp_df = pd.read_csv(f"{data_root}/viewpoints/{csv_name}")
        vp_tree, vp_coords = _build_tree(vp_df, x_col, y_col)
        vp_headings = vp_df[h_col].values.astype(np.float32)
        vp_z = vp_df[z_col].values.astype(np.float32) if z_col in vp_df.columns else np.zeros(len(vp_df))

        for r in radii:
            indices = vp_tree.query_radius(grid_centers, r=r)
            ff_count = np.zeros(n_nodes, dtype=np.float32)  # front-facing count
            ff_ratio = np.zeros(n_nodes, dtype=np.float32)  # front-facing ratio
            align_mean = np.zeros(n_nodes, dtype=np.float32)
            align_max = np.zeros(n_nodes, dtype=np.float32)
            ff_near = np.full(n_nodes, r, dtype=np.float32)  # nearest front-facing distance
            ff_kernel = np.zeros(n_nodes, dtype=np.float32)  # directional kernel sum

            for i, idx in enumerate(indices):
                if len(idx) > 0:
                    vp_x = vp_coords[idx, 0]
                    vp_y = vp_coords[idx, 1]
                    headings = vp_headings[idx]
                    gx, gy = grid_centers[i]
                    bearings = _bearing(vp_x, vp_y, gx, gy)
                    deltas = _wrap_delta(bearings - headings)
                    is_ff = np.abs(deltas) <= fov_half
                    cos_align = np.cos(np.deg2rad(deltas))
                    dists = np.linalg.norm(vp_coords[idx] - grid_centers[i], axis=1)

                    ff_count[i] = is_ff.sum()
                    total = len(idx)
                    ff_ratio[i] = ff_count[i] / total if total > 0 else 0.0
                    align_mean[i] = cos_align.mean()
                    align_max[i] = cos_align.max()
                    if is_ff.sum() > 0:
                        ff_near[i] = dists[is_ff].min()
                    ff_kernel[i] = np.sum(np.maximum(cos_align, 0) * np.exp(-dists / 50.0))

            feature_names.append(f"{prefix}_ff_count_{r}"); feature_blocks.append(ff_count)
            feature_names.append(f"{prefix}_ff_ratio_{r}"); feature_blocks.append(ff_ratio)
            feature_names.append(f"{prefix}_align_mean_{r}"); feature_blocks.append(align_mean)
            feature_names.append(f"{prefix}_align_max_{r}"); feature_blocks.append(align_max)
            feature_names.append(f"{prefix}_ff_nearest_{r}"); feature_blocks.append(ff_near)
            feature_names.append(f"{prefix}_dir_kernel_{r}"); feature_blocks.append(ff_kernel)

        # Distance decay kernels
        for s in dist_scales:
            indices = vp_tree.query_radius(grid_centers, r=max(radii))
            kernel_vals = np.zeros(n_nodes, dtype=np.float32)
            for i, idx in enumerate(indices):
                if len(idx) > 0:
                    dists = np.linalg.norm(vp_coords[idx] - grid_centers[i], axis=1)
                    kernel_vals[i] = np.sum(np.exp(-dists / s))
            feature_names.append(f"{prefix}_kernel_{s}"); feature_blocks.append(kernel_vals)

        # Height quantiles
        indices = vp_tree.query_radius(grid_centers, r=h_radius)
        p25, p50, p75 = np.zeros(n_nodes, dtype=np.float32), np.zeros(n_nodes, dtype=np.float32), np.zeros(n_nodes, dtype=np.float32)
        for i, idx in enumerate(indices):
            if len(idx) > 0:
                zs = vp_z[idx]
                p25[i] = np.percentile(zs, 25); p50[i] = np.percentile(zs, 50); p75[i] = np.percentile(zs, 75)
        feature_names.append(f"{prefix}_z_p25"); feature_blocks.append(p25)
        feature_names.append(f"{prefix}_z_p50"); feature_blocks.append(p50)
        feature_names.append(f"{prefix}_z_p75"); feature_blocks.append(p75)

    # ================================================================
    # BLOCK 3: WVI building context
    # ================================================================
    print("  Computing WVI building context...")
    wvi_df = pd.read_csv(f"{data_root}/viewpoints/window_view_location.csv")
    wvi_tree, wvi_coords = _build_tree(wvi_df, "X", "Y")
    wvi_headings = wvi_df["Heading"].values.astype(np.float32)
    wvi_bldg = wvi_df["bldg_ID"].values
    wvi_z = wvi_df["Z"].values.astype(np.float32)
    wvi_roof = wvi_df["ROOFLEVEL"].values.astype(np.float32)
    wvi_base = wvi_df["BASELEVEL"].values.astype(np.float32)

    for r in [100, 250]:
        indices = wvi_tree.query_radius(grid_centers, r=r)
        n_bldgs = np.zeros(n_nodes, dtype=np.float32)
        mean_win_per_bldg = np.zeros(n_nodes, dtype=np.float32)
        max_win_per_bldg = np.zeros(n_nodes, dtype=np.float32)
        ff_win_ratio = np.zeros(n_nodes, dtype=np.float32)
        ff_win_near = np.full(n_nodes, r, dtype=np.float32)
        ff_win_kernel = np.zeros(n_nodes, dtype=np.float32)
        high_win_ratio = np.zeros(n_nodes, dtype=np.float32)
        low_win_ratio = np.zeros(n_nodes, dtype=np.float32)

        for i, idx in enumerate(indices):
            if len(idx) > 0:
                bldgs = wvi_bldg[idx]
                n_bldgs[i] = len(np.unique(bldgs))
                bldg_ids, bldg_counts = np.unique(bldgs, return_counts=True)
                mean_win_per_bldg[i] = bldg_counts.mean()
                max_win_per_bldg[i] = bldg_counts.max()

                gx, gy = grid_centers[i]
                vp_x = wvi_coords[idx, 0]; vp_y = wvi_coords[idx, 1]
                bearings = _bearing(vp_x, vp_y, gx, gy)
                deltas = _wrap_delta(bearings - wvi_headings[idx])
                is_ff = np.abs(deltas) <= fov_half
                total = len(idx)
                ff_win_ratio[i] = is_ff.sum() / total if total > 0 else 0.0

                if is_ff.sum() > 0:
                    ff_dists = np.linalg.norm(wvi_coords[idx][is_ff] - grid_centers[i], axis=1)
                    ff_win_near[i] = ff_dists.min()
                    cos_align = np.cos(np.deg2rad(deltas[is_ff]))
                    ff_win_kernel[i] = np.sum(np.maximum(cos_align, 0) * np.exp(-ff_dists / 50.0))

                zs = wvi_z[idx]
                if len(zs) > 0:
                    z_median = np.median(zs)
                    high_win_ratio[i] = (zs > z_median).sum() / total
                    low_win_ratio[i] = (zs <= z_median).sum() / total

        for stat, arr in [("n_bldgs", n_bldgs), ("mean_win_bldg", mean_win_per_bldg),
                          ("max_win_bldg", max_win_per_bldg), ("ff_win_ratio", ff_win_ratio),
                          ("ff_win_near", ff_win_near), ("ff_win_kernel", ff_win_kernel),
                          ("high_win_ratio", high_win_ratio), ("low_win_ratio", low_win_ratio)]:
            feature_names.append(f"wvi_{stat}_{r}"); feature_blocks.append(arr)

    # ================================================================
    # BLOCK 4: SVI road context
    # ================================================================
    print("  Computing SVI road context...")
    svi_df = pd.read_csv(f"{data_root}/viewpoints/SVI_position_road.csv")
    svi_tree, svi_coords = _build_tree(svi_df, "X", "Y")
    svi_headings = svi_df["Heading"].values.astype(np.float32)
    svi_routes = svi_df["RouteID"].values

    for r in [100, 250]:
        indices = svi_tree.query_radius(grid_centers, r=r)
        n_routes = np.zeros(n_nodes, dtype=np.float32)
        ff_svi_ratio = np.zeros(n_nodes, dtype=np.float32)
        ff_svi_near = np.full(n_nodes, r, dtype=np.float32)

        for i, idx in enumerate(indices):
            if len(idx) > 0:
                n_routes[i] = len(np.unique(svi_routes[idx]))
                gx, gy = grid_centers[i]
                vp_x = svi_coords[idx, 0]; vp_y = svi_coords[idx, 1]
                bearings = _bearing(vp_x, vp_y, gx, gy)
                deltas = _wrap_delta(bearings - svi_headings[idx])
                is_ff = np.abs(deltas) <= fov_half
                ff_svi_ratio[i] = is_ff.sum() / len(idx) if len(idx) > 0 else 0.0
                if is_ff.sum() > 0:
                    ff_svi_near[i] = np.linalg.norm(svi_coords[idx][is_ff] - grid_centers[i], axis=1).min()

        for stat, arr in [("n_routes", n_routes), ("ff_svi_ratio", ff_svi_ratio), ("ff_svi_near", ff_svi_near)]:
            feature_names.append(f"svi_{stat}_{r}"); feature_blocks.append(arr)

    # ================================================================
    # BLOCK 5: DVI height level context
    # ================================================================
    print("  Computing DVI height level context...")
    dvi_df = pd.read_csv(f"{data_root}/viewpoints/DVI_position_60.csv")
    dvi_tree, dvi_coords = _build_tree(dvi_df, "x", "y")
    dvi_headings = dvi_df["heading"].values.astype(np.float32)
    dvi_points = dvi_df["points_id"].values
    dvi_height_id = dvi_df["height_id"].values
    dvi_z = dvi_df["z"].values.astype(np.float32)

    for r in [100, 250]:
        indices = dvi_tree.query_radius(grid_centers, r=r)
        n_points = np.zeros(n_nodes, dtype=np.float32)
        n_heights = np.zeros(n_nodes, dtype=np.float32)
        ff_dvi_ratio = np.zeros(n_nodes, dtype=np.float32)
        ff_dvi_near = np.full(n_nodes, r, dtype=np.float32)

        for i, idx in enumerate(indices):
            if len(idx) > 0:
                n_points[i] = len(np.unique(dvi_points[idx]))
                n_heights[i] = len(np.unique(dvi_height_id[idx]))
                gx, gy = grid_centers[i]
                vp_x = dvi_coords[idx, 0]; vp_y = dvi_coords[idx, 1]
                bearings = _bearing(vp_x, vp_y, gx, gy)
                deltas = _wrap_delta(bearings - dvi_headings[idx])
                is_ff = np.abs(deltas) <= fov_half
                ff_dvi_ratio[i] = is_ff.sum() / len(idx) if len(idx) > 0 else 0.0
                if is_ff.sum() > 0:
                    ff_dvi_near[i] = np.linalg.norm(dvi_coords[idx][is_ff] - grid_centers[i], axis=1).min()

        for stat, arr in [("n_points", n_points), ("n_heights", n_heights), ("ff_dvi_ratio", ff_dvi_ratio), ("ff_dvi_near", ff_dvi_near)]:
            feature_names.append(f"dvi_{stat}_{r}"); feature_blocks.append(arr)

    # ================================================================
    # BLOCK 6: Grid self features
    # ================================================================
    x_min, x_max = grid_centers_x.min(), grid_centers_x.max()
    y_min, y_max = grid_centers_y.min(), grid_centers_y.max()
    x_local = (grid_centers_x - x_min) / (x_max - x_min)
    y_local = (grid_centers_y - y_min) / (y_max - y_min)
    feature_names.append("x_local"); feature_blocks.append(x_local.astype(np.float32))
    feature_names.append("y_local"); feature_blocks.append(y_local.astype(np.float32))

    norm_row = grid_df["grid_idx_x"].values.astype(np.float32) / 244.0
    norm_col = grid_df["grid_idx_y"].values.astype(np.float32) / 246.0
    feature_names.append("norm_row"); feature_blocks.append(norm_row)
    feature_names.append("norm_col"); feature_blocks.append(norm_col)

    tile_names = grid_df["ori_ply_name"].values
    tile_x = np.zeros(n_nodes, dtype=np.float32); tile_y = np.zeros(n_nodes, dtype=np.float32)
    for i, tn in enumerate(tile_names):
        if isinstance(tn, str) and tn.startswith("Tile_"):
            parts = tn.replace("Tile_", "").split("_")
            if len(parts) >= 2:
                tile_x[i] = float(parts[0]); tile_y[i] = float(parts[1])
    feature_names.append("tile_x"); feature_blocks.append(tile_x)
    feature_names.append("tile_y"); feature_blocks.append(tile_y)

    grid_tree = KDTree(grid_centers)
    density_indices = grid_tree.query_radius(grid_centers, r=density_radius)
    density = np.array([len(idx) - 1 for idx in density_indices], dtype=np.float32)
    feature_names.append("neighbor_density"); feature_blocks.append(density)

    # ================================================================
    # Log-transform count features
    # ================================================================
    count_cols = [k for k in feature_names if "count" in k]
    for i, name in enumerate(feature_names):
        if name in count_cols:
            feature_blocks[i] = np.log1p(feature_blocks[i])

    # ================================================================
    # Assemble and validate
    # ================================================================
    X = np.stack(feature_blocks, axis=1).astype(np.float32)
    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()
    print(f"\nFeature matrix: {X.shape}")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    if has_nan or has_inf:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    t1 = time.time()
    print(f"  Features: {X.shape[1]} dims, compute time: {t1-t0:.1f}s")
    print("V2 features built successfully.\n")
    return X, feature_names
