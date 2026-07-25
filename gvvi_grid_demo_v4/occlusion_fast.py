"""Fast occlusion proxy: viewpoint-grid same-side check with building facades."""
import numpy as np, pandas as pd, time
from sklearn.neighbors import KDTree

def build_occlusion_fast(data_root, grid_df, radii=[100, 250]):
    t0 = time.time()
    centers_x = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    centers_y = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    centers = np.stack([centers_x, centers_y], axis=1)
    n = len(grid_df)

    # SVI
    svi = pd.read_csv(f"{data_root}/viewpoints/SVI_position_road.csv")
    svi_coords = svi[["X", "Y"]].values.astype(np.float64)
    svi_tree = KDTree(svi_coords)

    # WVI facades (dedup by position + normal)
    wvi = pd.read_csv(f"{data_root}/viewpoints/window_view_location.csv")
    # Normalize normals
    nx = wvi["normal_x"].values.astype(np.float64)
    ny = wvi["normal_y"].values.astype(np.float64)
    nm = np.sqrt(nx**2 + ny**2) + 1e-10
    nx /= nm; ny /= nm
    # Use facade midpoints and normals (dedup)
    mid_x = (wvi["line_0_x"] + wvi["line_1_x"]) / 2
    mid_y = (wvi["line_0_y"] + wvi["line_1_y"]) / 2
    # Deduplicate: group by rounded midpoint + bldg_ID
    bldg_ids = wvi["bldg_ID"].values
    facade_data = {}
    for i in range(len(wvi)):
        key = (round(mid_x.iloc[i], 1), round(mid_y.iloc[i], 1), bldg_ids[i])
        if key not in facade_data:
            facade_data[key] = (mid_x.iloc[i], mid_y.iloc[i], nx[i], ny[i])
    facade_mid = np.array([[v[0], v[1]] for v in facade_data.values()])
    facade_nx = np.array([v[2] for v in facade_data.values()])
    facade_ny = np.array([v[3] for v in facade_data.values()])
    facade_tree = KDTree(facade_mid)
    n_facades = len(facade_mid)
    print(f"Grids: {n}, SVI: {len(svi_coords)}, Unique facades: {n_facades}")

    features = {}
    for r in radii:
        occ_ratio = np.zeros(n, dtype=np.float32)
        occ_count = np.zeros(n, dtype=np.float32)
        clear_ratio = np.zeros(n, dtype=np.float32)
        mean_blocks = np.zeros(n, dtype=np.float32)

        svi_idxs = svi_tree.query_radius(centers, r=r)
        for i in range(n):
            if i % 5000 == 0: print(f"  r={r}: {i}/{n}...")
            svi_idx = svi_idxs[i]
            if len(svi_idx) == 0:
                continue
            gx, gy = centers[i]

            # Find nearby facades
            fidx = facade_tree.query_radius([[gx, gy]], r=r*1.5)[0]
            if len(fidx) == 0:
                clear_ratio[i] = 1.0
                continue

            n_blocked = 0
            total_blocks = 0

            for si in svi_idx:
                sx, sy = svi_coords[si]
                # Check each nearby facade: is it between viewpoint and grid?
                n_blocks = 0
                for fi in fidx:
                    fx, fy = facade_mid[fi]
                    # Vector from facade to grid
                    dgx, dgy = gx - fx, gy - fy
                    # Vector from facade to viewpoint
                    dvx, dvy = sx - fx, sy - fy
                    # Dot with facade normal
                    dot_g = dgx * facade_nx[fi] + dgy * facade_ny[fi]
                    dot_v = dvx * facade_nx[fi] + dvy * facade_ny[fi]
                    # Facade blocks if grid is in front AND viewpoint is behind
                    # (different sides of the facade)
                    if dot_g > 0 and dot_v < 0:
                        n_blocks += 1
                if n_blocks > 0:
                    n_blocked += 1
                total_blocks += n_blocks

            occ_count[i] = n_blocked
            occ_ratio[i] = n_blocked / len(svi_idx)
            clear_ratio[i] = 1.0 - n_blocked / len(svi_idx)
            mean_blocks[i] = total_blocks / max(len(svi_idx), 1)

        for stat, arr in [("occ_count", occ_count), ("occ_ratio", occ_ratio),
                          ("clear_ratio", clear_ratio), ("mean_blocks", mean_blocks)]:
            features[f"fast_{stat}_{r}"] = arr

    print(f"Done in {time.time()-t0:.1f}s")
    return features


if __name__ == "__main__":
    data_root = "/root/GVVI_GNN_Demo"
    grid_df = pd.read_csv(f"{data_root}/greenery/df_grid_ply.csv")
    feats = build_occlusion_fast(data_root, grid_df)
    X = np.stack(list(feats.values()), axis=1).astype(np.float32)
    names = list(feats.keys())
    np.savez_compressed(f"{data_root}/gvvi_grid_demo_v4/processed/fast_occlusion.npz",
                        features=X, names=np.array(names))
    print(f"Saved: {X.shape}, {names}")
