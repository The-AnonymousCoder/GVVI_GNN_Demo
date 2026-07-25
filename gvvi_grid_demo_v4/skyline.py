"""Fast skyline coverage: per-grid building angular coverage."""
import numpy as np, pandas as pd, time
from sklearn.neighbors import KDTree

def build_skyline(data_root, grid_df, radii=[50, 100, 250]):
    t0 = time.time()
    centers_x = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    centers_y = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    centers = np.stack([centers_x, centers_y], axis=1); n = len(grid_df)

    wvi = pd.read_csv(f"{data_root}/viewpoints/window_view_location.csv")
    # Deduplicate facade line segments
    facade_cols = ["line_0_x","line_0_y","line_1_x","line_1_y","bldg_ID"]
    facades = wvi[facade_cols].drop_duplicates(subset=["line_0_x","line_0_y","line_1_x","line_1_y"])
    print(f"Unique facades: {len(facades)}")

    # Facade midpoints for KDTree
    fmx = ((facades["line_0_x"] + facades["line_1_x"]) / 2).values
    fmy = ((facades["line_0_y"] + facades["line_1_y"]) / 2).values
    fmid = np.stack([fmx, fmy], axis=1)
    ftree = KDTree(fmid)

    # Facade endpoints for angular computation
    fl0x = facades["line_0_x"].values; fl0y = facades["line_0_y"].values
    fl1x = facades["line_1_x"].values; fl1y = facades["line_1_y"].values

    features = {}
    for r in radii:
        sky_cover = np.zeros(n, dtype=np.float32)
        n_bldg = np.zeros(n, dtype=np.float32)
        near_dist = np.full(n, r, dtype=np.float32)
        total_length = np.zeros(n, dtype=np.float32)

        fidxs = ftree.query_radius(centers, r=r)
        for i in range(n):
            if i % 5000 == 0: print(f"  r={r}: {i}/{n}")
            fi = fidxs[i]
            if len(fi) == 0: continue

            gx, gy = centers[i]
            n_bldg[i] = len(np.unique(facades["bldg_ID"].values[fi]))

            # Angular coverage: for each facade, compute start/end angles from grid
            angles = []
            for j in fi:
                a0 = np.arctan2(fl0y[j]-gy, fl0x[j]-gx)
                a1 = np.arctan2(fl1y[j]-gy, fl1x[j]-gx)
                # Normalize to [-pi, pi] and ensure a0 < a1
                if a0 > a1: a0, a1 = a1, a0
                if a1 - a0 > np.pi:  # wraps around
                    angles.append((a1, np.pi))
                    angles.append((-np.pi, a0))
                else:
                    angles.append((a0, a1))
                # Distance
                d = np.sqrt((fmid[j,0]-gx)**2 + (fmid[j,1]-gy)**2)
                if d < near_dist[i]: near_dist[i] = d
                total_length[i] += np.sqrt((fl1x[j]-fl0x[j])**2 + (fl1y[j]-fl0y[j])**2)

            # Merge angular ranges to compute coverage
            if angles:
                angles.sort()
                merged = [list(angles[0])]
                for a in angles[1:]:
                    if a[0] <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], a[1])
                    else:
                        merged.append(list(a))
                coverage = sum(hi-lo for lo, hi in merged)
                sky_cover[i] = coverage / (2*np.pi)

        for s, a in [("sky_cover", sky_cover), ("n_bldg", n_bldg),
                      ("near_dist", near_dist), ("total_length", total_length)]:
            features[f"sky_{s}_{r}"] = a

    print(f"Done in {time.time()-t0:.1f}s")
    return features


if __name__ == "__main__":
    data_root = "/root/GVVI_GNN_Demo"
    grid_df = pd.read_csv(f"{data_root}/greenery/df_grid_ply.csv")
    feats = build_skyline(data_root, grid_df)
    X = np.stack(list(feats.values()), axis=1).astype(np.float32)
    names = list(feats.keys())
    np.savez_compressed(f"{data_root}/gvvi_grid_demo_v4/processed/skyline.npz",
                        features=X, names=np.array(names))
    print(f"Saved: {X.shape}, {names}")
