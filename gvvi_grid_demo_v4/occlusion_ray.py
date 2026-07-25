"""Line-of-sight occlusion features using WVI facade line segments."""
import numpy as np, pandas as pd, time
from sklearn.neighbors import KDTree

def segments_intersect(p1, p2, p3, p4):
    """Check if line segments (p1,p2) and (p3,p4) intersect."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def build_occlusion_features(data_root, grid_df, svi_df, wvi_df, radii=[100, 250]):
    """For each grid, check sight-line occlusion from SVI viewpoints."""
    t0 = time.time()
    centers_x = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    centers_y = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    centers = np.stack([centers_x, centers_y], axis=1)
    n = len(grid_df)

    # SVI positions
    svi_coords = svi_df[["X", "Y"]].values.astype(np.float64)
    svi_tree = KDTree(svi_coords)

    # WVI facade lines (deduplicated per unique facade)
    facade_cols = ["line_0_x", "line_0_y", "line_1_x", "line_1_y"]
    facades = wvi_df[facade_cols].drop_duplicates().values.astype(np.float64)
    # Build KDTree on facade midpoints
    facade_mid_x = (facades[:, 0] + facades[:, 2]) / 2
    facade_mid_y = (facades[:, 1] + facades[:, 3]) / 2
    facade_mid = np.stack([facade_mid_x, facade_mid_y], axis=1)
    facade_tree = KDTree(facade_mid)

    print(f"Grids: {n}, SVI: {len(svi_coords)}, Facades: {len(facades)}")

    features = {}
    for r in radii:
        occ_count = np.zeros(n, dtype=np.float32)      # occluded viewpoints count
        occ_ratio = np.zeros(n, dtype=np.float32)      # fraction occluded
        n_blocks = np.zeros(n, dtype=np.float32)       # avg blocking facades per viewpoint
        max_block_dist = np.full(n, r, dtype=np.float32)  # max distance to blocking facade

        svi_idxs = svi_tree.query_radius(centers, r=r)
        for i in range(n):
            if i % 5000 == 0: print(f"  r={r}: {i}/{n}...")
            svi_idx = svi_idxs[i]
            if len(svi_idx) == 0:
                occ_ratio[i] = 0.0
                continue

            gx, gy = centers[i]
            # Find nearby facades (within 2*r to catch all potential occlusions)
            nearby_facades = facade_tree.query_radius([[gx, gy]], r=r*2)[0]
            if len(nearby_facades) == 0:
                continue

            n_blocked = 0
            total_blocks = 0
            max_bd = 0

            for si in svi_idx:
                sx, sy = svi_coords[si]
                blocked = False
                n_blocks_vp = 0
                for fi in nearby_facades:
                    l0x, l0y, l1x, l1y = facades[fi]
                    # Quick bounding box check
                    min_x = min(sx, gx); max_x = max(sx, gx)
                    min_y = min(sy, gy); max_y = max(sy, gy)
                    if max(l0x, l1x) < min_x or min(l0x, l1x) > max_x:
                        continue
                    if max(l0y, l1y) < min_y or min(l0y, l1y) > max_y:
                        continue
                    # Exact intersection test
                    if segments_intersect((sx,sy), (gx,gy), (l0x,l0y), (l1x,l1y)):
                        blocked = True
                        n_blocks_vp += 1
                        dist = np.sqrt((l0x-sx)**2 + (l0y-sy)**2)
                        if dist > max_bd: max_bd = dist
                if blocked:
                    n_blocked += 1
                total_blocks += n_blocks_vp

            occ_count[i] = n_blocked
            occ_ratio[i] = n_blocked / len(svi_idx)
            n_blocks[i] = total_blocks / max(len(svi_idx), 1)
            max_block_dist[i] = max_bd

        for stat, arr in [("occ_count", occ_count), ("occ_ratio", occ_ratio),
                          ("n_blocks", n_blocks), ("max_block_dist", max_block_dist)]:
            features[f"ray_{stat}_{r}"] = arr

    print(f"Occlusion features done in {time.time()-t0:.1f}s")
    return features


if __name__ == "__main__":
    data_root = "/root/GVVI_GNN_Demo"
    grid_df = pd.read_csv(f"{data_root}/greenery/df_grid_ply.csv")
    svi_df = pd.read_csv(f"{data_root}/viewpoints/SVI_position_road.csv")
    wvi_df = pd.read_csv(f"{data_root}/viewpoints/window_view_location.csv")

    feats = build_occlusion_features(data_root, grid_df, svi_df, wvi_df)
    # Save
    X = np.stack(list(feats.values()), axis=1).astype(np.float32)
    names = list(feats.keys())
    print(f"Saved: {X.shape}, names={names}")
    np.savez_compressed(f"{data_root}/gvvi_grid_demo_v4/processed/ray_occlusion.npz",
                        features=X, names=np.array(names))
