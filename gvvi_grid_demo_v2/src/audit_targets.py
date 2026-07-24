"""Audit targets: verify official GVVI replay and report data statistics."""
import numpy as np
import rasterio


def normalize_raster(array):
    return (array - array.min()) / (array.max() - array.min())


def audit_targets(data_root, grid_df):
    """Verify official GVVI replay and print target statistics."""
    print("=" * 60)
    print("AUDIT: Official GVVI Replay Check")
    print("=" * 60)

    # Read all reference rasters
    with rasterio.open(f"{data_root}/reference/GVVI_S.tif") as src:
        gvvi_s = src.read(1)
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
    with rasterio.open(f"{data_root}/reference/GVVI_W.tif") as src:
        gvvi_w = src.read(1)
    with rasterio.open(f"{data_root}/reference/GVVI_D.tif") as src:
        gvvi_d = src.read(1)
    with rasterio.open(f"{data_root}/reference/GVVI.tif") as src:
        gvvi_official = src.read(1)

    print(f"Raster shape: {height} x {width}")
    print(f"CRS: {crs}")
    print(f"GVVI_S range: [{gvvi_s.min():.1f}, {gvvi_s.max():.1f}]")
    print(f"GVVI_W range: [{gvvi_w.min():.1f}, {gvvi_w.max():.1f}]")
    print(f"GVVI_D range: [{gvvi_d.min():.1f}, {gvvi_d.max():.1f}]")

    # Check grid indices are in bounds
    xs = grid_df["grid_idx_x"].values
    ys = grid_df["grid_idx_y"].values
    assert xs.min() >= 0 and xs.max() < height, f"grid_idx_x out of bounds: [{xs.min()}, {xs.max()}] vs height={height}"
    assert ys.min() >= 0 and ys.max() < width, f"grid_idx_y out of bounds: [{ys.min()}, {ys.max()}] vs width={width}"
    print(f"Grid indices in bounds: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")

    # Extract per-node targets
    targets_s = gvvi_s[xs, ys]
    targets_w = gvvi_w[xs, ys]
    targets_d = gvvi_d[xs, ys]
    targets_gvvi = gvvi_official[xs, ys]

    # Official replay
    s_norm = normalize_raster(gvvi_s)
    w_norm = normalize_raster(gvvi_w)
    d_norm = normalize_raster(gvvi_d)
    replay = 3.0 / 6.0 * s_norm + 2.0 / 6.0 * w_norm + 1.0 / 6.0 * d_norm

    replay_nodes = replay[xs, ys]
    diff = np.abs(replay_nodes - targets_gvvi)
    max_diff = diff.max()

    print(f"\nOfficial replay check:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {diff.mean():.6e}")
    print(f"  Replay matches official: {max_diff < 1e-10}")

    if max_diff >= 1e-10:
        print("ERROR: Official GVVI replay does not match reference/GVVI.tif!")
        raise RuntimeError("Official GVVI replay failed. Check normalization and indexing.")

    # Normalize per-node targets to [0, 1] using global min/max from full rasters
    g_min_s, g_max_s = gvvi_s.min(), gvvi_s.max()
    g_min_w, g_max_w = gvvi_w.min(), gvvi_w.max()
    g_min_d, g_max_d = gvvi_d.min(), gvvi_d.max()
    eps = 1e-10
    norm_s = (targets_s - g_min_s) / (g_max_s - g_min_s + eps)
    norm_w = (targets_w - g_min_w) / (g_max_w - g_min_w + eps)
    norm_d = (targets_d - g_min_d) / (g_max_d - g_min_d + eps)
    norm_gvvi = targets_gvvi  # Already normalized per official replay

    # Statistics
    nonzero_s = (targets_s > 0).sum()
    nonzero_w = (targets_w > 0).sum()
    nonzero_d = (targets_d > 0).sum()
    nonzero_gvvi = (targets_gvvi > 0).sum()

    print(f"\nNode target statistics ({len(grid_df)} nodes):")
    print(f"  GVVI_S: nonzero={nonzero_s}, norm_mean={norm_s.mean():.4f}, norm_std={norm_s.std():.4f}")
    print(f"  GVVI_W: nonzero={nonzero_w}, norm_mean={norm_w.mean():.4f}, norm_std={norm_w.std():.4f}")
    print(f"  GVVI_D: nonzero={nonzero_d}, norm_mean={norm_d.mean():.4f}, norm_std={norm_d.std():.4f}")
    print(f"  GVVI:   nonzero={nonzero_gvvi}, mean={norm_gvvi.mean():.4f}, std={norm_gvvi.std():.4f}")

    print("\nAUDIT PASSED: Official GVVI replay verified.\n")

    targets = {
        "gvvi_s": norm_s.astype(np.float32),
        "gvvi_w": norm_w.astype(np.float32),
        "gvvi_d": norm_d.astype(np.float32),
        "gvvi": norm_gvvi.astype(np.float32),
    }
    raster_meta = {
        "height": height,
        "width": width,
        "transform": transform,
        "crs": crs,
        "gvvi_s_raw": gvvi_s,
        "gvvi_w_raw": gvvi_w,
        "gvvi_d_raw": gvvi_d,
        "gvvi_official": gvvi_official,
    }
    return targets, raster_meta
