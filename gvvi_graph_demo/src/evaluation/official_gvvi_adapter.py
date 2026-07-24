"""
Phase 2: GVVI Replay Test.

Reproduce the official GVVI computation using:
1. Exploded positive edges (aggregated per greenery grid)
2. The veg_raster_buffer_100.tif for grid mask
3. Official normalize_raster (min-max normalization)
4. SVI:WVI:DVI = 3:2:1 weighting

This validates that our data understanding is correct before training any model.
"""

import pandas as pd
import numpy as np
import rasterio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def normalize_raster(array):
    """Official min-max normalization from utils.py."""
    array_min = array.min()
    array_max = array.max()
    if array_max == array_min:
        return np.zeros_like(array, dtype=np.float64)
    array_normal = (array - array_min) / (array_max - array_min)
    return array_normal


def write_raster(data, save_path, height, width, transform, crs):
    """Write a GeoTIFF raster."""
    with rasterio.open(save_path, 'w', driver='GTiff',
                       height=height, width=width, count=1,
                       dtype=data.dtype, transform=transform, crs=crs) as dst:
        dst.write(data, 1)


def compute_total_pixel_rasters(positive_edges_df, veg_raster_path, output_dir):
    """
    Aggregate edge pixel counts per greenery grid for each view type.
    Write intermediate total_pixel rasters (pre-normalization).
    """
    with rasterio.open(veg_raster_path) as src:
        raster = src.read(1)
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    # Parse greenery_id back to grid_idx
    positive_edges = positive_edges_df.copy()
    coords = positive_edges['greenery_id'].str.split('_', expand=True)
    positive_edges['grid_idx_x'] = coords[0].astype(int)
    positive_edges['grid_idx_y'] = coords[1].astype(int)

    name_dict = {'SVI': 'GVVI_S', 'WVI': 'GVVI_W', 'DVI': 'GVVI_D'}
    output_rasters = {}

    for view_type in ['SVI', 'WVI', 'DVI']:
        # Aggregate total_pixel per greenery grid
        vt_edges = positive_edges[positive_edges['view_type'] == view_type]
        grid_sums = vt_edges.groupby(['grid_idx_x', 'grid_idx_y'])['pixel_count'].sum().reset_index()
        grid_sums = grid_sums.rename(columns={'pixel_count': 'total_pixel'})

        # Create raster
        grid = np.zeros((height, width), dtype=np.float64)

        for _, row in grid_sums.iterrows():
            x, y = int(row['grid_idx_x']), int(row['grid_idx_y'])
            value = raster[x, y]
            if value != 0:
                grid[x, y] += row['total_pixel']

        save_path = os.path.join(output_dir, f"replay_{name_dict[view_type]}.tif")
        write_raster(grid, save_path, height, width, transform, crs)
        output_rasters[view_type] = save_path

        nonzero = (grid != 0).sum()
        print(f"  {view_type} ({name_dict[view_type]}): "
              f"non-zero cells={nonzero}, max={grid.max():.1f}, min={grid[grid>0].min():.1f}")

    return output_rasters, height, width, transform, crs, raster


def compute_gvvi(intermediate_rasters, veg_raster_path, output_dir):
    """
    Normalize each intermediate raster and compute weighted GVVI.
    SVI:WVI:DVI = 3:2:1
    """
    with rasterio.open(veg_raster_path) as src:
        mask_raster = src.read(1)
        mask = mask_raster != 0
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    name_dict = {'SVI': 'GVVI_S', 'WVI': 'GVVI_W', 'DVI': 'GVVI_D'}
    weight_dict = {'SVI': 3, 'WVI': 2, 'DVI': 1}
    total_weight = sum(weight_dict.values())

    gvvi_layers = []
    normalized_rasters = {}

    for view_type in ['SVI', 'WVI', 'DVI']:
        with rasterio.open(intermediate_rasters[view_type]) as src:
            raster = src.read(1).astype(np.float64)

        print(f"  Processing {view_type} raster: max={raster.max():.1f}, min={raster.min():.1f}")

        # Normalize using official min-max
        raster_norm = normalize_raster(raster)

        # Save normalized version
        norm_path = os.path.join(output_dir, f"replay_{name_dict[view_type]}_norm.tif")
        write_raster(raster_norm, norm_path, height, width, transform, crs)
        normalized_rasters[view_type] = norm_path

        weight = weight_dict[view_type] / total_weight
        print(f"  Weight for {view_type}: {weight:.4f}")

        weighted = raster_norm * weight
        gvvi_layers.append(weighted)

    # Sum weighted layers
    gvvi = np.sum(gvvi_layers, axis=0).reshape((height, width))

    print(f"\n  Final GVVI (masked): max={gvvi[mask].max():.6f}, "
          f"min={gvvi[mask].min():.6f}, mean={gvvi[mask].mean():.6f}")

    gvvi_path = os.path.join(output_dir, "replay_GVVI.tif")
    write_raster(gvvi, gvvi_path, height, width, transform, crs)

    return gvvi, gvvi_path, mask, normalized_rasters


def compare_with_reference(base_dir, replay_dir):
    """Compare replay rasters with official reference rasters."""
    import json

    comparisons = {}
    ref_files = {
        'GVVI_S': 'reference/GVVI_S.tif',
        'GVVI_W': 'reference/GVVI_W.tif',
        'GVVI_D': 'reference/GVVI_D.tif',
        'GVVI': 'reference/GVVI.tif',
    }

    print(f"\n{'='*60}")
    print("Comparing replay results with official reference:")

    for name, ref_path in ref_files.items():
        ref_full = os.path.join(base_dir, ref_path)
        replay_full = os.path.join(replay_dir, f"replay_{name}.tif")

        if not os.path.exists(replay_full):
            print(f"  {name}: replay file not found, skipping")
            continue

        with rasterio.open(ref_full) as src:
            ref_arr = src.read(1).astype(np.float64)
        with rasterio.open(replay_full) as src:
            replay_arr = src.read(1).astype(np.float64)

        # Compare on non-zero mask
        mask = (ref_arr != 0) | (replay_arr != 0)

        if mask.sum() > 0:
            diff = np.abs(ref_arr[mask] - replay_arr[mask])
            max_err = diff.max()
            mean_err = diff.mean()
            exact_match = (diff < 1e-10).sum()
            within_1pct = (diff < 0.01 * np.maximum(ref_arr[mask], 1e-10)).sum()
            consistent_ratio = exact_match / mask.sum()

            comparisons[name] = {
                'max_error': float(max_err),
                'mean_error': float(mean_err),
                'exact_match_cells': int(exact_match),
                'total_nonzero_cells': int(mask.sum()),
                'consistent_ratio': float(consistent_ratio),
                'ref_min': float(ref_arr[mask].min()),
                'ref_max': float(ref_arr[mask].max()),
                'replay_min': float(replay_arr[mask].min()),
                'replay_max': float(replay_arr[mask].max()),
            }

            print(f"  {name}:")
            print(f"    Max error: {max_err:.10f}")
            print(f"    Mean error: {mean_err:.10f}")
            print(f"    Consistent cells: {exact_match}/{mask.sum()} ({consistent_ratio*100:.2f}%)")
            print(f"    Ref range: [{comparisons[name]['ref_min']:.2f}, {comparisons[name]['ref_max']:.2f}]")
            print(f"    Replay range: [{comparisons[name]['replay_min']:.2f}, {comparisons[name]['replay_max']:.2f}]")

            if max_err < 1e-6:
                print(f"    ✓ PASS: floating-point tolerance")
            else:
                print(f"    ⚠ Differences detected")

    return comparisons


def main():
    base_dir = os.environ.get('GVVI_BASE_DIR',
        "/Users/wangfugui/线上交流_forLI/05_3D_CIM_概念示例/GVVI_GNN_Demo实施包")

    output_dir = os.path.join(base_dir, "gvvi_graph_demo/outputs")
    replay_dir = os.path.join(output_dir, "replay")
    os.makedirs(replay_dir, exist_ok=True)

    # Load positive edges
    pos_path = os.path.join(base_dir, "gvvi_graph_demo/processed/positive_edges.parquet")
    if not os.path.exists(pos_path):
        print("ERROR: positive_edges.parquet not found. Run explode_relations.py first.")
        sys.exit(1)

    positive_edges = pd.read_parquet(pos_path)
    print(f"Loaded {len(positive_edges)} positive edges")

    veg_raster_path = os.path.join(base_dir, "greenery/veg_raster_buffer_100.tif")

    # Step 1: Compute total_pixel rasters
    print(f"\n{'='*60}")
    print("Step 1: Computing total_pixel rasters from exploded edges...")
    intermediate_rasters, height, width, transform, crs, mask_raster = \
        compute_total_pixel_rasters(positive_edges, veg_raster_path, replay_dir)

    # Step 2: Compute GVVI with normalization and weighting
    print(f"\n{'='*60}")
    print("Step 2: Computing GVVI with 3:2:1 weighting...")
    gvvi, gvvi_path, mask, norm_rasters = compute_gvvi(
        intermediate_rasters, veg_raster_path, replay_dir)

    # Step 3: Compare with reference
    comparisons = compare_with_reference(base_dir, replay_dir)

    # Save comparison report
    import json
    report = {
        'replay_test': 'Comparison of replayed GVVI with official reference',
        'method': 'Uses exploded positive edges, official normalize_raster, 3:2:1 weighting',
        'results': comparisons,
        'all_pass': all(c['max_error'] < 1e-6 for c in comparisons.values())
    }
    with open(os.path.join(output_dir, "replay_test.json"), 'w') as f:
        json.dump(report, f, indent=2)

    # Write replay_test.md
    md_path = os.path.join(output_dir, "replay_test.md")
    with open(md_path, 'w') as f:
        f.write("# GVVI Replay Test Report\n\n")
        f.write("## Purpose\n\n")
        f.write("Verify that our data processing pipeline correctly reproduces ")
        f.write("the official GVVI computation before training any GNN model.\n\n")
        f.write("## Method\n\n")
        f.write("1. Explode relations CSV files into positive edges\n")
        f.write("2. Aggregate pixel counts per greenery grid for each view type\n")
        f.write("3. Write intermediate total_pixel rasters\n")
        f.write("4. Apply official `normalize_raster` (min-max normalization)\n")
        f.write("5. Apply 3:2:1 weighting: `GVVI = 3/6*GVVI_S + 2/6*GVVI_W + 1/6*GVVI_D`\n\n")
        f.write("## Results\n\n")
        f.write("| Raster | Max Error | Mean Error | Consistent Cells | Status |\n")
        f.write("|--------|-----------|------------|------------------|--------|\n")
        for name, c in comparisons.items():
            status = "✓ PASS" if c['max_error'] < 1e-6 else "⚠ DIFF"
            f.write(f"| {name} | {c['max_error']:.10f} | {c['mean_error']:.10f} | "
                    f"{c['consistent_ratio']*100:.2f}% | {status} |\n")

        all_pass = all(c['max_error'] < 1e-6 for c in comparisons.values())
        f.write(f"\n## Conclusion\n\n")
        if all_pass:
            f.write("✓ All rasters match the official reference within floating-point tolerance. ")
            f.write("Data understanding is confirmed. Proceed to model training.\n")
        else:
            f.write("⚠ Some differences detected between replay and reference. ")
            f.write("Investigate before proceeding.\n")

    print(f"\nSaved replay test report to {md_path}")
    return gvvi, comparisons


if __name__ == '__main__':
    main()
