# GVVI Grid Demo V2 — Audit Report

## 1. Official GVVI Replay

- GVVI = 3/6 × normalize(GVVI_S) + 2/6 × normalize(GVVI_W) + 1/6 × normalize(GVVI_D)
- Max difference vs reference/GVVI.tif: 0.0
- Status: PASS

## 2. Test Set Integrity

- Test nodes: 4,146, originally from `gvvi_grid_demo/processed/split_seed42.npz`
- All V2 splits verified to preserve same test_idx
- No test node participated in training
- Status: PASS

## 3. Label Leakage Check

### Input features do NOT include:
- R/G/B unique colors
- total_pixel, count, img_id, pixel_wvi
- Any GVVI value (S, W, D, or composite)
- Neighbor GVVI values
- Test-label-derived statistics

### Input features DO include:
- Viewpoint positions (X, Y, Z) and headings
- Derived geometric features (distance, bearing, height, direction)
- Building context (bldg_ID counts, per-building window stats)
- Road context (RouteID diversity)
- Height layer context (points_id, height_id)
- Grid spatial coordinates (x_local, y_local, norm_row, norm_col)
- Tile identifiers

Status: PASS — no direct label leakage. Spatial coordinates allow learning the study area's GVVI field, consistent with transductive node regression.

## 4. Predictions Consistency

- `predictions.csv` → metrics.json: independently verifiable
- CSV → GeoTIFF: same grid_idx_x/grid_idx_y indexing
- Composite GVVI: 3:2:1 ratio verified

Status: PASS

## 5. Data Integrity

- Grid nodes: 27,597 (no duplicates on grid_idx_x/grid_idx_y)
- GVVI_S raster: 247×245 float64, nonzero=14,244
- GVVI_W raster: 247×245 float64, nonzero=20,219
- GVVI_D raster: 247×245 float64, nonzero=20,123
- GVVI raster: 247×245 float64, nonzero=20,290
- CRS: EPSG:2326 (Hong Kong 1980 Grid)

Status: PASS

## 6. Split Configuration

- V1 (baseline): 70% train, 15% val, 15% test
- V2: 80% train, 5% val, 15% test (same test nodes)
- Stratification: GVVI decile-based
- Seed: 42 fixed

Status: PASS

## 7. Model Architecture Verification

### V1 (best model)
- Type: 2-layer GraphSAGE
- Params: ~61K
- Hidden: 256→128
- Loss: MSE + composite GVVI MSE
- Epochs: 600 max, patience 60

### V2 variants tested
- V2-A: multi-scale residual GraphSAGE + dual heads + complex loss (~810K params)
- V2-B: multi-scale residual GraphSAGE + dual heads + simple MSE (~810K params)
- V2-C: simple GraphSAGE + V2 features (~257K params)

## 8. Optimization Summary

Three independent optimization runs on RTX 4090, all with frozen test set:

| Run | Features | Model | Epochs | GVVI R² | GVVI Spearman |
|-----|----------|-------|--------|---------|---------------|
| V1 | 73 dims | Simple SAGE | 600 | 0.3759 | 0.7697 |
| V2-A | 196 dims | MultiScale | 348 | 0.3406 | 0.7728 |
| V2-B | 196 dims | MultiScale | ~350 | 0.3577 | 0.7700 |
| V2-C | 196 dims | Simple SAGE | 428 | 0.3721 | 0.7557 |

V1 has the highest composite GVVI R². V2-A has the highest SVI Spearman (0.841 vs 0.735).

## 9. Conclusion

- All data processing, splitting, and metric computation verified correct
- No test set modification, no label leakage
- Optimization ceiling at R² ≈ 0.38 attributed to missing building occlusion information
- V2 directional features improve SVI ranking but do not improve composite GVVI
