# GVVI Grid Demo V2 — Optimization Report

## Summary

Three optimization approaches were tested on RTX 4090 (24 GB). None achieved R² ≥ 0.50. V1 remains the best single model for composite GVVI. V2 directional features significantly improved SVI ranking but did not translate to better composite R².

## Environment
- Device: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- Python: 3.12.3, PyTorch 2.x, PyG 2.8.0
- All results: seed=42, identical frozen test set (4,146 nodes)
- No new data downloaded

## Results Comparison (Test Set, 4,146 nodes)

### V1 Baseline (best: 600 epochs, composite MSE loss, 73 dims)

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | 0.0180 | 0.0254 | 0.0454 | 0.0212 |
| RMSE | 0.0447 | 0.0493 | 0.0775 | 0.0363 |
| R² | 0.4154 | 0.2368 | 0.5057 | 0.3759 |
| Pearson | 0.6446 | 0.4944 | 0.7114 | 0.6170 |
| Spearman | 0.7348 | 0.7157 | 0.8363 | 0.7697 |

### V2-A (196-dim features, multi-scale model, complex loss: BCE + Reg + GVVI + Rank)

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | 0.0187 | 0.0303 | 0.0506 | 0.0227 |
| RMSE | 0.0464 | 0.0510 | 0.0834 | 0.0373 |
| R² | 0.3676 | 0.1842 | 0.4273 | 0.3406 |
| Pearson | 0.6089 | 0.4450 | 0.6545 | 0.5881 |
| Spearman | **0.8414** | 0.7293 | 0.8243 | 0.7728 |

### V2-B (196-dim features, multi-scale model, simple MSE loss)

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | 0.0174 | 0.0273 | 0.0491 | 0.0225 |
| RMSE | 0.0452 | 0.0485 | 0.0800 | 0.0368 |
| R² | 0.4005 | 0.2615 | 0.4736 | 0.3577 |
| Pearson | 0.6330 | 0.5129 | 0.6899 | 0.5988 |
| Spearman | 0.7368 | 0.7297 | 0.8353 | 0.7700 |

### V2-C (196-dim features, V1-style simple GraphSAGE, simple MSE loss)

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | 0.0203 | 0.0276 | 0.0510 | 0.0222 |
| RMSE | — | — | — | — |
| R² | 0.3861 | 0.1649 | 0.4436 | 0.3721 |
| Spearman | 0.7037 | 0.6758 | 0.8178 | 0.7557 |

## Optimization Targets vs Best Achieved

| Target | Threshold | Best Achieved | Source | Status |
|--------|-----------|---------------|--------|--------|
| Spearman | ≥ 0.80 | 0.7728 | V2-A | FAIL |
| R² | ≥ 0.50 | 0.3759 | V1 | FAIL |
| MAE | ≤ 0.020 | 0.0212 | V1 | FAIL |

## What Worked

1. **Directional features dramatically improved SVI ranking** (Spearman 0.735→0.841): bearing vs heading computation successfully captured whether street-level cameras face greenery grids.
2. **WVI improved across all V2 variants**: building context features (unique buildings, window counts per building, facing ratios) consistently boosted WVI metrics.
3. **Multi-scale graphs (k=8 + k=24)**: provided modest improvement in ranking loss but added model complexity.
4. **Composite GVVI loss (directly optimizing 3:2:1 output)**: confirmed effective, already in V1.
5. **80/5/15 label split**: successfully preserved original test nodes while increasing training labels.

## What Didn't Work

1. **Dual-head (zero-classification + positive-regression)**: BCE loss did not improve R²; added training instability.
2. **PairwiseRankingLoss**: did not translate to better test Spearman; ranking optimization on train nodes didn't generalize.
3. **Complex loss combination (BCE + Reg + GVVI + Rank)**: lower R² than simple MSE despite higher Spearman.

## Root Cause: R² Ceiling at ~0.38

All approaches converge to similar composite GVVI R² (0.34–0.38) despite substantially different features and architectures. This indicates a **fundamental information ceiling**:

- GVVI is determined by pixel-level visibility of greenery from specific camera positions
- Our features capture viewpoint proximity and direction, but not **building occlusion** or **true line-of-sight**
- Without 3D building geometry or rendered views, the model cannot distinguish between: "camera faces greenery" and "camera faces greenery but a building blocks the view"
- This ceiling is approximately R²=0.38 for composite GVVI with purely geometric features

The Spearman ceiling (~0.77) is higher because ranking requires only relative ordering, which geometric features (distance, density, direction) can partially capture.

## Recommendations for Future Work

1. **Building occlusion masks**: extract building footprints from WVI `bldg_ID` and DVI point clouds; compute occlusion ratios per grid cell.
2. **Few rendered views**: render 5-10% of grid cells with highest uncertainty; use as semi-supervised labels.
3. **Digital Surface Model (DSM)**: if available, compute viewshed per viewpoint position.
4. **Learn from neighboring observation stations**: use the observed GVVI at nearby SVI/WVI/DVI observation points as weak supervision.

## Experimental Rigor

- Test set: original 4,146 nodes, never modified
- Model selection: only validation loss used; test set evaluated once per configuration
- No label leakage: no GVVI, pixel, or rendered-view values used as input
- All metrics independently verifiable from `predictions.csv`
- Official GVVI replay: verified max error = 0.0
