# GVVI Grid Demo — Final Report

## Environment
- Device: NVIDIA GeForce RTX 4090 (24 GB VRAM)
- Python: 3.12.3
- PyTorch: 2.x, PyG: 2.8.0
- Total runtime: 47.7s (0.8 min)

## Model
- Two-layer GraphSAGE (256 → 128 hidden dims)
- Input: 73 features, BatchNorm + Dropout(0.2)
- Optimizer: AdamW, lr=0.001, weight_decay=0.0001
- Loss: Weighted MSE (3:2:1 ratio per official GVVI)
- Max epochs: 300 (trained to completion, val_loss still decreasing)
- Seed: 42 (fixed)
- Parameters: 61,635

## Test Set Results (4,146 nodes, 15% stratified split)

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | 0.0191 | 0.0265 | 0.0472 | 0.0219 |
| RMSE | 0.0450 | 0.0498 | 0.0792 | 0.0366 |
| R² | 0.4049 | 0.2228 | 0.4844 | 0.3670 |
| Pearson | 0.6371 | 0.4767 | 0.6960 | 0.6074 |
| Spearman | 0.7123 | 0.7010 | 0.8258 | 0.7584 |

## Minimum Display Standards Check

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Spearman | ≥ 0.70 | 0.7584 | PASS |
| R² | ≥ 0.50 | 0.3670 | FAIL |
| MAE | ≤ 0.15 | 0.0219 | PASS |

**Overall**: 2 of 3 standards met. Ranking accuracy (Spearman) and absolute error (MAE) pass; variance explained (R²) is below threshold.

## Interpretation

The GNN achieves strong ranking performance (Spearman=0.758) and low absolute error (MAE=0.022 on [0,1] scale), demonstrating that cheap spatial features (viewpoint proximity, height, heading) combined with two-layer GraphSAGE can rank greenery grids by GVVI with reasonable accuracy.

The R²=0.367 indicates that geometric features alone explain only 37% of pixel-level GVVI variance. This is expected: GVVI depends on line-of-sight visibility through building geometry, which our features do not capture. DVI (drone, top-down) achieves the best R²=0.484 as drone viewpoints are less occluded; WVI (window) is hardest (R²=0.223) due to building occlusion.

This Demo validates a specific use case:
> **In-study-area rapid GVVI hotspot screening and prioritization** — cheap spatial features + GNN can rank which uncomputed grid cells are likely to have high/low GVVI, enabling efficient allocation of expensive rendering computation.

The Demo does NOT demonstrate:
- Exact GVVI prediction for individual grid cells
- Cross-city generalization
- Full replacement of Cesium/OpenGL rendering

## Generated Files

- `outputs/metrics.json` — all test metrics
- `outputs/predictions.csv` — per-node predictions
- `outputs/pred_GVVI_S.tif` — 247×245 GeoTIFF
- `outputs/pred_GVVI_W.tif` — 247×245 GeoTIFF
- `outputs/pred_GVVI_D.tif` — 247×245 GeoTIFF
- `outputs/pred_GVVI.tif` — 247×245 GeoTIFF (3:2:1 composite)
- `outputs/figures/official_gvvi_map.png`
- `outputs/figures/predicted_gvvi_map.png`
- `outputs/figures/abs_error_map.png`
- `outputs/figures/scatter_test.png`
- `outputs/figures/metrics_table.png`

## Reproduction

```bash
git clone https://github.com/The-AnonymousCoder/GVVI_GNN_Demo.git
cd GVVI_GNN_Demo
git lfs pull
pip install -r gvvi_grid_demo/requirements.txt
python gvvi_grid_demo/run_all.py --data-root . --device cuda
```
