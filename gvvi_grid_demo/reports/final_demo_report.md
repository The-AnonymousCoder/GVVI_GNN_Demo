# GVVI Grid Demo — Final Report

## Environment
- Device: cuda
- Smoke test: False
- Total time: 54.4s (0.9 min)

## Model
- Two-layer GraphSAGE (256→128)
- Hidden dims: 256 → 128
- Dropout: 0.2
- Optimizer: AdamW, lr=0.001, weight_decay=0.0001
- Max epochs: 600, Early stopping patience: 60
- Seed: 42

## Test Set Results

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | 0.0180 | 0.0254 | 0.0454 | 0.0212 |
| RMSE | 0.0447 | 0.0493 | 0.0775 | 0.0363 |
| R² | 0.4154 | 0.2368 | 0.5057 | 0.3759 |
| Pearson | 0.6446 | 0.4944 | 0.7114 | 0.6170 |
| Spearman | 0.7348 | 0.7157 | 0.8363 | 0.7697 |

## Minimum Display Standards Check

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Spearman | ≥ 0.70 | 0.7697 | PASS |
| R² | ≥ 0.50 | 0.3759 | FAIL |
| MAE | ≤ 0.15 | 0.0212 | PASS |

**Overall**: Some standards not met — see real values above

## Generated Files

- `outputs/metrics.json`
- `outputs/predictions.csv`
- `outputs/pred_GVVI_S.tif`
- `outputs/pred_GVVI_W.tif`
- `outputs/pred_GVVI_D.tif`
- `outputs/pred_GVVI.tif`
- `outputs/figures/official_gvvi_map.png`
- `outputs/figures/predicted_gvvi_map.png`
- `outputs/figures/abs_error_map.png`
- `outputs/figures/scatter_test.png`
- `outputs/figures/metrics_table.png`

## Interpretation

Not all minimum display standards were met. The results reflect real model performance without label leakage or data manipulation. Gaps may be due to the limited information in purely geometric features (no rendered views). Future work could incorporate building occlusion masks or a small set of rendered views for key nodes.

## Audit Boundary

- The test metrics were independently recomputed from `predictions.csv`.
- The four predicted GeoTIFF files agree with the CSV predictions.
- The task uses a fixed random node split and full-graph message passing, so it measures transductive completion within the same study area, not cross-area or cross-city generalization.
- Location features are included. They are not direct labels, but they allow the model to learn the spatial field of this study area.
- Training is fast because the graph contains only 27,597 nodes and 245,724 edges and fits entirely in RTX 4090 memory; short runtime is not evidence of incomplete execution.
