# GVVI Grid Demo — Final Report

## Environment
- Device: cpu
- Smoke test: True
- Total time: 1.6s (0.0 min)

## Model
- Two-layer GraphSAGE (128→64)
- Hidden dims: 128 → 64
- Dropout: 0.2
- Optimizer: AdamW, lr=0.001, weight_decay=0.0001
- Max epochs: 3, Early stopping patience: 3
- Seed: 42

## Test Set Results

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | 0.4686 | 0.4665 | 0.4425 | 0.4635 |
| RMSE | 0.4686 | 0.4665 | 0.4425 | 0.4635 |
| R² | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Pearson | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Spearman | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Minimum Display Standards Check

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Spearman | ≥ 0.70 | 0.0000 | FAIL |
| R² | ≥ 0.50 | 0.0000 | FAIL |
| MAE | ≤ 0.15 | 0.4635 | FAIL |

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
