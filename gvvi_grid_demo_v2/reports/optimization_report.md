# GVVI Grid Demo V2 — Optimization Report

## Environment
- Device: cpu, Total time: 2.7s
- Best epoch: 3/3

## Model
- Multi-Scale Residual GraphSAGE (256→256→128, 64→32)
- Dual heads: zero-classification + positive-regression
- Graphs: k=8 (local) + k=24 (context)
- Composite loss: BCE + SmoothL1 + GVVI MSE + PairwiseRanking

## Test Set Results

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | 0.0682 | 0.0485 | 0.0909 | 0.0654 |
| RMSE | 0.0683 | 0.0486 | 0.0912 | 0.0655 |
| R² | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Pearson | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Spearman | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Targets Check

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Spearman | ≥ 0.80 | 0.0000 | FAIL |
| R² | ≥ 0.50 | 0.0000 | FAIL |
| MAE | ≤ 0.020 | 0.0654 | FAIL |

**Overall**: Some standards not met

## Interpretation
- Labels: 80% train, 5% val, original 15% test (4,146 nodes frozen).
- "Study-area transductive GVVI imputation — no rendered pixels used."
- "No new data downloaded. All features from viewpoint position/heading/building/road/height tables."
