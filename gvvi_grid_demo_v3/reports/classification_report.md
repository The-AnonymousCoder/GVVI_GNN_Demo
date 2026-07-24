# GVVI Grid Demo V3 — 10-Class Classification Report

## Approach

Convert GVVI regression into a 10-class classification problem, similar to air quality index (AQI) levels. Each greenery grid is assigned to one of 10 GVVI classes, ranging from "no visibility" (class 0) to "very high visibility" (class 9).

## GVVI Class Bins (Decile-Based on Non-Zero GVVI)

| Class | GVVI Range | Description |
|-------|-----------|-------------|
| 0 | [0.000, 0.000] | No visibility |
| 1 | [0.000, 0.008) | Trace |
| 2 | [0.008, 0.015) | Very low |
| 3 | [0.015, 0.024) | Low |
| 4 | [0.024, 0.033) | Low-medium |
| 5 | [0.033, 0.046) | Medium |
| 6 | [0.046, 0.063) | Medium-high |
| 7 | [0.063, 0.094) | High |
| 8 | [0.094, 0.556) | Very high |
| 9 | [0.556, 0.556] | Max |

## Environment
- Device: NVIDIA GeForce RTX 4090 (24 GB)
- Model: Multi-Scale GraphSAGE (256→256→128, 176K params)
- Features: 196 dims with directional alignment
- Graphs: k=8 (local) + k=24 (context)
- Loss: CrossEntropy + ordinal MSE
- Best epoch: 83 (early stop)
- Time: 228s (3.8 min)

## Test Set Results (4,146 nodes)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | **43.37%** | 4.3x random baseline (10%) |
| Accuracy ±1 class | **64.52%** | 2/3 of predictions within 1 class |
| MAE (hard class) | 1.50 | Average 1.5 classes off |
| MAE (soft expected) | 1.33 | Using probability-weighted class |
| Spearman | **0.738** | Rank correlation (class-level) |
| Spearman (soft) | **0.766** | Using probability-weighted rank |
| F1 weighted | 0.426 | Class-imbalance-aware F1 |

## Error Distribution

| Error | Percentage | Cumulative |
|-------|-----------|------------|
| Exact (0) | 43.4% | 43.4% |
| Off-by-1 | 21.2% | 64.5% |
| Off-by-2 | 10.0% | 74.5% |
| Off-by-3+ | 25.5% | 100% |

## Key Insights

1. **43% accuracy is 4.3x random**: random guessing would achieve 10% (1/10 classes). The model captures substantial structure.
2. **65% within ±1 class**: most errors are near the true class, not random — the model understands the ordinal nature of GVVI.
3. **Spearman 0.766 (soft)**: comparable to the regression model's Spearman (0.770), confirming the classification approach doesn't lose ranking information.
4. **Large errors (3+ classes off) are 25.5%**: these are likely nodes where geometric features are insufficient (e.g., building-occluded views).

## Comparison with Regression (V1/V2)

| Approach | Spearman | Key Metric |
|----------|----------|------------|
| V1 Regression | 0.770 | R² = 0.376 |
| V2 Regression | 0.773 | R² = 0.341 |
| **V3 Classification** | **0.766 (soft)** | **Acc = 43.4%** |

The classification approach achieves comparable ranking performance while providing **interpretable class labels** — instead of predicting "GVVI = 0.0423", the model outputs "Class 5 (medium visibility)".

## Interpretation

The V3 classification model can be used for:
- **Rapid screening**: "Which areas have high GVVI (class 7+) vs low (class 0-3)?"
- **Zoning decisions**: "Classify neighborhoods into 3-5 visibility tiers for planning"
- **Priority allocation**: "Compute detailed GVVI only for class 6+ nodes"

The model benefits from:
- 80% training labels, original 15% frozen test set
- Directional features capturing camera-greenery alignment
- Multi-scale graph structure (local k=8, context k=24)
- Ordinal-aware loss respecting class ordering
