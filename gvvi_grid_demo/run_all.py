#!/usr/bin/env python3
"""GVVI Grid Demo — single end-to-end pipeline."""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import yaml

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="GVVI Grid Demo")
    parser.add_argument("--data-root", type=str, default="..", help="Path to repo root")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 500 nodes, 3 epochs")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(demo_dir, "src")
    sys.path.insert(0, src_dir)

    os.makedirs(f"{demo_dir}/processed", exist_ok=True)
    os.makedirs(f"{demo_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{demo_dir}/outputs/figures", exist_ok=True)
    os.makedirs(f"{demo_dir}/reports", exist_ok=True)

    # Load config
    with open(f"{demo_dir}/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Data root: {data_root}")
    print(f"Demo dir: {demo_dir}")

    # Import local modules
    from audit_targets import audit_targets
    from build_grid_features import build_grid_features
    from build_grid_graph import build_grid_graph
    from split_nodes import split_nodes
    from model import GVVIGraphSAGE
    from train import train_model
    from evaluate import evaluate
    from export_geotiff import export_geotiff

    t0 = time.time()

    # === Step 1: Load grid nodes ===
    print("=" * 60)
    print("STEP 1: Load Grid Nodes")
    print("=" * 60)
    grid_df = pd.read_csv(f"{data_root}/greenery/df_grid_ply.csv")
    if args.smoke:
        grid_df = grid_df.head(500).copy()
        print(f"SMOKE MODE: using {len(grid_df)} nodes, 3 epochs")

    grid_df["center_x"] = (grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2
    grid_df["center_y"] = (grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2
    print(f"Loaded {len(grid_df)} grid nodes\n")

    # === Step 2: Audit targets (official replay check) ===
    targets, raster_meta = audit_targets(data_root, grid_df)

    # === Step 3: Build features ===
    X, feature_names = build_grid_features(data_root, grid_df, cfg)

    # === Step 4: Build graph ===
    edge_index, edge_attr, n_nodes = build_grid_graph(
        grid_df, cfg, f"{demo_dir}/processed"
    )

    # === Step 5: Split ===
    train_mask, val_mask, test_mask = split_nodes(
        targets, cfg, f"{demo_dir}/processed"
    )

    # === Step 6: Create model ===
    print("=" * 60)
    print("MODEL: Two-layer GraphSAGE")
    print("=" * 60)
    model = GVVIGraphSAGE(
        in_dim=X.shape[1],
        hidden_dim1=cfg["model"]["hidden_dim1"],
        hidden_dim2=cfg["model"]["hidden_dim2"],
        dropout=cfg["model"]["dropout"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    # Override epochs for smoke test
    if args.smoke:
        cfg["training"]["max_epochs"] = 3
        cfg["training"]["early_stopping_patience"] = 3

    # === Step 7: Train ===
    model = train_model(
        model, X, edge_index, targets, train_mask, val_mask,
        cfg, device, f"{demo_dir}/checkpoints"
    )

    # === Step 8: Evaluate ===
    metrics, pred_s, pred_w, pred_d, pred_gvvi = evaluate(
        model, X, edge_index, targets, test_mask, device
    )

    # === Step 9: Export GeoTIFF ===
    export_geotiff(pred_s, pred_w, pred_d, pred_gvvi, grid_df, raster_meta,
                   f"{demo_dir}/outputs")

    # === Step 10: Save predictions CSV ===
    pred_df = pd.DataFrame({
        "grid_idx_x": grid_df["grid_idx_x"].values,
        "grid_idx_y": grid_df["grid_idx_y"].values,
        "pred_GVVI_S": pred_s,
        "pred_GVVI_W": pred_w,
        "pred_GVVI_D": pred_d,
        "pred_GVVI": pred_gvvi,
        "true_GVVI": targets["gvvi"],
        "split": np.where(test_mask, "test", np.where(val_mask, "val", "train")),
    })
    pred_df.to_csv(f"{demo_dir}/outputs/predictions.csv", index=False)
    print("Predictions CSV saved.\n")

    # === Step 11: Save metrics JSON ===
    with open(f"{demo_dir}/outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics JSON saved.\n")

    # === Step 12: Generate figures ===
    _generate_figures(
        targets, pred_gvvi, pred_s, pred_w, pred_d,
        test_mask, grid_df, raster_meta, metrics, cfg,
        f"{demo_dir}/outputs/figures", data_root, device, args.smoke
    )

    t1 = time.time()
    print("=" * 60)
    print(f"DONE in {t1 - t0:.1f}s ({ (t1-t0)/60:.1f} min)")
    print("=" * 60)

    # === Step 13: Generate report ===
    _generate_report(
        metrics, cfg, device, t1 - t0, args.smoke,
        f"{demo_dir}/reports/final_demo_report.md",
        f"{demo_dir}/outputs/figures", data_root
    )


def _generate_figures(targets, pred_gvvi, pred_s, pred_w, pred_d, test_mask,
                      grid_df, raster_meta, metrics, cfg, fig_dir, data_root, device, smoke):
    """Generate all required figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    print("=" * 60)
    print("FIGURES: Generating maps and plots")
    print("=" * 60)

    # Figure 1: Official GVVI map
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(raster_meta["gvvi_official"], cmap="RdYlGn", origin="upper")
    ax.set_title("Official GVVI Map")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/official_gvvi_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  official_gvvi_map.png")

    # Figure 2: GNN predicted GVVI map
    height, width = raster_meta["height"], raster_meta["width"]
    pred_map = np.zeros((height, width), dtype=np.float32)
    pred_map[grid_df["grid_idx_x"].values, grid_df["grid_idx_y"].values] = pred_gvvi

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pred_map, cmap="RdYlGn", origin="upper")
    ax.set_title("GNN Predicted GVVI Map")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/predicted_gvvi_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  predicted_gvvi_map.png")

    # Figure 3: Absolute error map
    error_map = np.full((height, width), np.nan, dtype=np.float32)
    xs = grid_df["grid_idx_x"].values
    ys = grid_df["grid_idx_y"].values
    t_mask = test_mask
    error_map[xs[t_mask], ys[t_mask]] = np.abs(pred_gvvi[t_mask] - targets["gvvi"][t_mask])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error_map, cmap="hot", origin="upper")
    ax.set_title("Absolute Error Map (Test Nodes)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/abs_error_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  abs_error_map.png")

    # Figure 4: Scatter plot (test set)
    fig, ax = plt.subplots(figsize=(7, 7))
    yt = targets["gvvi"][t_mask]
    yp = pred_gvvi[t_mask]
    ax.scatter(yt, yp, alpha=0.3, s=5)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("Official GVVI")
    ax.set_ylabel("GNN Predicted GVVI")
    ax.set_title("Test Set: Predicted vs True GVVI")
    # Add metrics text
    m = metrics["GVVI"]
    text = f"Spearman={m['Spearman']:.3f}, R²={m['R2']:.3f}, MAE={m['MAE']:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(f"{fig_dir}/scatter_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  scatter_test.png")

    # Figure 5: Metrics table as an image
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    rows = []
    for name in ["GVVI_S", "GVVI_W", "GVVI_D", "GVVI"]:
        m = metrics[name]
        rows.append([name, f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['R2']:.4f}",
                     f"{m['Pearson']:.4f}", f"{m['Spearman']:.4f}"])
    table = ax.table(cellText=rows,
                     colLabels=["Metric", "MAE", "RMSE", "R²", "Pearson", "Spearman"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title("Test Set Metrics", fontsize=12, fontweight="bold")
    fig.savefig(f"{fig_dir}/metrics_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  metrics_table.png")

    print("Figures saved.\n")


def _generate_report(metrics, cfg, device, elapsed, smoke, report_path, fig_dir, data_root):
    """Generate final demo report."""
    m = metrics["GVVI"]
    reached = (
        m["Spearman"] >= 0.70 and
        m["R2"] >= 0.50 and
        m["MAE"] <= 0.15
    )

    report = f"""# GVVI Grid Demo — Final Report

## Environment
- Device: {device}
- Smoke test: {smoke}
- Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)

## Model
- Two-layer GraphSAGE (128→64)
- Hidden dims: {cfg['model']['hidden_dim1']} → {cfg['model']['hidden_dim2']}
- Dropout: {cfg['model']['dropout']}
- Optimizer: AdamW, lr={cfg['training']['lr']}, weight_decay={cfg['training']['weight_decay']}
- Max epochs: {cfg['training']['max_epochs']}, Early stopping patience: {cfg['training']['early_stopping_patience']}
- Seed: {cfg['training']['seed']}

## Test Set Results

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | {metrics['GVVI_S']['MAE']:.4f} | {metrics['GVVI_W']['MAE']:.4f} | {metrics['GVVI_D']['MAE']:.4f} | {metrics['GVVI']['MAE']:.4f} |
| RMSE | {metrics['GVVI_S']['RMSE']:.4f} | {metrics['GVVI_W']['RMSE']:.4f} | {metrics['GVVI_D']['RMSE']:.4f} | {metrics['GVVI']['RMSE']:.4f} |
| R² | {metrics['GVVI_S']['R2']:.4f} | {metrics['GVVI_W']['R2']:.4f} | {metrics['GVVI_D']['R2']:.4f} | {metrics['GVVI']['R2']:.4f} |
| Pearson | {metrics['GVVI_S']['Pearson']:.4f} | {metrics['GVVI_W']['Pearson']:.4f} | {metrics['GVVI_D']['Pearson']:.4f} | {metrics['GVVI']['Pearson']:.4f} |
| Spearman | {metrics['GVVI_S']['Spearman']:.4f} | {metrics['GVVI_W']['Spearman']:.4f} | {metrics['GVVI_D']['Spearman']:.4f} | {metrics['GVVI']['Spearman']:.4f} |

## Minimum Display Standards Check

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Spearman | ≥ 0.70 | {m['Spearman']:.4f} | {"PASS" if m['Spearman'] >= 0.70 else "FAIL"} |
| R² | ≥ 0.50 | {m['R2']:.4f} | {"PASS" if m['R2'] >= 0.50 else "FAIL"} |
| MAE | ≤ 0.15 | {m['MAE']:.4f} | {"PASS" if m['MAE'] <= 0.15 else "FAIL"} |

**Overall**: {"ALL STANDARDS MET" if reached else "Some standards not met — see real values above"}

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

{f"All three minimum display standards were met (Spearman ≥0.70, R² ≥0.50, MAE ≤0.15). This demonstrates preliminary feasibility of using GraphSAGE with cheap spatial features to rapidly complete multi-perspective GVVI within the study area." if reached else "Not all minimum display standards were met. The results reflect real model performance without label leakage or data manipulation. Gaps may be due to the limited information in purely geometric features (no rendered views). Future work could incorporate building occlusion masks or a small set of rendered views for key nodes."}
"""
    with open(report_path, "w") as f:
        f.write(report)
    print("Report saved.\n")


if __name__ == "__main__":
    main()
