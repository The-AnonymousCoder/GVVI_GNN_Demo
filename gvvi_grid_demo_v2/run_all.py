#!/usr/bin/env python3
"""GVVI Grid Demo V2 — end-to-end optimization pipeline."""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, torch, yaml
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="..")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "features", "train", "eval"])
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(demo_dir, "src")
    sys.path.insert(0, src_dir)

    for d in ["processed", "checkpoints", "outputs", "outputs/figures", "reports"]:
        os.makedirs(f"{demo_dir}/{d}", exist_ok=True)

    with open(f"{demo_dir}/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}, Data root: {data_root}")

    from audit_targets import audit_targets
    from build_grid_features_v2 import build_grid_features_v2
    from build_grid_graph import build_grid_graph
    from split_nodes import split_nodes
    from model import MultiScaleResSAGE
    from train import train_model
    from evaluate import evaluate
    from export_geotiff import export_geotiff

    t0 = time.time()

    # Load grid
    print("=" * 60); print("STEP 1: Load Grid Nodes"); print("=" * 60)
    grid_df = pd.read_csv(f"{data_root}/greenery/df_grid_ply.csv")
    if args.smoke:
        grid_df = grid_df.head(500).copy()
        print(f"SMOKE MODE: {len(grid_df)} nodes")
    grid_df["center_x"] = (grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2
    grid_df["center_y"] = (grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2
    print(f"Loaded {len(grid_df)} grid nodes\n")

    # Audit
    targets, raster_meta = audit_targets(data_root, grid_df)

    # Features V2
    features, feature_names = build_grid_features_v2(data_root, grid_df, cfg)

    # Standardize (fit on train only after split)
    # We'll do a pre-split here for scaling, then real split
    from sklearn.preprocessing import StandardScaler
    v1_split_path = f"{os.path.dirname(demo_dir)}/gvvi_grid_demo/processed/split_seed42.npz"
    v1 = np.load(v1_split_path)
    v1_train_idx = v1["train_idx"]
    scaler = StandardScaler()
    scaler.fit(features[v1_train_idx])
    features = scaler.transform(features).astype(np.float32)
    print(f"Features standardized: {features.shape[1]} dims (fit on V1 train nodes)\n")

    # Graphs
    ei_local, ei_context, n_nodes = build_grid_graph(grid_df, cfg, f"{demo_dir}/processed")

    # Split 80/5/15
    train_mask, val_mask, test_mask = split_nodes(targets, cfg, f"{demo_dir}/processed", v1_split_path)

    # Model
    if args.smoke:
        cfg["training"]["max_epochs"] = 3
        cfg["training"]["early_stopping_patience"] = 3
        cfg["model"]["hidden_dim"] = 64
        cfg["model"]["sage_dim1"] = 64
        cfg["model"]["sage_dim2"] = 32

    print("=" * 60); print("MODEL: Multi-Scale Residual GraphSAGE"); print("=" * 60)
    model = MultiScaleResSAGE(in_dim=features.shape[1],
                              hidden_dim=cfg["model"]["hidden_dim"],
                              sage_dim1=cfg["model"]["sage_dim1"],
                              sage_dim2=cfg["model"]["sage_dim2"],
                              dropout=cfg["model"]["dropout"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")

    # Train
    model, best_epoch = train_model(model, features, ei_local, ei_context,
                                     targets, train_mask, val_mask, cfg, device,
                                     f"{demo_dir}/checkpoints")

    # Evaluate
    metrics, pred_s, pred_w, pred_d, pred_gvvi = evaluate(
        model, features, ei_local, ei_context, targets, test_mask, device)

    # Export GeoTIFF
    export_geotiff(pred_s, pred_w, pred_d, pred_gvvi, grid_df, raster_meta, f"{demo_dir}/outputs")

    # Predictions CSV
    pred_df = pd.DataFrame({
        "grid_idx_x": grid_df["grid_idx_x"].values,
        "grid_idx_y": grid_df["grid_idx_y"].values,
        "pred_GVVI_S": pred_s, "pred_GVVI_W": pred_w, "pred_GVVI_D": pred_d, "pred_GVVI": pred_gvvi,
        "true_GVVI": targets["gvvi"],
        "split": np.where(test_mask, "test", np.where(val_mask, "val", "train")),
    })
    pred_df.to_csv(f"{demo_dir}/outputs/predictions.csv", index=False)

    with open(f"{demo_dir}/outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    t1 = time.time()
    print(f"\nDONE in {t1-t0:.1f}s ({ (t1-t0)/60:.1f} min)")

    # Figures
    _figures(targets, pred_gvvi, test_mask, grid_df, raster_meta, metrics,
             cfg, f"{demo_dir}/outputs/figures", data_root, best_epoch)

    # Report
    _report(metrics, cfg, device, t1-t0, best_epoch, f"{demo_dir}/reports/optimization_report.md",
            f"{demo_dir}/outputs/figures")

    # Copy baseline
    import shutil
    baseline_dir = f"{demo_dir}/baseline_v1"
    v1_metrics = f"{os.path.dirname(demo_dir)}/gvvi_grid_demo/outputs/metrics.json"
    if os.path.exists(v1_metrics):
        shutil.copy(v1_metrics, f"{baseline_dir}/metrics.json")
        shutil.copy(f"{os.path.dirname(demo_dir)}/gvvi_grid_demo/reports/final_demo_report.md",
                    f"{baseline_dir}/final_demo_report.md")
        print("Baseline V1 preserved.")


def _figures(targets, pred_gvvi, test_mask, grid_df, raster_meta, metrics, cfg, fig_dir, data_root, best_epoch):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    print("Generating figures...")

    # Official map
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(raster_meta["gvvi_official"], cmap="RdYlGn", origin="upper")
    ax.set_title("Official GVVI Map"); plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/official_gvvi_map.png", dpi=150, bbox_inches="tight"); plt.close()

    # V2 predicted map
    h, w = raster_meta["height"], raster_meta["width"]
    pred_map = np.zeros((h, w), dtype=np.float32)
    pred_map[grid_df["grid_idx_x"].values, grid_df["grid_idx_y"].values] = pred_gvvi
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pred_map, cmap="RdYlGn", origin="upper")
    ax.set_title("V2 Predicted GVVI Map"); plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/predicted_gvvi_map.png", dpi=150, bbox_inches="tight"); plt.close()

    # Error map (test nodes only)
    error_map = np.full((h, w), np.nan, dtype=np.float32)
    xs = grid_df["grid_idx_x"].values; ys = grid_df["grid_idx_y"].values
    t_mask = test_mask
    error_map[xs[t_mask], ys[t_mask]] = np.abs(pred_gvvi[t_mask] - targets["gvvi"][t_mask])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error_map, cmap="hot", origin="upper")
    ax.set_title("Absolute Error (Test Nodes)"); plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/abs_error_map.png", dpi=150, bbox_inches="tight"); plt.close()

    # Scatter (test)
    fig, ax = plt.subplots(figsize=(7, 7))
    yt = targets["gvvi"][t_mask]; yp = pred_gvvi[t_mask]
    ax.scatter(yt, yp, alpha=0.3, s=5)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("Official GVVI"); ax.set_ylabel("V2 Predicted GVVI")
    ax.set_title("Test Set: Predicted vs True")
    m = metrics["GVVI"]
    ax.text(0.05, 0.95, f"Spearman={m['Spearman']:.3f}, R²={m['R2']:.3f}, MAE={m['MAE']:.3f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(f"{fig_dir}/scatter_test.png", dpi=150, bbox_inches="tight"); plt.close()

    # Training curve placeholder (actual curve from log)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, f"Training completed\nBest epoch: {best_epoch}\nSee run_gpu.log for loss curves",
            transform=ax.transAxes, ha="center", fontsize=12)
    ax.set_title("Training Summary"); ax.axis("off")
    fig.savefig(f"{fig_dir}/training_summary.png", dpi=150, bbox_inches="tight"); plt.close()

    # Metrics table
    fig, ax = plt.subplots(figsize=(10, 4)); ax.axis("off")
    rows = []
    for name in ["GVVI_S", "GVVI_W", "GVVI_D", "GVVI"]:
        m = metrics[name]
        rows.append([name, f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['R2']:.4f}",
                     f"{m['Pearson']:.4f}", f"{m['Spearman']:.4f}"])
    table = ax.table(cellText=rows, colLabels=["Metric", "MAE", "RMSE", "R²", "Pearson", "Spearman"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 1.5)
    ax.set_title("Test Set Metrics (V2)", fontsize=12, fontweight="bold")
    fig.savefig(f"{fig_dir}/metrics_table.png", dpi=150, bbox_inches="tight"); plt.close()
    print("Figures saved.")


def _report(metrics, cfg, device, elapsed, best_epoch, report_path, fig_dir):
    m = metrics["GVVI"]
    reached = m["Spearman"] >= 0.80 and m["R2"] >= 0.50 and m["MAE"] <= 0.020
    report = f"""# GVVI Grid Demo V2 — Optimization Report

## Environment
- Device: {device}, Total time: {elapsed:.1f}s
- Best epoch: {best_epoch}/{cfg['training']['max_epochs']}

## Model
- Multi-Scale Residual GraphSAGE (256→256→128, {cfg['model']['sage_dim1']}→{cfg['model']['sage_dim2']})
- Dual heads: zero-classification + positive-regression
- Graphs: k={cfg['graph']['k_local']} (local) + k={cfg['graph']['k_context']} (context)
- Composite loss: BCE + SmoothL1 + GVVI MSE + PairwiseRanking

## Test Set Results

| Metric | GVVI_S | GVVI_W | GVVI_D | GVVI |
|--------|--------|--------|--------|------|
| MAE | {metrics['GVVI_S']['MAE']:.4f} | {metrics['GVVI_W']['MAE']:.4f} | {metrics['GVVI_D']['MAE']:.4f} | {m['MAE']:.4f} |
| RMSE | {metrics['GVVI_S']['RMSE']:.4f} | {metrics['GVVI_W']['RMSE']:.4f} | {metrics['GVVI_D']['RMSE']:.4f} | {m['RMSE']:.4f} |
| R² | {metrics['GVVI_S']['R2']:.4f} | {metrics['GVVI_W']['R2']:.4f} | {metrics['GVVI_D']['R2']:.4f} | {m['R2']:.4f} |
| Pearson | {metrics['GVVI_S']['Pearson']:.4f} | {metrics['GVVI_W']['Pearson']:.4f} | {metrics['GVVI_D']['Pearson']:.4f} | {m['Pearson']:.4f} |
| Spearman | {metrics['GVVI_S']['Spearman']:.4f} | {metrics['GVVI_W']['Spearman']:.4f} | {metrics['GVVI_D']['Spearman']:.4f} | {m['Spearman']:.4f} |

## Targets Check

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Spearman | ≥ 0.80 | {m['Spearman']:.4f} | {"PASS" if m['Spearman']>=0.80 else "FAIL"} |
| R² | ≥ 0.50 | {m['R2']:.4f} | {"PASS" if m['R2']>=0.50 else "FAIL"} |
| MAE | ≤ 0.020 | {m['MAE']:.4f} | {"PASS" if m['MAE']<=0.020 else "FAIL"} |

**Overall**: {"ALL STANDARDS MET" if reached else "Some standards not met"}

## Interpretation
- Labels: 80% train, 5% val, original 15% test (4,146 nodes frozen).
- "Study-area transductive GVVI imputation — no rendered pixels used."
- "No new data downloaded. All features from viewpoint position/heading/building/road/height tables."
"""
    with open(report_path, "w") as f:
        f.write(report)
    print("Report saved.")


if __name__ == "__main__":
    main()
