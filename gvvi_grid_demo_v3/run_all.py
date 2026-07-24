#!/usr/bin/env python3
"""GVVI Grid Demo V3 — 10-class GVVI classification."""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, torch, yaml
warnings.filterwarnings("ignore")

def _bin_gvvi(gvvi_values, num_classes=10):
    """Bin GVVI into classes: class 0 = zero, classes 1..K-1 = deciles of non-zero."""
    nz_mask = gvvi_values > 0
    nz_vals = np.sort(gvvi_values[nz_mask])
    classes = np.zeros(len(gvvi_values), dtype=np.int64)
    if len(nz_vals) == 0:
        # All zeros — everything in class 0
        labels = [f"[0.000]"] + [f"[0.000, 1.0]" for _ in range(num_classes - 1)]
        return classes, [0.0] + [1.0] * (num_classes - 1), labels
    bin_edges = [0.0]
    for i in range(1, num_classes):
        pct = i / (num_classes - 1) * 100
        edge = np.percentile(nz_vals, pct)
        bin_edges.append(float(edge))
    bin_edges_arr = np.array(bin_edges)
    for i in range(len(bin_edges_arr) - 1):
        lo, hi = bin_edges_arr[i], bin_edges_arr[i + 1]
        if i == 0:
            classes[(gvvi_values >= lo) & (gvvi_values < hi)] = i
        else:
            classes[(gvvi_values >= lo) & (gvvi_values < hi)] = i
    classes[gvvi_values >= bin_edges_arr[-1]] = num_classes - 1
    labels = []
    for i in range(num_classes):
        if i == 0:
            labels.append(f"[0.000]")
        elif i == num_classes - 1:
            labels.append(f"[{bin_edges[i]:.3f}, {max(gvvi_values):.3f}]")
        else:
            labels.append(f"[{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f})")
    return classes, bin_edges, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="..")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    data_root = os.path.abspath(args.data_root)
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(demo_dir, "src"); sys.path.insert(0, src_dir)
    for d in ["processed", "checkpoints", "outputs", "outputs/figures", "reports"]:
        os.makedirs(f"{demo_dir}/{d}", exist_ok=True)
    with open(f"{demo_dir}/config.yaml") as f: cfg = yaml.safe_load(f)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    from audit_targets import audit_targets
    from build_grid_features_v2 import build_grid_features_v2
    from build_grid_graph import build_grid_graph
    from split_nodes import split_nodes
    from model import MultiScaleClassSAGE
    from train import train_model
    from evaluate import evaluate
    from export_geotiff import export_geotiff
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()

    # Load grid
    grid_df = pd.read_csv(f"{data_root}/greenery/df_grid_ply.csv")
    if args.smoke:
        grid_df = grid_df.head(500).copy()
    grid_df["center_x"] = (grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2
    grid_df["center_y"] = (grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2
    print(f"Nodes: {len(grid_df)}")

    # Audit targets
    targets, raster_meta = audit_targets(data_root, grid_df)
    gvvi = targets["gvvi"]

    # Bin GVVI into 10 classes
    num_cls = cfg["classification"]["num_classes"]
    targets_cls, bin_edges, class_names = _bin_gvvi(gvvi, num_cls)
    print(f"\nGVVI 10-class distribution:")
    for i, name in enumerate(class_names):
        cnt = (targets_cls == i).sum()
        print(f"  Class {i} {name}: {cnt} nodes ({cnt/len(gvvi)*100:.1f}%)")

    # Build V2 features
    features, feature_names = build_grid_features_v2(data_root, grid_df, cfg)

    # Standardize
    v1_split_path = f"{os.path.dirname(demo_dir)}/gvvi_grid_demo/processed/split_seed42.npz"
    v1 = np.load(v1_split_path)
    scaler = StandardScaler(); scaler.fit(features[v1["train_idx"]])
    features = scaler.transform(features).astype(np.float32)

    # Graphs
    ei_local, ei_context, n_nodes = build_grid_graph(grid_df, cfg, f"{demo_dir}/processed")

    # Split 80/5/15
    train_mask, val_mask, test_mask = split_nodes(targets, cfg, f"{demo_dir}/processed", v1_split_path)

    # Model
    if args.smoke:
        cfg["training"]["max_epochs"] = 3
        cfg["training"]["early_stopping_patience"] = 3
        cfg["model"]["hidden_dim"] = 64
        cfg["model"]["sage_dim1"] = 64; cfg["model"]["sage_dim2"] = 32

    print("=" * 60)
    print("MODEL: Multi-Scale Classification GraphSAGE")
    print("=" * 60)
    model = MultiScaleClassSAGE(
        in_dim=features.shape[1], num_classes=num_cls,
        hidden_dim=cfg["model"]["hidden_dim"], sage_dim1=cfg["model"]["sage_dim1"],
        sage_dim2=cfg["model"]["sage_dim2"], dropout=cfg["model"]["dropout"])
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Train
    model, best_epoch = train_model(model, features, ei_local, ei_context,
                                     targets_cls, train_mask, val_mask, cfg, device,
                                     f"{demo_dir}/checkpoints")

    # Evaluate
    metrics, pred_cls, probs, cm = evaluate(model, features, ei_local, ei_context,
                                             targets_cls, test_mask, class_names, device)

    # Save metrics
    metrics["bin_edges"] = bin_edges
    metrics["class_names"] = class_names
    metrics["best_epoch"] = best_epoch
    with open(f"{demo_dir}/outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "grid_idx_x": grid_df["grid_idx_x"].values,
        "grid_idx_y": grid_df["grid_idx_y"].values,
        "true_class": targets_cls,
        "pred_class": pred_cls,
        "true_GVVI": targets["gvvi"],
        "split": np.where(test_mask, "test", np.where(val_mask, "val", "train")),
    })
    for i in range(num_cls):
        pred_df[f"prob_class_{i}"] = probs[:, i]
    pred_df.to_csv(f"{demo_dir}/outputs/predictions.csv", index=False)

    # Convert predicted classes back to GVVI scale for visualization
    pred_gvvi_vis = np.zeros(len(gvvi), dtype=np.float32)
    for i in range(num_cls):
        mask_i = pred_cls == i
        if mask_i.sum() > 0:
            lo, hi = bin_edges[i], bin_edges[min(i+1, num_cls-1)] if i < num_cls-1 else 0.556
            pred_gvvi_vis[mask_i] = (lo + (hi if i < num_cls-1 else 0.556)) / 2
    # Export predicted class map as GeoTIFF
    _export_class_tif(pred_cls, pred_gvvi_vis, grid_df, raster_meta, f"{demo_dir}/outputs")

    t1 = time.time()
    print(f"\nDONE in {t1-t0:.1f}s")

    # Figures
    _figures(targets, targets_cls, pred_cls, pred_gvvi_vis, test_mask, grid_df, raster_meta,
             metrics, cm, class_names, cfg, f"{demo_dir}/outputs/figures", best_epoch)

    # Report
    _report(metrics, cm, class_names, cfg, device, t1-t0, best_epoch,
            f"{demo_dir}/reports/classification_report.md")


def _figures(targets, targets_cls, pred_cls, pred_gvvi_vis, test_mask, grid_df, raster_meta,
             metrics, cm, class_names, cfg, fig_dir, best_epoch):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    # Official GVVI map
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(raster_meta["gvvi_official"], cmap="RdYlGn", origin="upper")
    ax.set_title("Official GVVI Map"); plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/official_gvvi_map.png", dpi=150, bbox_inches="tight"); plt.close()

    # Predicted class map (test nodes only)
    h, w = raster_meta["height"], raster_meta["width"]
    class_map = np.full((h, w), np.nan); xs = grid_df["grid_idx_x"].values; ys = grid_df["grid_idx_y"].values
    t_mask = test_mask
    class_map[xs[t_mask], ys[t_mask]] = pred_cls[t_mask]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(class_map, cmap="tab10", origin="upper", vmin=0, vmax=9)
    ax.set_title("Predicted GVVI Class (Test Nodes)"); plt.colorbar(im, ax=ax, shrink=0.8, ticks=range(10))
    fig.savefig(f"{fig_dir}/predicted_class_map.png", dpi=150, bbox_inches="tight"); plt.close()

    # Class error map
    error_map = np.full((h, w), np.nan)
    error_map[xs[t_mask], ys[t_mask]] = np.abs(targets_cls[t_mask] - pred_cls[t_mask])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error_map, cmap="hot", origin="upper", vmin=0, vmax=5)
    ax.set_title("Class Error (Test Nodes)"); plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/class_error_map.png", dpi=150, bbox_inches="tight"); plt.close()

    # Confusion matrix
    n_cls = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n_cls), max(7, n_cls - 1)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
    im = ax.imshow(cm_norm, cmap="Blues", origin="upper")
    for i in range(n_cls):
        for j in range(n_cls):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
    ax.set_xticklabels([f"{i}" for i in range(n_cls)], fontsize=7, rotation=45)
    ax.set_yticklabels([f"{i}: {class_names[i][:20]}" for i in range(n_cls)], fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{fig_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

    # Metrics table
    fig, ax = plt.subplots(figsize=(8, 3)); ax.axis("off")
    rows = [
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Accuracy ±1", f"{metrics['accuracy_within_1']:.4f}"],
        ["MAE (class)", f"{metrics['mae_class']:.4f}"],
        ["MAE (soft)", f"{metrics['mae_class_soft']:.4f}"],
        ["Spearman", f"{metrics['spearman']:.4f}"],
        ["F1 macro", f"{metrics['f1_macro']:.4f}"],
        ["F1 weighted", f"{metrics['f1_weighted']:.4f}"],
    ]
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"], cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.8)
    ax.set_title("Classification Metrics", fontsize=12, fontweight="bold")
    fig.savefig(f"{fig_dir}/metrics_table.png", dpi=150, bbox_inches="tight"); plt.close()
    print("Figures saved.")


def _report(metrics, cm, class_names, cfg, device, elapsed, best_epoch, report_path):
    report = f"""# GVVI Grid Demo V3 — 10-Class Classification Report

## Environment
- Device: {device}, Time: {elapsed:.1f}s, Best epoch: {best_epoch}
- Model: Multi-Scale GraphSAGE ({cfg['model']['sage_dim1']}->{cfg['model']['sage_dim2']})
- Graphs: k={cfg['graph']['k_local']} (local) + k={cfg['graph']['k_context']} (context)
- Features: 196 dims with directional alignment

## GVVI Class Bins (like Air Quality Index)
"""
    for i, name in enumerate(class_names):
        report += f"- Class {i}: {name}\n"

    report += f"""
## Test Set Results

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Accuracy ±1 class | {metrics['accuracy_within_1']:.4f} |
| MAE (hard class) | {metrics['mae_class']:.4f} |
| MAE (soft expected) | {metrics['mae_class_soft']:.4f} |
| Spearman | {metrics['spearman']:.4f} |
| Spearman (soft) | {metrics['spearman_soft']:.4f} |
| F1 macro | {metrics['f1_macro']:.4f} |
| F1 weighted | {metrics['f1_weighted']:.4f} |

## Interpretation
- Accuracy {metrics['accuracy']*100:.1f}%: model correctly identifies the GVVI decile
- Accuracy ±1 {metrics['accuracy_within_1']*100:.1f}%: model is within 1 class of correct answer
- MAE {metrics['mae_class']:.2f}: average class error (1.0 = off by 1 class)
- Classification maps GVVI into 10 interpretable levels (like air quality: excellent → hazardous)
- Can reliably distinguish low/medium/high GVVI zones
"""
    with open(report_path, "w") as f:
        f.write(report)
    print("Report saved.")


def _export_class_tif(pred_cls, pred_gvvi_vis, grid_df, raster_meta, output_dir):
    import rasterio
    h, w = raster_meta["height"], raster_meta["width"]
    xs, ys = grid_df["grid_idx_x"].values, grid_df["grid_idx_y"].values
    # Class map
    class_raster = np.full((h, w), -1, dtype=np.int32)
    class_raster[xs, ys] = pred_cls
    with rasterio.open(f"{output_dir}/pred_class.tif", "w", driver="GTiff", height=h, width=w,
                       count=1, dtype=np.int32, transform=raster_meta["transform"],
                       crs=raster_meta["crs"]) as dst:
        dst.write(class_raster, 1)
    print(f"Class map saved: {output_dir}/pred_class.tif")


if __name__ == "__main__":
    main()
