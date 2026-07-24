"""Evaluate on test set and compute metrics."""
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate(model, x, edge_index, targets, test_mask, device):
    """Evaluate on test nodes and return metrics."""
    print("=" * 60)
    print("EVALUATE: Test Set Metrics")
    print("=" * 60)

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(x_tensor, edge_index).cpu().numpy()

    y_s = targets["gvvi_s"]
    y_w = targets["gvvi_w"]
    y_d = targets["gvvi_d"]
    y_gvvi = targets["gvvi"]

    # Compute predicted GVVI
    pred_s = pred[:, 0]
    pred_w = pred[:, 1]
    pred_d = pred[:, 2]
    pred_gvvi = 3.0/6.0 * pred_s + 2.0/6.0 * pred_w + 1.0/6.0 * pred_d

    t_mask = test_mask
    metrics = {}

    for name, y_true, y_pred in [
        ("GVVI_S", y_s, pred_s),
        ("GVVI_W", y_w, pred_w),
        ("GVVI_D", y_d, pred_d),
        ("GVVI", y_gvvi, pred_gvvi),
    ]:
        yt = y_true[t_mask]
        yp = y_pred[t_mask]
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        # Only compute correlation if there's variance
        if yt.std() > 1e-10:
            pearson, _ = pearsonr(yt, yp)
            spearman, _ = spearmanr(yt, yp)
        else:
            pearson, spearman = 0.0, 0.0
        metrics[name] = {
            "MAE": round(float(mae), 6),
            "RMSE": round(float(rmse), 6),
            "R2": round(float(r2), 6),
            "Pearson": round(float(pearson), 6),
            "Spearman": round(float(spearman), 6),
        }
        print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Pearson={pearson:.4f}, Spearman={spearman:.4f}")

    return metrics, pred_s, pred_w, pred_d, pred_gvvi
