"""Evaluate on test set and compute metrics."""
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate(model, features, edge_index_local, edge_index_context,
             targets, test_mask, device):
    print("=" * 60)
    print("EVALUATE: Test Set Metrics")
    print("=" * 60)

    x_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    ei_local = edge_index_local.to(device)
    ei_context = edge_index_context.to(device) if edge_index_context is not None else None
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        preds, p_nonzeros, p_values = model(x_tensor, ei_local, ei_context)
        preds = preds.cpu().numpy()

    y_s = targets["gvvi_s"]; y_w = targets["gvvi_w"]; y_d = targets["gvvi_d"]; y_gvvi = targets["gvvi"]

    pred_s = preds[:, 0]; pred_w = preds[:, 1]; pred_d = preds[:, 2]
    pred_gvvi = 3.0/6.0 * pred_s + 2.0/6.0 * pred_w + 1.0/6.0 * pred_d

    t_mask = test_mask
    metrics = {}
    for name, y_true, y_pred in [("GVVI_S", y_s, pred_s), ("GVVI_W", y_w, pred_w),
                                  ("GVVI_D", y_d, pred_d), ("GVVI", y_gvvi, pred_gvvi)]:
        yt = y_true[t_mask]; yp = y_pred[t_mask]
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        pearson, _ = (pearsonr(yt, yp) if yt.std() > 1e-10 else (0.0, 0.0))
        spearman, _ = (spearmanr(yt, yp) if yt.std() > 1e-10 else (0.0, 0.0))
        metrics[name] = {"MAE": round(float(mae), 6), "RMSE": round(float(rmse), 6),
                         "R2": round(float(r2), 6), "Pearson": round(float(pearson), 6),
                         "Spearman": round(float(spearman), 6)}
        print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, Pearson={pearson:.4f}, Spearman={spearman:.4f}")
    return metrics, pred_s, pred_w, pred_d, pred_gvvi
