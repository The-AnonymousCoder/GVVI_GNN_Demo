"""Training loop with early stopping."""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader


def train_model(model, x, edge_index, targets, train_mask, val_mask, cfg, device, checkpoint_dir):
    """Train GraphSAGE and save best checkpoint."""
    print("=" * 60)
    print("TRAIN: GraphSAGE GVVI Regression")
    print("=" * 60)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    max_epochs = cfg["training"]["max_epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    composite_loss_weight = cfg["training"].get("composite_loss_weight", 0.0)

    w_s = 3.0 / 6.0
    w_w = 2.0 / 6.0
    w_d = 1.0 / 6.0

    # Prepare targets
    y_s = torch.tensor(targets["gvvi_s"], dtype=torch.float32).to(device)
    y_w = torch.tensor(targets["gvvi_w"], dtype=torch.float32).to(device)
    y_d = torch.tensor(targets["gvvi_d"], dtype=torch.float32).to(device)
    y = torch.stack([y_s, y_w, y_d], dim=1)

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)

    t_mask = torch.tensor(train_mask, dtype=torch.bool).to(device)
    v_mask = torch.tensor(val_mask, dtype=torch.bool).to(device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6)

    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    ckpt_path = f"{checkpoint_dir}/best_model.pt"

    n_nodes = x_tensor.shape[0]

    print(f"Device: {device}")
    print(f"Nodes: {n_nodes}, Features: {x_tensor.shape[1]}")
    print(f"Train nodes: {t_mask.sum().item()}, Val nodes: {v_mask.sum().item()}")
    print(f"Max epochs: {max_epochs}, Patience: {patience}")

    # Save initial checkpoint
    torch.save(model.state_dict(), ckpt_path)

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(x_tensor, edge_index)
        loss_s = loss_fn(pred[t_mask, 0], y[t_mask, 0])
        loss_w = loss_fn(pred[t_mask, 1], y[t_mask, 1])
        loss_d = loss_fn(pred[t_mask, 2], y[t_mask, 2])
        pred_gvvi = w_s * pred[t_mask, 0] + w_w * pred[t_mask, 1] + w_d * pred[t_mask, 2]
        true_gvvi = w_s * y[t_mask, 0] + w_w * y[t_mask, 1] + w_d * y[t_mask, 2]
        loss_gvvi = loss_fn(pred_gvvi, true_gvvi)
        loss = (
            w_s * loss_s + w_w * loss_w + w_d * loss_d
            + composite_loss_weight * loss_gvvi
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Epoch {epoch:3d}: NaN/Inf train loss detected, skipping...")
            patience_counter += 1
            if patience_counter >= patience:
                break
            continue

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_tensor, edge_index)
            v_loss_s = loss_fn(val_pred[v_mask, 0], y[v_mask, 0])
            v_loss_w = loss_fn(val_pred[v_mask, 1], y[v_mask, 1])
            v_loss_d = loss_fn(val_pred[v_mask, 2], y[v_mask, 2])
            val_pred_gvvi = (
                w_s * val_pred[v_mask, 0]
                + w_w * val_pred[v_mask, 1]
                + w_d * val_pred[v_mask, 2]
            )
            val_true_gvvi = (
                w_s * y[v_mask, 0]
                + w_w * y[v_mask, 1]
                + w_d * y[v_mask, 2]
            )
            v_loss_gvvi = loss_fn(val_pred_gvvi, val_true_gvvi)
            val_loss = (
                w_s * v_loss_s + w_w * v_loss_w + w_d * v_loss_d
                + composite_loss_weight * v_loss_gvvi
            )

        if torch.isnan(val_loss) or torch.isinf(val_loss):
            print(f"Epoch {epoch:3d}: NaN/Inf val loss detected, skipping...")
            patience_counter += 1
            if patience_counter >= patience:
                break
            continue

        scheduler.step(val_loss)

        if epoch <= 5 or epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: train_loss={loss.item():.6f}, val_loss={val_loss.item():.6f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best: {best_epoch}, val_loss={best_val_loss:.6f})")
            break

    if epoch == max_epochs:
        print(f"Reached max epochs {max_epochs} (best: {best_epoch}, val_loss={best_val_loss:.6f})")

    model.load_state_dict(torch.load(ckpt_path))
    print(f"Best model loaded from epoch {best_epoch}\n")
    return model
