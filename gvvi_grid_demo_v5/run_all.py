"""V5: Heterogeneous graph — grid nodes + viewpoint nodes."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, time, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
from torch_geometric.nn import SAGEConv
import rasterio

# ====== Model: Viewpoint aggregation + Grid SAGE ======
class HeteroSAGE(nn.Module):
    """First aggregate viewpoint→grid, then SAGE on grid-grid graph."""
    def __init__(self, grid_dim, vp_dim, num_classes=5, h=256):
        super().__init__()
        # Grid encoder
        self.grid_enc = nn.Sequential(nn.Linear(grid_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.15))
        # Viewpoint encoder
        self.vp_enc = nn.Sequential(nn.Linear(vp_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.15))
        # Aggregation: viewpoint→grid message
        self.vp_to_grid = SAGEConv(h, h, flow="source_to_target")
        self.gn1 = nn.BatchNorm1d(h)
        # Grid→grid SAGE
        self.grid_sage1 = SAGEConv(h, h); self.gn2 = nn.BatchNorm1d(h)
        self.grid_sage2 = SAGEConv(h, h//2); self.gn3 = nn.BatchNorm1d(h//2)
        # Heads
        self.head = nn.Sequential(nn.Linear(h//2, h//4), nn.GELU(), nn.Linear(h//4, num_classes))
        self.dropout = 0.15

    def forward(self, x_grid, x_vp, ei_grid, ei_vp2grid):
        # Encode
        g = self.grid_enc(x_grid)
        v = self.vp_enc(x_vp)
        # Concatenate all node features for SAGE: [grid_nodes, vp_nodes]
        h_all = torch.cat([g, v], dim=0)
        # VP→Grid message passing (only vp→grid edges)
        h_grid = F.gelu(self.gn1(self.vp_to_grid(h_all, ei_vp2grid)))
        # Grid→Grid SAGE
        h_grid = F.gelu(self.gn2(self.grid_sage1(h_grid, ei_grid)))
        h_grid = F.dropout(h_grid, p=self.dropout, training=self.training)
        h_grid = F.gelu(self.gn3(self.grid_sage2(h_grid, ei_grid)))
        return self.head(h_grid)

# ====== Build viewpoint features ======
def build_vp_features(data_root):
    vp_feats = []
    for prefix, csv_name, xc, yc, zc, hc in [
        ("svi", "SVI_position_road.csv", "X", "Y", "Z", "Heading"),
        ("wvi", "window_view_location.csv", "X", "Y", "Z", "Heading"),
        ("dvi", "DVI_position_60.csv", "x", "y", "z", "heading"),
    ]:
        df = pd.read_csv(f"{data_root}/viewpoints/{csv_name}")
        x = df[xc].values.astype(np.float32)
        y = df[yc].values.astype(np.float32)
        z = df[zc].values.astype(np.float32)
        h = df[hc].values.astype(np.float32)
        h_rad = np.deg2rad(h)
        sin_h = np.sin(h_rad); cos_h = np.cos(h_rad)
        # Normalize position
        x_n = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-8)
        z_n = (z - z.min()) / (z.max() - z.min() + 1e-8)
        feats = np.stack([x_n, y_n, z_n, sin_h, cos_h], axis=1)
        vp_feats.append(feats)
    return np.concatenate(vp_feats, axis=0).astype(np.float32)

# ====== Build heterogeneous graph ======
def build_hetero_graph(grid_df, data_root, r=50):
    cx = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    cy = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    gcenters = np.stack([cx, cy], axis=1)
    n_grids = len(grid_df)

    # Grid-grid edges (k=8 spatial kNN)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=9); nn.fit(gcenters); _, gidx = nn.kneighbors(gcenters)
    g_edges = set()
    for i in range(n_grids):
        for j in gidx[i, 1:]: g_edges.add((i, j) if i < j else (j, i))
    g_src, g_dst = zip(*g_edges)
    ei_grid = torch.tensor([list(g_src) + list(g_dst), list(g_dst) + list(g_src)], dtype=torch.long)

    # Viewpoint→Grid edges (within radius r)
    vp_offsets = [0]  # cumulative offset for viewpoint indices
    vp_coords_list = []
    for prefix, csv_name, xc, yc in [
        ("svi", "SVI_position_road.csv", "X", "Y"),
        ("wvi", "window_view_location.csv", "X", "Y"),
        ("dvi", "DVI_position_60.csv", "x", "y"),
    ]:
        df = pd.read_csv(f"{data_root}/viewpoints/{csv_name}")
        vp_coords_list.append(df[[xc, yc]].values.astype(np.float64))
        vp_offsets.append(vp_offsets[-1] + len(df))
    vp_coords = np.concatenate(vp_coords_list, axis=0)
    vp_tree = KDTree(vp_coords)
    n_vps = len(vp_coords)

    print(f"Grids: {n_grids}, Viewpoints: {n_vps}")
    print(f"Grid-grid edges: {ei_grid.shape[1]}")

    # VP→Grid edges (directed: vp → grid)
    vp2g_src, vp2g_dst = [], []
    g_idxs = vp_tree.query_radius(gcenters, r=r)
    for gi, vp_idx in enumerate(g_idxs):
        # vp indices are in [0, n_vps), grid indices need offset
        vp2g_src.extend(vp_idx)  # viewpoint index (in full vp range)
        vp2g_dst.extend([gi] * len(vp_idx))  # grid index
    ei_vp2grid = torch.tensor([vp2g_src, vp2g_dst], dtype=torch.long)

    print(f"VP→Grid edges: {ei_vp2grid.shape[1]}")

    # VP→Grid edges use concatenated indexing: grid nodes first [0, n_grids), then vp nodes [n_grids, n_grids+n_vps)
    ei_vp2grid_global = ei_vp2grid.clone()
    ei_vp2grid_global[0] += n_grids  # offset vp indices

    return ei_grid, ei_vp2grid_global, n_vps, vp_offsets

# ====== Main ======
def main(data_root="..", device_str="cuda", smoke=False):
    data_root = "/root/GVVI_GNN_Demo" if "root" in data_root else data_root
    t0 = time.time()

    # Load grid
    grid_df = pd.read_csv(f"{data_root}/greenery/df_grid_ply.csv")
    if smoke: grid_df = grid_df.head(500).copy()
    n = len(grid_df)
    print(f"Grid nodes: {n}")

    # Targets
    with rasterio.open(f"{data_root}/reference/GVVI.tif") as src: gvvi = src.read(1)
    xs = grid_df["grid_idx_x"].values; ys = grid_df["grid_idx_y"].values; vals = gvvi[xs, ys]
    gvvi100 = vals * 100
    classes = np.zeros(n, dtype=np.int64)
    classes[(vals > 0) & (gvvi100 < 5)] = 1
    classes[(gvvi100 >= 5) & (gvvi100 < 15)] = 2
    classes[(gvvi100 >= 15) & (gvvi100 < 30)] = 3
    classes[gvvi100 >= 30] = 4

    # Grid features (from V4 — base 196 dim)
    fbase = np.load(f"{data_root}/gvvi_grid_demo_v2/processed/features_v2.npz")["features"]
    if smoke: fbase = fbase[:500]
    # Standardize
    split = np.load(f"{data_root}/gvvi_grid_demo_v3/processed/split_train80_val5_test15_seed42.npz")
    tm = split["train_mask"]; vm = split["val_mask"]
    if smoke: tm = tm[:500]; vm = vm[:500]
    sm = np.zeros(n, dtype=bool)
    test_idx = split["test_idx"]; test_idx = test_idx[test_idx < n]
    sm[test_idx] = True

    scaler = StandardScaler(); scaler.fit(fbase[tm])
    x_grid = scaler.transform(fbase).astype(np.float32)
    print(f"Grid features: {x_grid.shape}")

    # Viewpoint features
    x_vp = build_vp_features(data_root)
    print(f"VP features: {x_vp.shape}")

    # Build graphs
    ei_grid, ei_vp2grid, n_vps, _ = build_hetero_graph(grid_df, data_root, r=50)

    # Device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    xg = torch.tensor(x_grid).float().to(device)
    xv = torch.tensor(x_vp).float().to(device)
    yt = torch.tensor(classes).long().to(device)
    eig = ei_grid.to(device); eiv = ei_vp2grid.to(device)

    # Model
    model = HeteroSAGE(x_grid.shape[1], x_vp.shape[1], 5).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    cnts = np.bincount(classes[tm], minlength=5); cnts = np.maximum(cnts, 1)
    cb_w = torch.tensor((1 - 0.999) / (1 - 0.999 ** cnts), device=device).float()
    cb_w = cb_w / cb_w.sum() * 5

    if smoke:
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        for ep in range(1, 4):
            model.train(); opt.zero_grad()
            logits = model(xg, xv, eig, eiv)
            loss = F.cross_entropy(logits[tm], yt[tm], weight=cb_w, label_smoothing=0.05)
            loss.backward(); opt.step()
            print(f"Smoke ep {ep}: loss={loss.item():.4f}")
        print("Smoke test PASSED")
        return

    # Full training
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=30, min_lr=1e-6)
    best_a, best_e, pat = 0.0, 0, 0
    ckpt = f"{data_root}/gvvi_grid_demo_v5/checkpoints/best_hetero.pt"
    torch.save(model.state_dict(), ckpt)

    for ep in range(1, 1501):
        model.train(); opt.zero_grad()
        logits = model(xg, xv, eig, eiv)
        ce = F.cross_entropy(logits[tm], yt[tm], weight=cb_w, label_smoothing=0.05, reduction="none")
        pt = torch.exp(-ce); focal = ((1 - pt) ** 1.5 * ce).mean()
        probs = F.softmax(logits[tm], dim=-1); ci = torch.arange(5, device=device).float()
        pred_ord = (probs * ci).sum(dim=-1); ordinal = F.smooth_l1_loss(pred_ord, yt[tm].float())
        loss = focal + 0.3 * ordinal
        if not torch.isnan(loss): loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            vp_pred = model(xg, xv, eig, eiv)[vm].argmax(dim=-1).cpu().numpy()
            va = accuracy_score(classes[vm], vp_pred)
        sched.step(va)
        if va > best_a - 1e-8: best_a = va; best_e = ep; pat = 0; torch.save(model.state_dict(), ckpt)
        else: pat += 1
        if ep <= 5 or ep % 50 == 0: print(f"Ep {ep}: val={va*100:.1f}%, best={best_a*100:.1f}%")
        if pat >= 150: break

    model.load_state_dict(torch.load(ckpt))
    print(f"Best: ep {best_e}")

    model.eval()
    with torch.no_grad():
        preds = model(xg, xv, eig, eiv).argmax(dim=-1).cpu().numpy()
    yt_test = classes[sm]; yp_test = preds[sm]
    acc = accuracy_score(yt_test, yp_test)
    sp, _ = spearmanr(yt_test, yp_test)
    w1 = (np.abs(yt_test - yp_test) <= 1).mean()

    print(f"\n===== V5 Heterogeneous Graph =====")
    print(f"Exact Acc: {acc*100:.1f}%, +-1: {w1*100:.1f}%, Spearman: {sp:.4f}")
    print(f"Total time: {time.time()-t0:.0f}s")
    for c in range(5):
        cm = (yt_test == c)
        if cm.sum():
            rec = (yp_test[cm] == c).mean()
            print(f"  C{c}: rec={rec*100:.1f}%, n={cm.sum()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.data_root, args.device, args.smoke)
