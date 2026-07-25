"""Simple SAGE + label context features."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from scipy.stats import spearmanr
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import NearestNeighbors
import rasterio

class LabelContextSAGE(nn.Module):
    def __init__(self, in_dim, num_classes=4, h1=256, h2=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(0.15))
        self.conv1 = SAGEConv(h1, h1); self.bn1 = nn.BatchNorm1d(h1)
        self.conv2 = SAGEConv(h1, h2); self.bn2 = nn.BatchNorm1d(h2)
        self.head = nn.Sequential(nn.Linear(h2, h2//2), nn.GELU(), nn.Dropout(0.15), nn.Linear(h2//2, num_classes))
    def forward(self, x, ei):
        x = self.proj(x); x = F.gelu(self.bn1(self.conv1(x, ei)))
        x = F.dropout(x, p=0.15, training=self.training)
        x = F.gelu(self.bn2(self.conv2(x, ei)))
        return self.head(x)

base = "/root/GVVI_GNN_Demo"
grid_df = pd.read_csv(f"{base}/greenery/df_grid_ply.csv")
with rasterio.open(f"{base}/reference/GVVI.tif") as src: gvvi = src.read(1)
xs = grid_df["grid_idx_x"].values; ys = grid_df["grid_idx_y"].values; vals = gvvi[xs, ys]

split_v1 = np.load(f"{base}/gvvi_grid_demo/processed/split_seed42.npz")
test_idx = split_v1["test_idx"]
known_idx = np.setdiff1d(np.arange(len(vals)), test_idx)
rng = np.random.default_rng(42)
pool = known_idx.copy(); rng.shuffle(pool)
n_train = int(len(pool) * 80/85)
train_mask = np.zeros(len(vals), dtype=bool); train_mask[pool[:n_train]] = True
val_mask = np.zeros(len(vals), dtype=bool); val_mask[pool[n_train:]] = True

train_vals = vals[train_mask]
thresh = [np.percentile(train_vals, p) for p in [60, 85, 97]]
print("Thresholds:", [round(float(t),6) for t in thresh])
classes = np.zeros(len(vals), dtype=np.int64)
classes[(vals>=thresh[0])&(vals<thresh[1])] = 1
classes[(vals>=thresh[1])&(vals<thresh[2])] = 2
classes[vals>=thresh[2]] = 3

fdata = np.load(f"{base}/gvvi_grid_demo_v3/processed/features_v3.npz")
feats_raw = fdata["features"]; feats_raw = np.nan_to_num(feats_raw, nan=0, posinf=0, neginf=0)
scaler = StandardScaler(); scaler.fit(feats_raw[train_mask])
feats = scaler.transform(feats_raw).astype(np.float32)

# Label context: 5 dims = known_flag + known_gvvi + class_0_onehot + class_1_onehot + class_2_onehot
label_aug = np.zeros((len(vals), 5), dtype=np.float32)
for i in known_idx:
    label_aug[i, 0] = 1.0
    label_aug[i, 1] = vals[i] / 0.556
    if classes[i] < 3:
        label_aug[i, 2+classes[i]] = 1.0

aug_feats = np.concatenate([feats, label_aug], axis=1).astype(np.float32)
print(f"Features: {aug_feats.shape}")

cx = ((grid_df["grid_coords_min_x"]+grid_df["grid_coords_max_x"])/2).values
cy = ((grid_df["grid_coords_min_y"]+grid_df["grid_coords_max_y"])/2).values
centers = np.stack([cx, cy], axis=1)
nn_m = NearestNeighbors(n_neighbors=9); nn_m.fit(centers)
_, nidx = nn_m.kneighbors(centers)
edges = set()
for i in range(len(vals)):
    for j in nidx[i,1:]: edges.add((i,j) if i<j else (j,i))
src, dst = zip(*edges)
ei = torch.tensor([list(src)+list(dst), list(dst)+list(src)], dtype=torch.long)
print(f"Graph: {len(edges)*2} edges")

device = torch.device("cuda")
x_t = torch.tensor(aug_feats).float().to(device); y_t = torch.tensor(classes).long().to(device)

cnts = np.bincount(classes[train_mask], minlength=4); cnts = np.maximum(cnts, 1)
alpha = torch.tensor(1.0/cnts, device=device).float(); alpha = alpha / alpha.sum() * 4

torch.manual_seed(42); np.random.seed(42)
model = LabelContextSAGE(aug_feats.shape[1], 4).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=30, min_lr=1e-6)
best_acc, best_ep, pat = 0.0, 0, 0
ckpt = f"{base}/gvvi_grid_demo_v4/checkpoints/best_simple.pt"
torch.save(model.state_dict(), ckpt)

for ep in range(1, 1501):
    model.train(); opt.zero_grad()
    logits = model(x_t, ei.to(device))
    loss = F.cross_entropy(logits[train_mask], y_t[train_mask], weight=alpha)
    if not torch.isnan(loss): loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()
    with torch.no_grad():
        v_pred = model(x_t, ei.to(device))[val_mask].argmax(dim=-1).cpu().numpy()
        v_acc = accuracy_score(classes[val_mask], v_pred)
    sched.step(v_acc)
    if v_acc > best_acc - 1e-8: best_acc = v_acc; best_ep = ep; pat = 0; torch.save(model.state_dict(), ckpt)
    else: pat += 1
    if ep <= 5 or ep % 50 == 0:
        t_pred = logits[train_mask].argmax(dim=-1).cpu().numpy()
        print(f"Ep {ep}: loss={loss.item():.4f}, train={accuracy_score(classes[train_mask],t_pred)*100:.1f}%, val={v_acc*100:.1f}%, best={best_acc*100:.1f}%")
    if pat >= 100: break

model.load_state_dict(torch.load(ckpt))
print(f"Best: epoch {best_ep}")

model.eval()
with torch.no_grad():
    preds = model(x_t, ei.to(device)).argmax(dim=-1).cpu().numpy()

yt = classes[test_idx]; yp = preds[test_idx]
acc = accuracy_score(yt, yp); bacc = balanced_accuracy_score(yt, yp)
f1m = f1_score(yt, yp, average="macro"); f1w = f1_score(yt, yp, average="weighted")
w1 = (np.abs(yt-yp) <= 1).mean(); sp, _ = spearmanr(yt, yp)

print(f"\n=== Simple SAGE + Label Context (no random mask) ===")
print(f"Exact Acc: {acc*100:.1f}%, Balanced: {bacc*100:.1f}%")
print(f"Macro-F1: {f1m:.4f}, Weighted-F1: {f1w:.4f}, +-1: {w1*100:.1f}%, Spearman: {sp:.4f}")
for c in range(4):
    cm = (yt==c)
    if cm.sum():
        prec = (yp[yp==c]==c).sum() / max((yp==c).sum(), 1)
        rec = (yp[cm]==c).mean()
        f1c = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
        print(f"  Class {c}: prec={prec*100:.1f}%, rec={rec*100:.1f}%, f1={f1c:.3f}, n={cm.sum()}")
