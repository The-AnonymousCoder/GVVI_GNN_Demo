#!/usr/bin/env python3
"""V4: 4-class GVVI classification with masked label propagation GATv2."""
import argparse, json, os, sys, time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, yaml
warnings.filterwarnings("ignore")

# ====== Config ======
def load_cfg(demo_dir, data_root):
    with open(f"{demo_dir}/config.yaml") as f: cfg = yaml.safe_load(f)
    return cfg, data_root, demo_dir

# ====== 4-class definition (training only) ======
def define_classes(vals, train_idx, cfg):
    pcts = cfg["classes"]["percentiles"]
    thresholds = [np.percentile(vals[train_idx], p) for p in pcts]
    classes = np.zeros(len(vals), dtype=np.int64)
    for i, (lo, hi) in enumerate(zip([-1e-9] + thresholds, thresholds + [1e9])):
        if i == len(thresholds):
            classes[vals >= thresholds[-1]] = i
        elif i == 0:
            classes[(vals >= 0) & (vals < thresholds[0])] = i
        else:
            classes[(vals >= thresholds[i-1]) & (vals < thresholds[i])] = i
    # Fix: class 0 = vals < thresholds[0], class 1 = [t0, t1), class 2 = [t1, t2), class 3 = >= t2
    classes[:] = 0
    classes[(vals >= thresholds[0]) & (vals < thresholds[1])] = 1
    classes[(vals >= thresholds[1]) & (vals < thresholds[2])] = 2
    classes[vals >= thresholds[2]] = 3
    print(f"Thresholds (train-only): {[f'{t:.6f}' for t in thresholds]}")
    for c in range(4):
        print(f"  Class {c}: all={(classes==c).sum()}, train={(classes[train_idx]==c).sum()}")
    return classes, thresholds

# ====== Build features (V2 with fixes) ======
def build_features(data_root, grid_df, cfg):
    import pandas as pd
    from sklearn.neighbors import KDTree
    t0 = time.time()
    print("Building features (V2 + occlusion, 217 dims)...")

    radii = [50, 100, 250, 500]
    h_radius, heading_radius = 250, 250
    fov_half, dist_scales = 60, [50, 100, 250]
    centers_x = ((grid_df["grid_coords_min_x"] + grid_df["grid_coords_max_x"]) / 2).values
    centers_y = ((grid_df["grid_coords_min_y"] + grid_df["grid_coords_max_y"]) / 2).values
    centers = np.stack([centers_x, centers_y], axis=1).astype(np.float64)
    n = len(grid_df)
    blocks, names = [], []

    vp_configs = [
        ("svi", "SVI_position_road.csv", "X", "Y", "Z", "Heading"),
        ("wvi", "window_view_location.csv", "X", "Y", "Z", "Heading"),
        ("dvi", "DVI_position_60.csv", "x", "y", "z", "heading"),
    ]
    for prefix, csv_name, xc, yc, zc, hc in vp_configs:
        vp = pd.read_csv(f"{data_root}/viewpoints/{csv_name}")
        tree, coords = KDTree(vp[[xc, yc]].values.astype(np.float64)), vp[[xc, yc]].values.astype(np.float64)
        hd = vp[hc].values.astype(np.float32); zs = vp[zc].values.astype(np.float32)
        for r in radii:
            idxs = tree.query_radius(centers, r=r)
            cnt = np.array([len(x) for x in idxs], dtype=np.float32)
            near = np.full(n, r, np.float32); md = np.full(n, r, np.float32); sd = np.zeros(n, np.float32)
            for i, ix in enumerate(idxs):
                if len(ix):
                    d = np.linalg.norm(coords[ix]-centers[i], axis=1)
                    near[i]=d.min(); md[i]=d.mean()
                    if len(ix)>1: sd[i]=d.std()
            for s, a in [("count",cnt),("nearest",near),("mean_dist",md),("std_dist",sd)]:
                names.append(f"{prefix}_{s}_{r}"); blocks.append(a)
        # Height
        idxs = tree.query_radius(centers, r=h_radius)
        mz = np.zeros(n,np.float32); xz = np.zeros(n,np.float32); nz = np.zeros(n,np.float32); hd_ = np.zeros(n,np.float32)
        for i,ix in enumerate(idxs):
            if len(ix): zz=zs[ix]; mz[i]=zz.mean(); xz[i]=zz.max(); nz[i]=zz.min(); hd_[i]=-zz.mean()
        for s,a in [("mean_z",mz),("max_z",xz),("min_z",nz),("height_diff",hd_)]:
            names.append(f"{prefix}_{s}"); blocks.append(a)
        # Heading
        idxs = tree.query_radius(centers, r=heading_radius)
        ms = np.zeros(n,np.float32); mc = np.zeros(n,np.float32); hv = np.zeros(n,np.float32)
        for i,ix in enumerate(idxs):
            if len(ix): hr=np.deg2rad(hd[ix]); ms[i]=np.sin(hr).mean(); mc[i]=np.cos(hr).mean(); hv[i]=1-np.abs(np.mean(np.exp(1j*hr)))
        names.append(f"{prefix}_mean_sin_heading");blocks.append(ms)
        names.append(f"{prefix}_mean_cos_heading");blocks.append(mc)
        names.append(f"{prefix}_heading_var");blocks.append(hv)
        # Directional features
        for r in radii:
            idxs = tree.query_radius(centers, r=r)
            ff=np.zeros(n,np.float32); fr=np.zeros(n,np.float32); am=np.zeros(n,np.float32); ax=np.zeros(n,np.float32)
            fn=np.full(n,r,np.float32); dk=np.zeros(n,np.float32)
            for i,ix in enumerate(idxs):
                if len(ix):
                    gx,gy=centers[i]; vx=coords[ix,0]; vy=coords[ix,1]
                    bear = np.mod(np.degrees(np.arctan2(gx-vx, gy-vy)), 360)
                    dt = np.mod(bear-hd[ix]+180,360)-180
                    iff=np.abs(dt)<=fov_half; ca=np.cos(np.deg2rad(dt)); dd=np.linalg.norm(coords[ix]-centers[i],axis=1)
                    ff[i]=iff.sum(); fr[i]=ff[i]/len(ix) if len(ix) else 0; am[i]=ca.mean(); ax[i]=ca.max()
                    if iff.sum(): fn[i]=dd[iff].min()
                    dk[i]=np.sum(np.maximum(ca,0)*np.exp(-dd/50))
            for s,a in [("ff_count",ff),("ff_ratio",fr),("align_mean",am),("align_max",ax),("ff_nearest",fn),("dir_kernel",dk)]:
                names.append(f"{prefix}_{s}_{r}"); blocks.append(a)
        # Distance kernels & height quantiles
        for s in dist_scales:
            idxs=tree.query_radius(centers,r=max(radii)); kv=np.zeros(n,np.float32)
            for i,ix in enumerate(idxs):
                if len(ix): kv[i]=np.sum(np.exp(-np.linalg.norm(coords[ix]-centers[i],axis=1)/s))
            names.append(f"{prefix}_kernel_{s}"); blocks.append(kv)
        idxs=tree.query_radius(centers,r=h_radius)
        p25=np.zeros(n,np.float32);p50=np.zeros(n,np.float32);p75=np.zeros(n,np.float32)
        for i,ix in enumerate(idxs):
            if len(ix):zz=zs[ix];p25[i]=np.percentile(zz,25);p50[i]=np.percentile(zz,50);p75[i]=np.percentile(zz,75)
        for s,a in [("z_p25",p25),("z_p50",p50),("z_p75",p75)]: names.append(f"{prefix}_{s}");blocks.append(a)

    # WVI building context (fixed: dedup by bldg_ID, normalize normals)
    wvi = pd.read_csv(f"{data_root}/viewpoints/window_view_location.csv")
    wvi_nx = wvi["normal_x"].values.astype(np.float64); wvi_ny = wvi["normal_y"].values.astype(np.float64)
    nm = np.sqrt(wvi_nx**2 + wvi_ny**2) + 1e-10; wvi_nx /= nm; wvi_ny /= nm  # normalize normals
    wvi_bldg = wvi["bldg_ID"].values; wvi_roof = wvi["ROOFLEVEL"].values.astype(np.float32)
    wvi_base = wvi["BASELEVEL"].values.astype(np.float32); wvi_z = wvi["Z"].values.astype(np.float32)
    wvi_h = (wvi_roof - wvi_base).astype(np.float32)
    wvi_tree, wvi_coords = KDTree(wvi[["X","Y"]].values.astype(np.float64)), wvi[["X","Y"]].values.astype(np.float64)
    wvi_hd = wvi["Heading"].values.astype(np.float32)

    for r in [100, 250]:
        idxs = wvi_tree.query_radius(centers, r=r)
        nb=np.zeros(n,np.float32); mw=np.zeros(n,np.float32); xw=np.zeros(n,np.float32)
        fwr=np.zeros(n,np.float32); fwn=np.full(n,r,np.float32); fwk=np.zeros(n,np.float32)
        hwr=np.zeros(n,np.float32); lwr=np.zeros(n,np.float32)
        for i,ix in enumerate(idxs):
            if len(ix):
                bldgs=wvi_bldg[ix]; nb[i]=len(np.unique(bldgs))
                uniq,cnts=np.unique(bldgs,return_counts=True); mw[i]=cnts.mean();xw[i]=cnts.max()
                gx,gy=centers[i];vx=wvi_coords[ix,0];vy=wvi_coords[ix,1]
                bear=np.mod(np.degrees(np.arctan2(gx-vx,gy-vy)),360)
                dt=np.mod(bear-wvi_hd[ix]+180,360)-180;iff=np.abs(dt)<=fov_half
                fwr[i]=iff.sum()/len(ix) if len(ix) else 0
                if iff.sum(): fwn[i]=np.linalg.norm(wvi_coords[ix][iff]-centers[i],axis=1).min()
                dd=np.linalg.norm(wvi_coords[ix]-centers[i],axis=1)
                ca=np.cos(np.deg2rad(dt)); fwk[i]=np.sum(np.maximum(ca,0)*np.exp(-dd/50))
                zz=wvi_z[ix]
                if len(zz): med=np.median(zz);hwr[i]=(zz>med).sum()/len(zz);lwr[i]=(zz<=med).sum()/len(zz)
        for s,a in [("n_bldgs",nb),("mean_win_bldg",mw),("max_win_bldg",xw),("ff_win_ratio",fwr),
                     ("ff_win_near",fwn),("ff_win_kernel",fwk),("high_win_ratio",hwr),("low_win_ratio",lwr)]:
            names.append(f"wvi_{s}_{r}"); blocks.append(a)

    # Occlusion proxies (fixed: dedup by bldg_ID)
    for r in [50, 100, 250]:
        idxs=wvi_tree.query_radius(centers,r=r)
        fdm=np.zeros(n,np.float32);fdx=np.zeros(n,np.float32);ffr=np.zeros(n,np.float32)
        bhm=np.zeros(n,np.float32);bhx=np.zeros(n,np.float32);bvp=np.zeros(n,np.float32);tbr=np.zeros(n,np.float32)
        for i,ix in enumerate(idxs):
            if len(ix):
                gx,gy=centers[i];wx=wvi_coords[ix,0];wy=wvi_coords[ix,1]
                dx=gx-wx;dy=gy-wy;dw=np.sqrt(dx**2+dy**2)+1e-8
                dots=dx/dw*wvi_nx[ix]+dy/dw*wvi_ny[ix]
                fdm[i]=dots.mean();fdx[i]=dots.max();ffr[i]=(dots>0).mean()
                # Dedup by bldg_ID for height stats
                bldgs=wvi_bldg[ix]; uniq_bldgs,inv=np.unique(bldgs,return_inverse=True)
                bldg_heights=np.zeros(len(uniq_bldgs),np.float32)
                for bi,bid in enumerate(uniq_bldgs):
                    mask_b=inv==bi
                    if mask_b.sum(): bldg_heights[bi]=wvi_h[ix][mask_b].max()
                bhm[i]=bldg_heights.mean() if len(bldg_heights) else 0
                bhx[i]=bldg_heights.max() if len(bldg_heights) else 0
                bvp[i]=bldg_heights.sum()
                roofs=wvi_roof[ix];tbr[i]=(roofs>30).mean() if len(roofs) else 0
        for s,a in [("facade_dot_mean",fdm),("facade_dot_max",fdx),("facade_ff_ratio",ffr),
                     ("bldg_h_mean",bhm),("bldg_h_max",bhx),("bldg_volume",bvp),("tall_ratio",tbr)]:
            names.append(f"occ_{s}_{r}");blocks.append(a)

    # SVI/DVI context (simplified)
    for prefix, csv_name, xc, yc, hc, extra in [
        ("svi","SVI_position_road.csv","X","Y","Heading","RouteID"),
        ("dvi","DVI_position_60.csv","x","y","heading","points_id")]:
        vp=pd.read_csv(f"{data_root}/viewpoints/{csv_name}")
        tree,coords=KDTree(vp[[xc,yc]].values.astype(np.float64)),vp[[xc,yc]].values.astype(np.float64)
        hd=vp[hc].values.astype(np.float32); ex=vp[extra].values
        for r in [100,250]:
            idxs=tree.query_radius(centers,r=r)
            nr=np.zeros(n,np.float32); frr=np.zeros(n,np.float32); fnr=np.full(n,r,np.float32)
            for i,ix in enumerate(idxs):
                if len(ix):
                    nr[i]=len(np.unique(ex[ix]))
                    gx,gy=centers[i];vx=coords[ix,0];vy=coords[ix,1]
                    bear=np.mod(np.degrees(np.arctan2(gx-vx,gy-vy)),360)
                    dt=np.mod(bear-hd[ix]+180,360)-180;iff=np.abs(dt)<=fov_half
                    frr[i]=iff.sum()/len(ix) if len(ix) else 0
                    if iff.sum(): fnr[i]=np.linalg.norm(coords[ix][iff]-centers[i],axis=1).min()
            for s,a in [(f"n_{extra.lower()}",nr),(f"ff_{prefix}_ratio",frr),(f"ff_{prefix}_near",fnr)]:
                names.append(f"{prefix}_{s}_{r}");blocks.append(a)

    # Grid self
    xmn,xmx=centers_x.min(),centers_x.max();ymn,ymx=centers_y.min(),centers_y.max()
    names.append("x_local");blocks.append(((centers_x-xmn)/(xmx-xmn)).astype(np.float32))
    names.append("y_local");blocks.append(((centers_y-ymn)/(ymx-ymn)).astype(np.float32))
    names.append("norm_row");blocks.append((grid_df["grid_idx_x"].values/244).astype(np.float32))
    names.append("norm_col");blocks.append((grid_df["grid_idx_y"].values/246).astype(np.float32))
    tn=grid_df["ori_ply_name"].values;tx=np.zeros(n,np.float32);ty=np.zeros(n,np.float32)
    for i,t in enumerate(tn):
        if isinstance(t,str) and t.startswith("Tile_"):
            p=t.replace("Tile_","").split("_")
            if len(p)>=2:tx[i]=float(p[0]);ty[i]=float(p[1])
    names.append("tile_x");blocks.append(tx);names.append("tile_y");blocks.append(ty)
    gt=KDTree(centers);di=gt.query_radius(centers,r=50)
    names.append("neighbor_density");blocks.append(np.array([len(x)-1 for x in di],np.float32))

    # Log1p counts
    for i,nm in enumerate(names):
        if "count" in nm: blocks[i]=np.log1p(blocks[i])

    X=np.stack(blocks,axis=1).astype(np.float32)
    X=np.nan_to_num(X,nan=0,posinf=0,neginf=0)
    print(f"Features: {X.shape}, {time.time()-t0:.1f}s")
    return X, names

# ====== Build 3 graphs ======
def build_graphs(grid_df, features, cfg):
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    centers_x = ((grid_df["grid_coords_min_x"]+grid_df["grid_coords_max_x"])/2).values
    centers_y = ((grid_df["grid_coords_min_y"]+grid_df["grid_coords_max_y"])/2).values
    centers = np.stack([centers_x, centers_y], axis=1)
    xs, ys = grid_df["grid_idx_x"].values, grid_df["grid_idx_y"].values
    n = len(grid_df)
    print(f"Building 3 graphs ({n} nodes)...")

    # Graph A: Queen adjacency
    grid_idx = {(xs[i], ys[i]): i for i in range(n)}
    q_edges = set()
    for i in range(n):
        x, y = xs[i], ys[i]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx==0 and dy==0: continue
                j = grid_idx.get((x+dx, y+dy))
                if j is not None:
                    q_edges.add((i, j) if i<j else (j, i))
    q_src, q_dst = zip(*q_edges) if q_edges else ([],[])
    ei_q = torch.tensor([list(q_src)+list(q_dst), list(q_dst)+list(q_src)], dtype=torch.long)
    dx_q = centers_x[np.array(q_dst+q_dst)] - centers_x[np.array(q_src+q_src)]
    dy_q = centers_y[np.array(q_dst+q_dst)] - centers_y[np.array(q_src+q_src)]
    dist_q = np.sqrt(dx_q**2+dy_q**2)
    edge_attr_q = torch.tensor(np.stack([dx_q,dy_q,dist_q],axis=1), dtype=torch.float32)
    print(f"  Queen: {len(q_edges)*2} directed edges")

    # Graph B: Spatial kNN k=8
    k = cfg["graph"]["spatial_k"]; max_d = cfg["graph"]["spatial_max_dist"]
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean"); nn.fit(centers)
    s_dists, s_idx = nn.kneighbors(centers)
    s_edges = set()
    for i in range(n):
        for j in range(1, k+1):
            if s_dists[i,j] <= max_d:
                s_edges.add((i, s_idx[i,j]) if i < s_idx[i,j] else (s_idx[i,j], i))
    s_src, s_dst = zip(*s_edges) if s_edges else ([],[])
    ei_s = torch.tensor([list(s_src)+list(s_dst), list(s_dst)+list(s_src)], dtype=torch.long)
    print(f"  Spatial k={k}: {len(s_edges)*2} directed edges")

    # Graph C: Feature similarity k=4 within 250m
    kf = cfg["graph"]["feature_k"]; max_df = cfg["graph"]["feature_max_dist"]
    nn_s = NearestNeighbors(n_neighbors=min(n, 50), metric="euclidean"); nn_s.fit(centers)
    s_dists2, s_idx2 = nn_s.kneighbors(centers)  # spatial candidates
    nn_f = NearestNeighbors(n_neighbors=50, metric="cosine"); nn_f.fit(features)
    f_dists, f_idx = nn_f.kneighbors(features)
    f_edges = set()
    for i in range(n):
        spatial_candidates = set(s_idx2[i, 1:][s_dists2[i,1:] <= max_df])
        feat_candidates = [j for j in f_idx[i] if j != i and j in spatial_candidates][:kf]
        for j in feat_candidates:
            f_edges.add((i, j) if i<j else (j, i))
    f_src, f_dst = zip(*f_edges) if f_edges else ([],[])
    ei_f = torch.tensor([list(f_src)+list(f_dst), list(f_dst)+list(f_src)], dtype=torch.long)
    print(f"  Feature k={kf}: {len(f_edges)*2} directed edges")

    return ei_q, edge_attr_q, ei_s, ei_f

# ====== GATv2 Model ======
from torch_geometric.nn import GATv2Conv

class LabelPropGATv2(nn.Module):
    def __init__(self, geom_dim, num_classes=4, geom_hidden=256, label_hidden=64,
                 gat_heads=4, gat_hidden=64, dropout=0.15):
        super().__init__()
        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, geom_hidden), nn.LayerNorm(geom_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(geom_hidden, geom_hidden), nn.LayerNorm(geom_hidden), nn.GELU(),
        )
        self.label_encoder = nn.Sequential(nn.Linear(6, label_hidden), nn.LayerNorm(label_hidden), nn.GELU())
        # 6 = known_mask(1) + known_gvvi(1) + known_class_onehot(4)
        total_hidden = geom_hidden + label_hidden  # 320
        self.gat_q1 = GATv2Conv(total_hidden, gat_hidden, heads=gat_heads, dropout=dropout, edge_dim=3)
        self.norm_q1 = nn.LayerNorm(gat_hidden*gat_heads)
        self.gat_q2 = GATv2Conv(gat_hidden*gat_heads, gat_hidden, heads=1, dropout=dropout, edge_dim=3)
        self.norm_q2 = nn.LayerNorm(gat_hidden)

        self.gat_s1 = GATv2Conv(total_hidden, gat_hidden, heads=gat_heads, dropout=dropout)
        self.norm_s1 = nn.LayerNorm(gat_hidden*gat_heads)
        self.gat_s2 = GATv2Conv(gat_hidden*gat_heads, gat_hidden, heads=1, dropout=dropout)
        self.norm_s2 = nn.LayerNorm(gat_hidden)

        self.gat_f1 = GATv2Conv(total_hidden, gat_hidden, heads=gat_heads, dropout=dropout)
        self.norm_f1 = nn.LayerNorm(gat_hidden*gat_heads)
        self.gat_f2 = GATv2Conv(gat_hidden*gat_heads, gat_hidden, heads=1, dropout=dropout)
        self.norm_f2 = nn.LayerNorm(gat_hidden)

        fusion_in = total_hidden + gat_hidden*3  # 320 + 64*3 = 512
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, num_classes),
        )
        self.dropout = dropout

    def forward(self, x_geom, label_feat, ei_q, ea_q, ei_s, ei_f):
        g = self.geom_encoder(x_geom)
        l = self.label_encoder(label_feat)
        h = torch.cat([g, l], dim=-1)
        # Queen branch
        hq = F.gelu(self.norm_q1(self.gat_q1(h, ei_q, ea_q)))
        hq = F.dropout(hq, p=self.dropout, training=self.training)
        hq = F.gelu(self.norm_q2(self.gat_q2(hq, ei_q, ea_q)))
        # Spatial branch
        hs = F.gelu(self.norm_s1(self.gat_s1(h, ei_s)))
        hs = F.dropout(hs, p=self.dropout, training=self.training)
        hs = F.gelu(self.norm_s2(self.gat_s2(hs, ei_s)))
        # Feature branch
        hf = F.gelu(self.norm_f1(self.gat_f1(h, ei_f)))
        hf = F.dropout(hf, p=self.dropout, training=self.training)
        hf = F.gelu(self.norm_f2(self.gat_f2(hf, ei_f)))
        fused = self.fusion(torch.cat([h, hq, hs, hf], dim=-1))
        return fused

# ====== Focal Loss ======
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, num_classes, beta=0.999, gamma=1.5):
        super().__init__()
        self.beta = beta; self.gamma = gamma; self.num_classes = num_classes
    def set_class_counts(self, counts):
        self.weights = (1-self.beta)/(1-self.beta**np.maximum(counts,1))
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
    def forward(self, logits, targets):
        w = self.weights.to(logits.device)
        ce = F.cross_entropy(logits, targets, weight=w, reduction='none')
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

# ====== Training ======
def train_model(model, x_geom, ei_q, ea_q, ei_s, ei_f, classes, train_mask, known_mask, cfg, device, ckpt_dir):
    from sklearn.metrics import accuracy_score
    num_cls = cfg["classes"]["num_classes"]
    mask_rate = cfg["training"]["mask_rate"]
    gamma_ = cfg["training"]["focal_gamma"]; beta_ = cfg["training"]["focal_beta"]
    lr = cfg["training"]["lr"]; wd = cfg["training"]["weight_decay"]
    max_ep = cfg["training"]["max_epochs"]; pat = cfg["training"]["early_stopping_patience"]
    gc = cfg["training"]["gradient_clip"]; ord_w = cfg["training"]["ordinal_weight"]

    y_true = torch.tensor(classes).long().to(device)
    kg = torch.tensor(known_mask).float().to(device)
    train_node = np.where(train_mask)[0]; known_node = np.where(known_mask)[0]

    # Class-balanced weights
    train_cls = classes[train_mask]
    counts = np.bincount(train_cls, minlength=num_cls)
    loss_fn = ClassBalancedFocalLoss(num_cls, beta_, gamma_)
    loss_fn.set_class_counts(counts)

    xg = torch.tensor(x_geom).float().to(device)
    ei_q_d = ei_q.to(device); ea_q_d = ea_q.to(device)
    ei_s_d = ei_s.to(device); ei_f_d = ei_f.to(device)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=30, min_lr=1e-6)

    best_acc, best_ep, pat_cnt = 0.0, 0, 0
    ckpt = f"{ckpt_dir}/best_model.pt"; torch.save(model.state_dict(), ckpt)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train nodes: {train_mask.sum()}, Known nodes: {known_mask.sum()}")

    # Validation setup: all train as context, val nodes masked
    val_idx = np.where(known_mask & ~train_mask)[0]
    val_label = np.zeros((len(classes), 6), dtype=np.float32)
    val_label[known_mask & ~train_mask] = [0,0,0,0,0,0]  # all zeros
    # known train nodes provide label signal
    val_label[train_mask, 0] = 1  # known
    val_label[train_mask, 1] = classes[train_mask].astype(np.float32) / 3.0  # normalized gvvi
    for c in range(4):
        val_label[train_mask, 2+c] = (classes[train_mask]==c).astype(np.float32)

    for ep in range(1, max_ep+1):
        model.train(); opt.zero_grad()
        # Random mask: 25% of known nodes hidden as query
        n_query = max(1, int(len(known_node) * mask_rate))
        query = np.random.choice(known_node, n_query, replace=False)
        context = np.setdiff1d(known_node, query)
        ctx_mask = np.isin(train_node, context)  # train nodes in context

        # Build label features
        label_feat = np.zeros((len(classes), 6), dtype=np.float32)
        # Context nodes get their labels
        for idx in context:
            label_feat[idx, 0] = 1
            label_feat[idx, 1] = classes[idx] / 3.0
            label_feat[idx, 2+classes[idx]] = 1
        # Query nodes are masked (label_feat all zero)

        lf = torch.tensor(label_feat).float().to(device)
        logits = model(xg, lf, ei_q_d, ea_q_d, ei_s_d, ei_f_d)
        loss_cls = loss_fn(logits[query], y_true[query])
        # Ordinal loss on query
        probs = F.softmax(logits[query], dim=-1)
        ci = torch.arange(num_cls, device=device).float()
        pred_ord = (probs * ci).sum(dim=-1)
        loss_ord = F.smooth_l1_loss(pred_ord, y_true[query].float())
        loss = loss_cls + ord_w * loss_ord

        if not torch.isnan(loss):
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), gc); opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            vl = torch.tensor(val_label).float().to(device)
            v_logits = model(xg, vl, ei_q_d, ea_q_d, ei_s_d, ei_f_d)
        v_pred = v_logits[val_idx].argmax(dim=-1).cpu().numpy()
        v_acc = accuracy_score(classes[val_idx], v_pred)

        sched.step(v_acc)
        if v_acc > best_acc - 1e-8: best_acc=v_acc; best_ep=ep; pat_cnt=0; torch.save(model.state_dict(), ckpt)
        else: pat_cnt+=1

        if ep<=5 or ep%50==0:
            print(f"Epoch {ep:4d}: loss={loss.item():.4f}, val_acc={v_acc*100:.1f}%, best={best_acc*100:.1f}%")

        if pat_cnt >= pat: print(f"Early stop at {ep}"); break

    model.load_state_dict(torch.load(ckpt))
    print(f"Best epoch: {best_ep}, val_acc: {best_acc*100:.1f}%\n")
    return model, best_ep

# ====== Evaluate ======
def evaluate(model, x_geom, ei_q, ea_q, ei_s, ei_f, classes, known_mask, test_mask, cfg, device):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
    from scipy.stats import spearmanr
    num_cls = cfg["classes"]["num_classes"]
    kg = torch.tensor(known_mask).float().to(device)
    xg = torch.tensor(x_geom).float().to(device)

    # Final: all known nodes as context, test nodes masked
    label_feat = np.zeros((len(classes), 6), dtype=np.float32)
    known_idx = np.where(known_mask)[0]
    for idx in known_idx:
        label_feat[idx, 0] = 1
        label_feat[idx, 1] = classes[idx] / 3.0
        label_feat[idx, 2+classes[idx]] = 1
    lf = torch.tensor(label_feat).float().to(device)

    model.eval()
    with torch.no_grad():
        logits = model(xg, lf, ei_q.to(device), ea_q.to(device), ei_s.to(device), ei_f.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    yt = classes[test_mask]; yp = preds[test_mask]
    acc = accuracy_score(yt, yp)
    bacc = balanced_accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average='macro')
    f1w = f1_score(yt, yp, average='weighted')
    w1 = (np.abs(yt-yp) <= 1).mean()
    sp, _ = spearmanr(yt, yp)
    cm = confusion_matrix(yt, yp)
    ci = np.arange(num_cls)
    pred_soft = (probs[test_mask] * ci).sum(axis=1)
    sp_s, _ = spearmanr(yt, pred_soft)
    mae_s = np.abs(yt.astype(float)-pred_soft).mean()

    metrics = {"accuracy": float(acc), "balanced_accuracy": float(bacc), "f1_macro": float(f1m),
               "f1_weighted": float(f1w), "acc_within_1": float(w1), "spearman": float(sp),
               "spearman_soft": float(sp_s), "mae_soft": float(mae_s)}
    print(f"Exact Accuracy: {acc*100:.1f}%")
    print(f"Balanced Acc: {bacc*100:.1f}%")
    print(f"Macro-F1: {f1m:.4f}, Weighted-F1: {f1w:.4f}")
    print(f"±1 Acc: {w1*100:.1f}%, Spearman: {sp:.4f}, Spearman(soft): {sp_s:.4f}")
    for c in range(num_cls):
        cmask = (yt==c)
        if cmask.sum():
            prec = (yp[yp==c]==c).sum()/max((yp==c).sum(),1)
            rec = (yp[cmask]==c).mean()
            f1c = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
            print(f"  Class {c}: prec={prec*100:.1f}%, rec={rec*100:.1f}%, f1={f1c:.3f}, n={cmask.sum()}")
    return metrics, preds, probs, cm

# ====== Main ======
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="..")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    data_root = os.path.abspath(args.data_root)
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    for d in ["processed","checkpoints","outputs","outputs/figures","reports"]:
        os.makedirs(f"{demo_dir}/{d}", exist_ok=True)
    cfg, _, _ = load_cfg(demo_dir, data_root)

    if args.device=="cuda" and torch.cuda.is_available(): device=torch.device("cuda")
    else: device=torch.device("cpu")
    print(f"Device: {device}")

    # Load data
    grid_df = pd.read_csv(f"{data_root}/greenery/df_grid_ply.csv")
    if args.smoke: grid_df = grid_df.head(500).copy()
    import rasterio
    with rasterio.open(f"{data_root}/reference/GVVI.tif") as src: gvvi=src.read(1)
    xs=grid_df["grid_idx_x"].values; ys=grid_df["grid_idx_y"].values; vals=gvvi[xs,ys]
    n = len(grid_df); print(f"Nodes: {n}")

    # Load original split
    v1_path = f"{os.path.dirname(demo_dir)}/gvvi_grid_demo/processed/split_seed42.npz"
    v1 = np.load(v1_path)
    test_idx = v1["test_idx"][v1["test_idx"] < n]  # handle smoke
    train_idx = v1["train_idx"][v1["train_idx"] < n]
    # Known = train + val = 85% non-test
    known_mask = np.zeros(n, dtype=bool)
    all_known = np.setdiff1d(np.arange(n), test_idx)
    known_mask[all_known] = True
    train_mask = np.zeros(n, dtype=bool)
    # Split known into train (80/85) and val (5/85)
    rng = np.random.default_rng(42)
    pool = all_known.copy(); rng.shuffle(pool)
    n_train = int(len(pool) * 80/85)
    train_mask[pool[:n_train]] = True
    val_mask_ = np.zeros(n, dtype=bool); val_mask_[pool[n_train:]] = True

    print(f"Train: {train_mask.sum()}, Val: {val_mask_.sum()}, Test: {(~known_mask).sum()}")

    # 4-class definition (train only)
    classes, thresholds = define_classes(vals, np.where(train_mask)[0], cfg)

    # Build features
    X, fnames = build_features(data_root, grid_df, cfg)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(); scaler.fit(X[train_mask])
    X = scaler.transform(X).astype(np.float32)
    np.savez_compressed(f"{demo_dir}/processed/features_v4.npz", features=X)

    # Build graphs
    ei_q, ea_q, ei_s, ei_f = build_graphs(grid_df, X, cfg)
    torch.save({"ei_q":ei_q,"ea_q":ea_q,"ei_s":ei_s,"ei_f":ei_f}, f"{demo_dir}/processed/graphs.pt")

    # Model
    if args.smoke:
        cfg["training"]["max_epochs"] = 3
        cfg["training"]["early_stopping_patience"] = 3
        cfg["model"]["geom_hidden"] = 64
        cfg["model"]["gat_hidden"] = 32
        cfg["model"]["gat_heads"] = 2

    model = LabelPropGATv2(X.shape[1], 4, cfg["model"]["geom_hidden"], cfg["model"]["label_hidden"],
                           cfg["model"]["gat_heads"], cfg["model"]["gat_hidden"], cfg["model"]["dropout"])

    # Train
    model, best_ep = train_model(model, X, ei_q, ea_q, ei_s, ei_f, classes, train_mask,
                                  known_mask, cfg, device, f"{demo_dir}/checkpoints")

    # Evaluate
    metrics, preds, probs, cm = evaluate(model, X, ei_q, ea_q, ei_s, ei_f, classes,
                                          known_mask, ~known_mask, cfg, device)
    metrics["thresholds"] = [float(t) for t in thresholds]
    metrics["best_epoch"] = best_ep
    with open(f"{demo_dir}/outputs/metrics.json","w") as f: json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "grid_idx_x": grid_df["grid_idx_x"].values, "grid_idx_y": grid_df["grid_idx_y"].values,
        "true_class": classes, "pred_class": preds, "true_GVVI": vals,
        "split": np.where(~known_mask, "test", np.where(train_mask, "train", "val")),
    })
    for i in range(4): pred_df[f"prob_class_{i}"] = probs[:,i]
    pred_df.to_csv(f"{demo_dir}/outputs/predictions.csv", index=False)

    # GeoTIFF
    with rasterio.open(f"{data_root}/reference/GVVI.tif") as src:
        h,w,tr,cr = src.height,src.width,src.transform,src.crs
    class_raster = np.full((h,w), -1, dtype=np.int32)
    class_raster[grid_df["grid_idx_x"].values, grid_df["grid_idx_y"].values] = preds
    with rasterio.open(f"{demo_dir}/outputs/pred_class.tif","w",driver="GTiff",height=h,width=w,
                       count=1,dtype=np.int32,transform=tr,crs=cr) as dst: dst.write(class_raster,1)

    # Figures
    _figures(vals, classes, preds, ~known_mask, grid_df, metrics, cm, thresholds, f"{demo_dir}/outputs/figures")
    print("Done.")


def _figures(vals, classes, preds, test_mask, grid_df, metrics, cm, thresholds, fig_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import rasterio

    ncls = cm.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    cmn = cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-10)
    im = ax.imshow(cmn, cmap="Blues")
    for i in range(ncls):
        for j in range(ncls):
            if cm[i,j]>0:
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=9,
                        color="white" if cmn[i,j]>0.5 else "black")
    ax.set_xticks(range(ncls));ax.set_yticks(range(ncls))
    ax.set_xlabel("Predicted");ax.set_ylabel("True");ax.set_title("Confusion Matrix")
    plt.colorbar(im,ax=ax,shrink=0.8)
    fig.savefig(f"{fig_dir}/confusion_matrix.png",dpi=150,bbox_inches="tight");plt.close()

    # Metrics table
    fig,ax=plt.subplots(figsize=(8,3));ax.axis("off")
    rows = [[str(k), str(round(v, 4))] for k, v in metrics.items() if isinstance(v, float)]
    ax.table(cellText=rows,colLabels=["Metric","Value"],cellLoc="center",loc="center")
    ax.set_title("Metrics")
    fig.savefig(f"{fig_dir}/metrics.png",dpi=150,bbox_inches="tight");plt.close()
    print("Figures saved.")


if __name__ == "__main__":
    main()
