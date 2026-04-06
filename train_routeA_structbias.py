#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_routeA_structbias.py
Route-A + Structural Bias + (optional) AUX TF/Pathway Reconstruction

✅ Fixes in this version
-----------------------
1) ALWAYS uses the CLI --lr_h5ad path (no hidden fallback to base/features_lr/...).
2) Hard sanity checks: if edge lr_idx exceeds lr_h5ad.n_obs -> raise early with clear message.
3) --at_csv: if passed without ".csv" but "<path>.csv" exists, it will auto-resolve.
4) edge_attr keeps ALL numeric columns except src/dst and LR name columns,
   so cell->spot edge_attr_dim can be 13 if your geomprob file contains those columns.

Outputs
-------
- emb_cell_xformer_structbias.pt
- emb_spot_xformer_structbias.pt
- emb_lr_xformer_structbias.pt
- hgt_xformer_structbias_best.pt
"""

import os, argparse, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import HeteroConv, TransformerConv


# ------------------ IO utils ------------------

def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, encoding="latin-1", engine="python")

def _resolve_path(p: str, base: str) -> Path:
    """
    Resolve a possibly-relative path against base.
    Also: if the file does not exist and p has no suffix but p+'.csv' exists -> use that.
    """
    P = Path(str(p))
    # absolute
    if P.is_absolute():
        if P.exists():
            return P
        # auto-append .csv
        if P.suffix == "" and Path(str(P) + ".csv").exists():
            return Path(str(P) + ".csv")
        return P

    # relative (try as given)
    if P.exists():
        return P.resolve()

    # relative to base
    Q = (Path(base) / P).resolve()
    if Q.exists():
        return Q

    # auto-append .csv
    if Q.suffix == "" and Path(str(Q) + ".csv").exists():
        return Path(str(Q) + ".csv")

    return Q

def _to_numpy_dense(x) -> np.ndarray:
    import scipy.sparse as sp
    if sp.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr

def _zscore_block(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    mu = np.mean(arr, axis=0, keepdims=True, dtype=np.float32)
    sd = np.std(arr, axis=0, keepdims=True, dtype=np.float32)
    sd = np.maximum(sd, eps)
    out = (arr - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)

def stack_obsm(adx: ad.AnnData, keys: List[str], who: str, block_zscore: bool) -> np.ndarray:
    mats = []
    for k in keys:
        if k in adx.obsm and adx.obsm[k] is not None:
            arr = _to_numpy_dense(adx.obsm[k])
            if arr.ndim == 2 and arr.shape[0] == adx.n_obs:
                if block_zscore:
                    arr = _zscore_block(arr)
                print(f"[obsm OK] {who}.{k}: {arr.shape}")
                mats.append(arr)
            else:
                warnings.warn(f"[obsm WARN] {who}.{k} bad shape {arr.shape}")
    if mats:
        X = np.concatenate(mats, axis=1).astype(np.float32, copy=False)
        X[~np.isfinite(X)] = 0.0
        return X
    X = _to_numpy_dense(adx.X)
    if block_zscore:
        X = _zscore_block(X)
    print(f"[obsm Fallback] {who}.X: {X.shape}")
    return X

def numeric_edge_attr(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in set(drop_cols)]
    num = df[keep].select_dtypes(include=[np.number]).copy()
    return num.fillna(0.0)


# ------------------ CLI ------------------

def get_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", required=True)

    # features
    ap.add_argument("--cell_h5ad", required=True)
    ap.add_argument("--spot_h5ad", required=True)
    ap.add_argument("--lr_h5ad",   required=True)

    # edges
    ap.add_argument("--secrete_csv",   required=True)  # cell->lr
    ap.add_argument("--bind_csv",      required=True)  # lr->cell
    ap.add_argument("--present_csv",   required=True)  # spot->lr
    ap.add_argument("--bind_spot_csv", required=True)  # lr->spot
    ap.add_argument("--at_csv",        required=True)  # cell->spot
    ap.add_argument("--nei_csv",       required=True)  # spot->spot

    ap.add_argument("--spdist1_csv", default="")
    ap.add_argument("--spdist2_csv", default="")
    ap.add_argument("--spdist3_csv", default="")

    # model / train
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out_dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--fanout", type=int, default=8)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--epochs_pre", type=int, default=5)
    ap.add_argument("--epochs_ft",  type=int, default=20)

    ap.add_argument("--lr_pre", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--train_rels", default="secrete,bind,present,at",
                    help="subset: secrete,bind,present,bind_spot,at,neighbor,spdist1,spdist2,spdist3")

    # optional aux recon loss
    ap.add_argument("--lambda_aux", type=float, default=0.0,
                    help="0 disables aux TF/path recon; try 0.05~0.2")
    ap.add_argument("--lambda_aux_path", type=float, default=1.0)
    ap.add_argument("--lambda_aux_tf",   type=float, default=1.0)

    ap.add_argument("--block_zscore", action="store_true",
                    help="z-score each obsm block before concat")

    # export embeddings
    ap.add_argument("--infer_batch", type=int, default=4096)
    ap.add_argument("--infer_full_neighbors", action="store_true")
    ap.add_argument("--infer_fanout", type=int, default=10)

    return ap.parse_args()


# ------------------ Build heterodata ------------------

def build_heterodata(args) -> HeteroData:
    cell_p = _resolve_path(args.cell_h5ad, args.base)
    spot_p = _resolve_path(args.spot_h5ad, args.base)
    lr_p   = _resolve_path(args.lr_h5ad,   args.base)

    for tag, fp in [("cell_h5ad", cell_p), ("spot_h5ad", spot_p), ("lr_h5ad", lr_p)]:
        if not fp.exists():
            raise FileNotFoundError(f"[missing] {tag}: {fp}")

    print(f"[paths] cell={cell_p}")
    print(f"[paths] spot={spot_p}")
    print(f"[paths] lr  ={lr_p}")

    ac   = ad.read_h5ad(str(cell_p))
    aspt = ad.read_h5ad(str(spot_p))
    alr  = ad.read_h5ad(str(lr_p))

    print(f"[n_obs] cell={ac.n_obs} spot={aspt.n_obs} lr={alr.n_obs}")

    # input features
    Xc = stack_obsm(ac,   ["X_expr_pca", "X_ct_onehot", "X_pathway_progeny14", "X_tfact_dorothea"],
                    who="cell", block_zscore=args.block_zscore)
    Xs = stack_obsm(aspt, ["X_sp_pathway_progeny14", "X_sp_tfact_dorothea",
                           "X_sp_niche_progeny_r1", "X_sp_niche_tfact_r1", "X_sp_pe"],
                    who="spot", block_zscore=args.block_zscore)
    Xl = _to_numpy_dense(alr.X)

    # aux targets (optional; used only if lambda_aux>0 and these exist)
    y_cell_path = _to_numpy_dense(ac.obsm["X_pathway_progeny14"]) if "X_pathway_progeny14" in ac.obsm else None
    y_cell_tf   = _to_numpy_dense(ac.obsm["X_tfact_dorothea"])    if "X_tfact_dorothea" in ac.obsm else None
    y_spot_path = _to_numpy_dense(aspt.obsm["X_sp_pathway_progeny14"]) if "X_sp_pathway_progeny14" in aspt.obsm else None
    y_spot_tf   = _to_numpy_dense(aspt.obsm["X_sp_tfact_dorothea"])    if "X_sp_tfact_dorothea" in aspt.obsm else None

    data = HeteroData()
    data["cell"].x = torch.from_numpy(Xc)
    data["spot"].x = torch.from_numpy(Xs)
    data["lr"].x   = torch.from_numpy(Xl)

    if y_cell_path is not None:
        data["cell"].y_path = torch.from_numpy(y_cell_path)
        print(f"[aux] cell.y_path: {tuple(data['cell'].y_path.shape)}")
    if y_cell_tf is not None:
        data["cell"].y_tf = torch.from_numpy(y_cell_tf)
        print(f"[aux] cell.y_tf:   {tuple(data['cell'].y_tf.shape)}")
    if y_spot_path is not None:
        data["spot"].y_path = torch.from_numpy(y_spot_path)
        print(f"[aux] spot.y_path: {tuple(data['spot'].y_path.shape)}")
    if y_spot_tf is not None:
        data["spot"].y_tf = torch.from_numpy(y_spot_tf)
        print(f"[aux] spot.y_tf:   {tuple(data['spot'].y_tf.shape)}")

    n_cell, n_spot, n_lr = ac.n_obs, aspt.n_obs, alr.n_obs

    def add_edges(fp: str, src_col: str, dst_col: str, et: Tuple[str,str,str], n_src: int, n_dst: int):
        p = _resolve_path(fp, args.base)
        if not p.exists():
            warnings.warn(f"[edge skip] {et}: file not found: {p}")
            return

        df = read_csv_any(p)
        if src_col not in df.columns or dst_col not in df.columns:
            warnings.warn(f"[edge skip] {et}: missing {src_col}/{dst_col} in {p}")
            return

        src = pd.to_numeric(df[src_col], errors="coerce")
        dst = pd.to_numeric(df[dst_col], errors="coerce")
        if src.isna().any() or dst.isna().any():
            bad = int(src.isna().sum() + dst.isna().sum())
            raise ValueError(f"[edge] {et}: {bad} NaN indices in {p}")

        src = src.astype(np.int64).to_numpy()
        dst = dst.astype(np.int64).to_numpy()

        # ✅ sanity check (this is what fixes your current crash early)
        smax = int(src.max()) if src.size else -1
        dmax = int(dst.max()) if dst.size else -1
        if smax >= n_src or dmax >= n_dst:
            raise ValueError(
                f"[edge index mismatch] {et} from {p}\n"
                f"  src max={smax} n_src={n_src}\n"
                f"  dst max={dmax} n_dst={n_dst}\n"
                f"Fix: ensure edges were built using THE SAME feature h5ad sizes."
            )

        ei = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        data[et].edge_index = ei

        # keep numeric edge_attr
        drop_like = {
            src_col, dst_col,
            "lr", "pair", "pair_name", "lr_id", "lr_key",
            "ligand", "receptor", "ip",
        }
        num = numeric_edge_attr(df, drop_cols=list(drop_like))
        if num.shape[1] > 0:
            ea = torch.from_numpy(num.to_numpy(np.float32))
            ea = torch.nan_to_num(ea, 0.0, 0.0, 0.0)
            mean = ea.mean(0, keepdim=True)
            std  = ea.std(0, keepdim=True).clamp_min(1e-6)
            data[et].edge_attr = (ea - mean) / std

        dim = int(data[et].edge_attr.size(-1)) if "edge_attr" in data[et] else 0
        print(f"[edge] {et} | n={ei.size(1)} | edge_attr_dim={dim} | file={p.name}")

    # main relations
    add_edges(args.secrete_csv,   "src_cell_idx", "dst_lr_idx",   ("cell","secrete","lr"),   n_cell, n_lr)
    add_edges(args.bind_csv,      "src_lr_idx",   "dst_cell_idx", ("lr","bind","cell"),      n_lr,   n_cell)
    add_edges(args.present_csv,   "src_spot_idx", "dst_lr_idx",   ("spot","present","lr"),   n_spot, n_lr)
    add_edges(args.bind_spot_csv, "src_lr_idx",   "dst_spot_idx", ("lr","bind","spot"),      n_lr,   n_spot)
    add_edges(args.at_csv,        "src_cell_idx", "dst_spot_idx", ("cell","at","spot"),      n_cell, n_spot)
    add_edges(args.nei_csv,       "src_spot_idx", "dst_spot_idx", ("spot","neighbor","spot"),n_spot, n_spot)

    # optional structural bias
    if args.spdist1_csv:
        add_edges(args.spdist1_csv, "src_spot_idx","dst_spot_idx",("spot","spdist1","spot"), n_spot, n_spot)
    if args.spdist2_csv:
        add_edges(args.spdist2_csv, "src_spot_idx","dst_spot_idx",("spot","spdist2","spot"), n_spot, n_spot)
    if args.spdist3_csv:
        add_edges(args.spdist3_csv, "src_spot_idx","dst_spot_idx",("spot","spdist3","spot"), n_spot, n_spot)

    # reverse edges share attrs
    for (s,r,d) in list(data.edge_types):
        rev = (d, f"rev_{r}", s)
        if rev in data.edge_types:
            continue
        if "edge_index" in data[(s,r,d)]:
            data[rev].edge_index = data[(s,r,d)].edge_index.flip(0)
            if "edge_attr" in data[(s,r,d)]:
                data[rev].edge_attr = data[(s,r,d)].edge_attr

    print("[meta]", data.metadata())
    return data


# ------------------ Model ------------------

class HeteroTransformerEncoder(nn.Module):
    def __init__(self, data: HeteroData, hidden: int, out_dim: int, heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.ntypes = list(data.node_types)
        in_dims = {nt: int(data[nt].x.size(-1)) for nt in self.ntypes}

        self.in_proj = nn.ModuleDict({nt: nn.Linear(in_dims[nt], hidden) for nt in self.ntypes})
        self.norms   = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in self.ntypes})
        self.dropout = nn.Dropout(dropout)
        self.out_proj= nn.ModuleDict({nt: nn.Linear(hidden, out_dim) for nt in self.ntypes})

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = {}
            for et in data.edge_types:
                edge_dim = int(data[et].edge_attr.size(-1)) if "edge_attr" in data[et] else None
                convs[et] = TransformerConv(
                    in_channels=(hidden, hidden),
                    out_channels=hidden,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    root_weight=True,
                    concat=False,
                )
            self.layers.append(HeteroConv(convs, aggr="sum"))

        # aux heads (optional)
        self.aux_heads = nn.ModuleDict()
        if "cell" in data.node_types:
            if hasattr(data["cell"], "y_path"):
                self.aux_heads["cell_path"] = nn.Linear(out_dim, int(data["cell"].y_path.size(-1)))
            if hasattr(data["cell"], "y_tf"):
                self.aux_heads["cell_tf"]   = nn.Linear(out_dim, int(data["cell"].y_tf.size(-1)))
        if "spot" in data.node_types:
            if hasattr(data["spot"], "y_path"):
                self.aux_heads["spot_path"] = nn.Linear(out_dim, int(data["spot"].y_path.size(-1)))
            if hasattr(data["spot"], "y_tf"):
                self.aux_heads["spot_tf"]   = nn.Linear(out_dim, int(data["spot"].y_tf.size(-1)))

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        h = {nt: F.relu(self.in_proj[nt](batch[nt].x.float())) for nt in self.ntypes}
        edge_attr_dict = {et: batch[et].edge_attr for et in batch.edge_types if "edge_attr" in batch[et]}
        for hetero in self.layers:
            h_new = hetero(
                x_dict=h,
                edge_index_dict={et: batch[et].edge_index for et in batch.edge_types},
                edge_attr_dict=edge_attr_dict
            )
            for nt in h:
                h[nt] = self.dropout(F.relu(self.norms[nt](h[nt] + h_new[nt])))
        z = {nt: self.out_proj[nt](h[nt]) for nt in h}
        for nt in z:
            z[nt] = torch.nan_to_num(z[nt], 0.0, 0.0, 0.0)
        return z


# ------------------ Loss utils ------------------

def safe_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1e-8)

def info_nce(z1: Dict[str,torch.Tensor], z2: Dict[str,torch.Tensor], temp=0.2) -> torch.Tensor:
    losses = []
    for nt in z1:
        a = safe_norm(z1[nt]); b = safe_norm(z2[nt])
        logits = (a @ b.t()) / max(1e-6, float(temp))
        logits = torch.nan_to_num(logits, 0.0, 0.0, 0.0)
        target = torch.arange(a.size(0), device=logits.device)
        losses.append(F.cross_entropy(logits, target))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=list(z1.values())[0].device)

def make_bce_with_pos_weight(y: torch.Tensor):
    pos = y.sum().clamp(min=1.0)
    neg = (y.numel() - y.sum()).clamp(min=1.0)
    pw = (neg / pos).detach()
    return nn.BCEWithLogitsLoss(pos_weight=pw)

def aux_recon_loss(model: HeteroTransformerEncoder, batch: HeteroData, z: Dict[str,torch.Tensor],
                   lambda_aux: float, lambda_path: float, lambda_tf: float) -> torch.Tensor:
    if lambda_aux <= 0:
        return torch.tensor(0.0, device=list(z.values())[0].device)

    loss = 0.0
    terms = 0

    if "cell" in batch.node_types and "cell" in z:
        if hasattr(batch["cell"], "y_path") and "cell_path" in model.aux_heads:
            pred = model.aux_heads["cell_path"](z["cell"])
            loss = loss + lambda_path * F.mse_loss(pred, batch["cell"].y_path.float())
            terms += 1
        if hasattr(batch["cell"], "y_tf") and "cell_tf" in model.aux_heads:
            pred = model.aux_heads["cell_tf"](z["cell"])
            loss = loss + lambda_tf * F.mse_loss(pred, batch["cell"].y_tf.float())
            terms += 1

    if "spot" in batch.node_types and "spot" in z:
        if hasattr(batch["spot"], "y_path") and "spot_path" in model.aux_heads:
            pred = model.aux_heads["spot_path"](z["spot"])
            loss = loss + lambda_path * F.mse_loss(pred, batch["spot"].y_path.float())
            terms += 1
        if hasattr(batch["spot"], "y_tf") and "spot_tf" in model.aux_heads:
            pred = model.aux_heads["spot_tf"](z["spot"])
            loss = loss + lambda_tf * F.mse_loss(pred, batch["spot"].y_tf.float())
            terms += 1

    if terms == 0:
        return torch.tensor(0.0, device=list(z.values())[0].device)

    return float(lambda_aux) * (loss / terms)


# ------------------ Loaders ------------------

def build_loaders(data: HeteroData, batch_size: int, hops: int, fanout: int, train_rels: str, num_workers: int):
    rel_map = {
        "secrete":  ("cell","secrete","lr"),
        "bind":     ("lr","bind","cell"),
        "present":  ("spot","present","lr"),
        "bind_spot":("lr","bind","spot"),
        "at":       ("cell","at","spot"),
        "neighbor": ("spot","neighbor","spot"),
        "spdist1":  ("spot","spdist1","spot"),
        "spdist2":  ("spot","spdist2","spot"),
        "spdist3":  ("spot","spdist3","spot"),
    }
    keys = [k.strip() for k in str(train_rels).split(",") if k.strip()]
    target_ets = [rel_map[k] for k in keys if k in rel_map and rel_map[k] in data.edge_types]

    num_neighbors = {et: [fanout]*hops for et in data.edge_types}
    loaders = {}
    for et in target_ets:
        ei = data[et].edge_index
        loaders[et] = LinkNeighborLoader(
            data=data,
            num_neighbors=num_neighbors,
            edge_label_index=(et, ei),
            neg_sampling_ratio=1.0,
            batch_size=batch_size,
            shuffle=True,
            disjoint=False,
            num_workers=max(0, int(num_workers)),
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,
        )
        print(f"[loader] {et} | edges={ei.size(1)}")

    if not loaders:
        raise RuntimeError("No non-empty train edge types found. Check --train_rels and your edge files.")
    return loaders


# ------------------ Export embeddings ------------------

@torch.no_grad()
def export_embeddings_batched(model, data, device, out_dir: Path, batch_size: int, full_neighbors: bool, fanout: int):
    model.eval()
    L = len(model.layers)
    num_neighbors = {et: ([-1]*L if full_neighbors else [fanout]*L) for et in data.edge_types}

    for nt in data.node_types:
        N = int(data[nt].num_nodes)
        loader = NeighborLoader(
            data,
            input_nodes=(nt, torch.arange(N)),
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False
        )
        Z = torch.empty((N, model.out_proj[nt].out_features), dtype=torch.float32, device="cpu")
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            n_id = batch[nt].n_id[:batch[nt].batch_size]
            Z[n_id.cpu()] = out[nt][:batch[nt].batch_size].detach().cpu()

        # IMPORTANT: filenames match what loop_train_score_spatial.py expects
        path = out_dir / f"emb_{nt}_xformer_structbias.pt"
        torch.save(Z, path)
        print("[save]", path, "|", tuple(Z.shape))


# ------------------ Train ------------------

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    data = build_heterodata(args)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.base) / "hgt_graph/hgt_xformer_structbias")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = HeteroTransformerEncoder(
        data=data,
        hidden=args.hidden,
        out_dim=args.out_dim,
        heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    loaders = build_loaders(
        data=data,
        batch_size=args.batch_size,
        hops=args.hops,
        fanout=args.fanout,
        train_rels=args.train_rels,
        num_workers=args.num_workers
    )

    # ---------- pretrain (InfoNCE) ----------
    opt_pre = torch.optim.AdamW(model.parameters(), lr=args.lr_pre, weight_decay=args.weight_decay)

    def fwd_once(batch):
        batch = batch.to(device)
        return model(batch)

    if args.epochs_pre > 0:
        model.train()
        for ep in range(1, args.epochs_pre + 1):
            total, steps = 0.0, 0
            for _, loader in loaders.items():
                for batch in loader:
                    opt_pre.zero_grad(set_to_none=True)
                    z1 = fwd_once(batch); z2 = fwd_once(batch)
                    loss = info_nce(z1, z2, temp=0.2)
                    if not torch.isfinite(loss):
                        continue
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    opt_pre.step()
                    total += float(loss.detach()); steps += 1
            print(f"[pretrain {ep}/{args.epochs_pre}] loss={total/max(1,steps):.6f}")

    # ---------- finetune (BCE + optional aux) ----------
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    opt_ft = torch.optim.AdamW(model.parameters(), lr=args.lr_ft, weight_decay=args.weight_decay)

    best = float("inf")
    for ep in range(1, args.epochs_ft + 1):
        model.train()
        tot_link, tot_aux, tot_all, cnt = 0.0, 0.0, 0.0, 0

        for et, loader in loaders.items():
            s_nt, _, t_nt = et
            for batch in loader:
                batch = batch.to(device)
                opt_ft.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                    z = model(batch)

                    src, dst = batch[et].edge_label_index
                    y = batch[et].edge_label.float().clamp_(0, 1)

                    logits = (z[s_nt][src] * z[t_nt][dst]).sum(-1)
                    logits = torch.nan_to_num(logits, 0.0, 0.0, 0.0)

                    crit = make_bce_with_pos_weight(y)
                    loss_link = crit(logits, y)

                    loss_aux = aux_recon_loss(
                        model, batch, z,
                        lambda_aux=float(args.lambda_aux),
                        lambda_path=float(args.lambda_aux_path),
                        lambda_tf=float(args.lambda_aux_tf),
                    )

                    loss = loss_link + loss_aux

                if not torch.isfinite(loss):
                    continue

                scaler.scale(loss).backward()
                if hasattr(scaler, "unscale_"):
                    scaler.unscale_(opt_ft)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(opt_ft)
                scaler.update()

                m = int(y.numel())
                tot_link += float(loss_link.detach()) * m
                tot_aux  += float(loss_aux.detach())  * m
                tot_all  += float(loss.detach())      * m
                cnt += m

        avg_link = tot_link / max(1, cnt)
        avg_aux  = tot_aux  / max(1, cnt)
        avg_all  = tot_all  / max(1, cnt)
        print(f"[finetune {ep}/{args.epochs_ft}] loss={avg_all:.6f} (link={avg_link:.6f}, aux={avg_aux:.6f})")

        if avg_all < best:
            best = avg_all
            ckpt = out_dir / "hgt_xformer_structbias_best.pt"
            torch.save({"model": model.state_dict()}, ckpt)
            print("[save]", ckpt)

    # ---------- export embeddings ----------
    export_embeddings_batched(
        model=model,
        data=data,
        device=device,
        out_dir=out_dir,
        batch_size=int(args.infer_batch),
        full_neighbors=bool(args.infer_full_neighbors),
        fanout=int(args.infer_fanout),
    )


if __name__ == "__main__":
    warnings.filterwarnings("once", category=UserWarning)
    main()
