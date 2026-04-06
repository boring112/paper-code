#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
loop_train_score_spatial.py
Train -> spatial LR scoring -> filter LR -> filter edges -> next round.

Fix in this version
-------------------
1) Rank stage uses lr_idx intersection ONLY (sender dst_lr_idx ∩ receiver src_lr_idx ∩ spot-lr lr_idx),
   no more "LR key -> lr_idx" mapping branch that can drop everything.
2) Safety: clip usable lr_idx by embedding length (0 <= lr < e_lr.shape[0]).
3) Pass-through aux args to train_script:
   --lambda_aux --lambda_aux_path --lambda_aux_tf --block_zscore --focal --focal_alpha --focal_gamma

Notes
-----
- This script assumes your edge CSVs use integer LR indices:
    sender:   src_cell_idx, dst_lr_idx
    receiver: src_lr_idx,   dst_cell_idx
    present:  src_spot_idx, dst_lr_idx
    bind_spot:src_lr_idx,   dst_spot_idx
- at_csv should be the FULL file with multiple numeric columns if you want >1 edge_attr dims,
  e.g. edges_cell_at_spot.geomprob.csv (not a 3-col stripped version).
"""

import os
import math
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, coo_matrix


# ------------------ utils ------------------

def pjoin(base: Path, p: str) -> Path:
    p = str(p)
    if p.startswith("/"):
        return Path(p)
    return (base / p).resolve()

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: List[str]):
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def read_csv_any(p: Path) -> pd.DataFrame:
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(p, encoding="latin-1", engine="python")

def torch_load_safe(p: Path):
    # avoid FutureWarning if possible (weights_only introduced in newer torch)
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(p, map_location="cpu")

def load_pt_tensor(p: Path) -> np.ndarray:
    obj = torch_load_safe(p)
    if isinstance(obj, torch.Tensor):
        return obj.float().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32, copy=False)
    if isinstance(obj, dict):
        for k in ("emb", "tensor", "x"):
            v = obj.get(k, None)
            if isinstance(v, torch.Tensor):
                return v.float().cpu().numpy()
        for v in obj.values():
            if isinstance(v, torch.Tensor):
                return v.float().cpu().numpy()
    raise ValueError(f"Unsupported pt content: {p}")

def stable_sigmoid(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.clip(x / max(1e-6, float(temp)), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))

def topk_indices_desc(v: np.ndarray, k: int) -> np.ndarray:
    if v.size == 0:
        return np.zeros((0,), dtype=np.int64)
    k = max(1, min(int(k), int(v.size)))
    if v.size <= k:
        return np.argsort(-v)
    idx = np.argpartition(-v, k - 1)[:k]
    return idx[np.argsort(-v[idx])]

def pairwise_sqdist_block(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xx = (X ** 2).sum(1, keepdims=True)
    yy = (Y ** 2).sum(1, keepdims=True).T
    return np.maximum(xx + yy - 2.0 * (X @ Y.T), 0.0)

def build_sparse_gaussian_kernel_from_embeddings(
    e_spot: np.ndarray,
    k: int = 20,
    sigma: Optional[float] = None,
    self_loop: bool = True,
    block: int = 512,
) -> csr_matrix:
    """
    Build symmetric kNN Gaussian kernel on spot embeddings.
    Row-normalized (diffusion operator).
    """
    n = e_spot.shape[0]
    rows, cols, dists = [], [], []

    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        D = pairwise_sqdist_block(e_spot[i0:i1], e_spot)  # (B, n)
        kk = min(k, n)
        nn_idx = np.argpartition(D, kth=kk - 1, axis=1)[:, :kk]
        d_top = np.take_along_axis(D, nn_idx, axis=1)
        order = np.argsort(d_top, axis=1)
        nn_idx = np.take_along_axis(nn_idx, order, axis=1)
        d_top = np.take_along_axis(d_top, order, axis=1)

        for r_off in range(i1 - i0):
            r = i0 + r_off
            neigh = nn_idx[r_off]
            dist = d_top[r_off]
            if self_loop and (r not in neigh):
                neigh = np.concatenate([np.array([r], dtype=np.int32), neigh[:-1]])
                dist = np.concatenate([np.array([0.0], dtype=np.float32), dist[:-1]])
            rows.extend([r] * len(neigh))
            cols.extend(neigh.astype(np.int32).tolist())
            dists.extend(dist.astype(np.float32).tolist())

    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    dists = np.asarray(dists, dtype=np.float32)

    if sigma is None:
        pos = dists[dists > 1e-12]
        sigma = float(np.median(pos)) if pos.size > 0 else 1.0
    sigma = max(1e-8, float(sigma))
    denom = 2.0 * (sigma ** 2)
    w = np.exp(-dists / denom).astype(np.float32)

    K = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    K = K.maximum(K.T)  # symmetrize
    row_sum = np.array(K.sum(axis=1)).ravel()
    row_sum[row_sum == 0.0] = 1.0
    K = K.multiply((1.0 / row_sum)[:, None])  # row-stochastic
    return K

def smooth_vec(A: csr_matrix, v: np.ndarray, steps: int = 1) -> np.ndarray:
    out = v.astype(np.float32, copy=True)
    for _ in range(max(0, int(steps))):
        out = A.dot(out)
    return out


# ------------------ spatial LR scoring ------------------

def spatial_rank_lrs(
    base: Path,
    emb_dir: Path,
    sender_csv: Path,
    receiver_csv: Path,
    present_csv: Path,
    bind_spot_csv: Path,
    at_csv: Path,
    nei_csv: Path,
    out_round_dir: Path,
    # params
    relay_mode: str = "kernel",
    knn: int = 20,
    sigma: float = 0.0,
    embed_smooth_steps: int = 1,
    at_prob_col: str = "prob_tangram",
    at_topk: int = 10,
    lrspot_topk: int = 50,
    perlr_topS: int = 200,
    perlr_topT: int = 200,
    perlr_topM: int = 1500,
    dot_temp: float = 3.0,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Returns:
      - lr_rank_df with columns: lr_idx, triad_max, triad_mean, n_triad_kept
      - usable_lr list (intersection)
    """
    ensure_dir(out_round_dir)

    # embeddings (round-specific)
    emb_cell = emb_dir / "emb_cell_xformer_structbias.pt"
    emb_lr   = emb_dir / "emb_lr_xformer_structbias.pt"
    emb_spot = emb_dir / "emb_spot_xformer_structbias.pt"

    e_cell = load_pt_tensor(emb_cell)
    e_lr   = load_pt_tensor(emb_lr)
    e_spot = load_pt_tensor(emb_spot) if emb_spot.exists() else None
    print(f"[emb] cell={e_cell.shape} lr={e_lr.shape} spot={None if e_spot is None else e_spot.shape}")

    # read edges
    s_df  = read_csv_any(sender_csv)
    r_df  = read_csv_any(receiver_csv)
    at_df = read_csv_any(at_csv)
    nei_df= read_csv_any(nei_csv)

    # LR spots: prefer present_csv if exists, else bind_spot
    if present_csv.exists():
        sl_df = read_csv_any(present_csv)
        if not {"src_spot_idx","dst_lr_idx"}.issubset(sl_df.columns):
            raise KeyError(f"present_csv missing src_spot_idx/dst_lr_idx: {present_csv}")
        lr_spot_pairs = sl_df.rename(columns={"src_spot_idx": "spot", "dst_lr_idx": "lr_idx"}).copy()
        lr_spot_pairs["w"] = pd.to_numeric(lr_spot_pairs.get("weight", 1.0), errors="coerce").fillna(1.0)
    else:
        ls_df = read_csv_any(bind_spot_csv)
        if not {"src_lr_idx","dst_spot_idx"}.issubset(ls_df.columns):
            raise KeyError(f"bind_spot_csv missing src_lr_idx/dst_spot_idx: {bind_spot_csv}")
        lr_spot_pairs = ls_df.rename(columns={"src_lr_idx": "lr_idx", "dst_spot_idx": "spot"}).copy()
        lr_spot_pairs["w"] = pd.to_numeric(lr_spot_pairs.get("weight", 1.0), errors="coerce").fillna(1.0)

    # sanity: required cols
    for c in ("src_cell_idx", "dst_lr_idx"):
        if c not in s_df.columns:
            raise KeyError(f"sender_csv missing {c}: {sender_csv}")
    for c in ("src_lr_idx", "dst_cell_idx"):
        if c not in r_df.columns:
            raise KeyError(f"receiver_csv missing {c}: {receiver_csv}")
    for c in ("src_cell_idx", "dst_spot_idx"):
        if c not in at_df.columns:
            raise KeyError(f"at_csv missing {c}: {at_csv}")
    for c in ("src_spot_idx", "dst_spot_idx"):
        if c not in nei_df.columns:
            raise KeyError(f"nei_csv missing {c}: {nei_csv}")
    for c in ("spot", "lr_idx"):
        if c not in lr_spot_pairs.columns:
            raise KeyError("lr_spot_pairs missing spot/lr_idx after rename")

    # n_spots from all
    n_spots = int(max(
        int(nei_df[["src_spot_idx", "dst_spot_idx"]].to_numpy().max()),
        int(at_df["dst_spot_idx"].max()),
        int(lr_spot_pairs["spot"].max()),
    )) + 1

    # cell->spot topk probs
    prob_col = at_prob_col if at_prob_col in at_df.columns else None
    if prob_col is None:
        for c in ("weight", "prob", "row_softmax", "col_softmax"):
            if c in at_df.columns:
                prob_col = c
                break
    if prob_col is None:
        at_df["prob"] = 1.0
    else:
        at_df["prob"] = pd.to_numeric(at_df[prob_col], errors="coerce").fillna(0.0)

    at_df = at_df.sort_values(["src_cell_idx", "prob"], ascending=[True, False])
    at_top = at_df.groupby("src_cell_idx").head(int(at_topk)).reset_index(drop=True)

    cell2spots: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cid, grp in at_top.groupby("src_cell_idx"):
        spots = grp["dst_spot_idx"].to_numpy(np.int32)
        probs = grp["prob"].to_numpy(np.float32)
        s = float(probs.sum())
        if s > 0:
            probs = probs / s
        cell2spots[int(cid)] = (spots, probs)

    # LR->spot topk and normalize
    lr_spot_pairs["lr_idx"] = pd.to_numeric(lr_spot_pairs["lr_idx"], errors="coerce")
    lr_spot_pairs["spot"]   = pd.to_numeric(lr_spot_pairs["spot"], errors="coerce")
    lr_spot_pairs = lr_spot_pairs.dropna(subset=["lr_idx","spot"]).copy()
    lr_spot_pairs["lr_idx"] = lr_spot_pairs["lr_idx"].astype(int)
    lr_spot_pairs["spot"]   = lr_spot_pairs["spot"].astype(int)

    lr_spot_pairs = lr_spot_pairs.sort_values(["lr_idx", "w"], ascending=[True, False])
    lr_spot_top = lr_spot_pairs.groupby("lr_idx").head(int(lrspot_topk)).reset_index(drop=True)

    w99 = lr_spot_top.groupby("lr_idx")["w"].quantile(0.99).rename("w99")
    lr_spot_top = lr_spot_top.merge(w99, on="lr_idx", how="left")
    lr_spot_top["w_norm"] = (lr_spot_top["w"] / lr_spot_top["w99"].replace(0, 1.0)).clip(0, 1)

    lr2spots: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for lr, sub in lr_spot_top.groupby("lr_idx"):
        lr2spots[int(lr)] = (
            sub["spot"].to_numpy(np.int32),
            sub["w_norm"].to_numpy(np.float32),
        )

    # sender/receiver maps (by lr_idx)
    send = s_df[["src_cell_idx", "dst_lr_idx"]].drop_duplicates()
    recv = r_df[["src_lr_idx", "dst_cell_idx"]].drop_duplicates()

    send["dst_lr_idx"] = pd.to_numeric(send["dst_lr_idx"], errors="coerce")
    recv["src_lr_idx"] = pd.to_numeric(recv["src_lr_idx"], errors="coerce")
    send = send.dropna(subset=["dst_lr_idx"]).copy()
    recv = recv.dropna(subset=["src_lr_idx"]).copy()
    send["dst_lr_idx"] = send["dst_lr_idx"].astype(int)
    recv["src_lr_idx"] = recv["src_lr_idx"].astype(int)

    lr2send = {int(lr): g["src_cell_idx"].to_numpy(np.int32) for lr, g in send.groupby("dst_lr_idx")}
    lr2recv = {int(lr): g["dst_cell_idx"].to_numpy(np.int32) for lr, g in recv.groupby("src_lr_idx")}

    usable = sorted(set(lr2send) & set(lr2recv) & set(lr2spots))
    # clip by embedding length (critical)
    usable = [lr for lr in usable if 0 <= lr < e_lr.shape[0]]
    print(f"[rank] usable LR idx (sender∩receiver∩spots) = {len(usable)}")

    # kernel build if requested
    K = None
    if relay_mode == "kernel":
        if e_spot is None:
            print("[warn] relay_mode=kernel but emb_spot missing -> fallback to none")
            relay_mode = "none"
        else:
            sigma_eff = None if float(sigma) <= 0 else float(sigma)
            K = build_sparse_gaussian_kernel_from_embeddings(
                e_spot.astype(np.float32, copy=False),
                k=int(knn),
                sigma=sigma_eff,
                self_loop=True,
            )

    def form_R(lr_idx: int) -> np.ndarray:
        src_spots, src_w = lr2spots[lr_idx]
        v = np.zeros(n_spots, dtype=np.float32)
        if src_spots.size > 0:
            v[src_spots] = np.maximum(v[src_spots], src_w)
        if relay_mode == "kernel" and K is not None:
            steps = max(1, int(embed_smooth_steps))
            v = smooth_vec(K, v, steps=steps)
            v = np.clip(v, 0.0, 1.0)
        return v

    lr_rows = []
    dot_temp = float(dot_temp)

    for lr in usable:
        R = form_R(lr)

        send_cells = lr2send.get(lr, np.zeros((0,), dtype=np.int32))
        recv_cells = lr2recv.get(lr, np.zeros((0,), dtype=np.int32))
        if send_cells.size == 0 or recv_cells.size == 0:
            continue

        lr_vec = e_lr[lr].astype(np.float32, copy=False)

        base_s = (e_cell[send_cells] * lr_vec[None, :]).sum(-1)
        base_s = stable_sigmoid(base_s, temp=dot_temp).astype(np.float32)
        sp_s = np.zeros_like(base_s, dtype=np.float32)
        for i, cid in enumerate(send_cells):
            m = cell2spots.get(int(cid), None)
            if m is None:
                continue
            spots, probs = m
            if spots.size:
                sp_s[i] = float(np.max(probs * R[spots]))
        score_s = base_s * sp_s

        base_t = (lr_vec[None, :] * e_cell[recv_cells]).sum(-1)
        base_t = stable_sigmoid(base_t, temp=dot_temp).astype(np.float32)
        sp_t = np.zeros_like(base_t, dtype=np.float32)
        for j, cid in enumerate(recv_cells):
            m = cell2spots.get(int(cid), None)
            if m is None:
                continue
            spots, probs = m
            if spots.size:
                sp_t[j] = float(np.max(probs * R[spots]))
        score_t = base_t * sp_t

        if score_s.size == 0 or score_t.size == 0:
            continue

        topS = min(int(perlr_topS), int(score_s.size))
        topT = min(int(perlr_topT), int(score_t.size))
        s_idx = topk_indices_desc(score_s, topS)
        t_idx = topk_indices_desc(score_t, topT)

        s_keep = score_s[s_idx]
        t_keep = score_t[t_idx]

        outer = np.outer(s_keep, t_keep).ravel()
        M = min(int(perlr_topM), int(outer.size))
        if M <= 0:
            continue
        if outer.size > M:
            topm_idx = np.argpartition(-outer, M - 1)[:M]
            topm_scores = outer[topm_idx]
        else:
            topm_scores = outer

        triad_max = float(np.max(topm_scores)) if topm_scores.size else 0.0
        triad_mean = float(np.mean(topm_scores)) if topm_scores.size else 0.0

        lr_rows.append((int(lr), triad_max, triad_mean, int(topm_scores.size)))

    lr_rank = pd.DataFrame(lr_rows, columns=["lr_idx", "triad_max", "triad_mean", "n_triad_kept"])
    lr_rank = lr_rank.sort_values("triad_max", ascending=False).reset_index(drop=True)

    out_lr = out_round_dir / "lr_pair_ranks.for_des.spatial.csv"
    lr_rank.to_csv(out_lr, index=False)
    print(f"[save] {out_lr} | rows={len(lr_rank)}")

    return lr_rank, usable


# ------------------ edge filtering ------------------

def filter_edges_by_lr(
    sender_csv: Path,
    receiver_csv: Path,
    present_csv: Path,
    bind_spot_csv: Path,
    keep_lr: List[int],
    out_dir: Path,
) -> Dict[str, Path]:
    ensure_dir(out_dir)
    keep = set(int(x) for x in keep_lr)

    S = read_csv_any(sender_csv)
    R = read_csv_any(receiver_csv)

    S2 = S[pd.to_numeric(S["dst_lr_idx"], errors="coerce").fillna(-1).astype(int).isin(keep)].copy()
    R2 = R[pd.to_numeric(R["src_lr_idx"], errors="coerce").fillna(-1).astype(int).isin(keep)].copy()

    out_sender = out_dir / "edges_cell_secrete_lr.filtered.csv"
    out_recv   = out_dir / "edges_lr_bind_cell.filtered.csv"
    S2.to_csv(out_sender, index=False)
    R2.to_csv(out_recv, index=False)

    out_map = {"sender": out_sender, "receiver": out_recv}

    if present_csv.exists():
        P = read_csv_any(present_csv)
        P2 = P[pd.to_numeric(P["dst_lr_idx"], errors="coerce").fillna(-1).astype(int).isin(keep)].copy()
        out_present = out_dir / "edges_spot_present_lr.filtered.csv"
        P2.to_csv(out_present, index=False)
        out_map["present"] = out_present

    if bind_spot_csv.exists():
        B = read_csv_any(bind_spot_csv)
        B2 = B[pd.to_numeric(B["src_lr_idx"], errors="coerce").fillna(-1).astype(int).isin(keep)].copy()
        out_bindspot = out_dir / "edges_lr_bind_spot.filtered.csv"
        B2.to_csv(out_bindspot, index=False)
        out_map["bind_spot"] = out_bindspot

    print(f"[filter] sender   {len(S)} -> {len(S2)}")
    print(f"[filter] receiver {len(R)} -> {len(R2)}")
    if "present" in out_map:
        print(f"[filter] present  {len(P)} -> {len(P2)}")
    if "bind_spot" in out_map:
        print(f"[filter] bind_spot {len(B)} -> {len(B2)}")

    return out_map


# ------------------ train runner ------------------

def run_train_one_round(args, round_i: int, edges_in: Dict[str, Path]) -> Path:
    """
    Call external training script, return the embedding dir (out_dir).
    """
    base = Path(args.base).resolve()
    out_dir = base / "hgt_graph" / f"{args.model_prefix}_round{round_i}"
    ensure_dir(out_dir)

    cell_h5ad = pjoin(base, args.cell_h5ad)
    spot_h5ad = pjoin(base, args.spot_h5ad)
    lr_h5ad   = pjoin(base, args.lr_h5ad)

    at_csv  = pjoin(base, args.at_csv)
    nei_csv = pjoin(base, args.nei_csv)

    spdist1 = pjoin(base, args.spdist1_csv) if args.spdist1_csv else None
    spdist2 = pjoin(base, args.spdist2_csv) if args.spdist2_csv else None
    spdist3 = pjoin(base, args.spdist3_csv) if args.spdist3_csv else None

    train_script = Path(args.train_script).resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"train_script not found: {train_script}")

    secrete_csv   = edges_in["sender"]
    bind_csv      = edges_in["receiver"]
    present_csv   = edges_in.get("present", pjoin(base, args.present_csv))
    bind_spot_csv = edges_in.get("bind_spot", pjoin(base, args.bind_spot_csv))

    cmd = [
        "python", str(train_script),
        "--base", str(base),
        "--cell_h5ad", str(cell_h5ad),
        "--spot_h5ad", str(spot_h5ad),
        "--lr_h5ad",   str(lr_h5ad),
        "--secrete_csv", str(secrete_csv),
        "--bind_csv",    str(bind_csv),
        "--present_csv", str(present_csv),
        "--bind_spot_csv", str(bind_spot_csv),
        "--at_csv",      str(at_csv),
        "--nei_csv",     str(nei_csv),
        "--out_dir",     str(out_dir),
        "--epochs_pre",  str(args.epochs_pre),
        "--epochs_ft",   str(args.epochs_ft),
        "--batch_size",  str(args.batch_size),
        "--fanout",      str(args.fanout),
        "--hops",        str(args.hops),
        "--hidden",      str(args.hidden),
        "--out_dim",     str(args.out_dim),
        "--heads",       str(args.heads),
        "--layers",      str(args.layers),
        "--dropout",     str(args.dropout),
    ]

    if args.amp:
        cmd.append("--amp")
    if args.train_rels:
        cmd += ["--train_rels", str(args.train_rels)]

    # aux pass-through
    if args.block_zscore:
        cmd.append("--block_zscore")
    if args.focal:
        cmd.append("--focal")
        cmd += ["--focal_alpha", str(args.focal_alpha), "--focal_gamma", str(args.focal_gamma)]
    if args.lambda_aux > 0:
        cmd += [
            "--lambda_aux", str(args.lambda_aux),
            "--lambda_aux_path", str(args.lambda_aux_path),
            "--lambda_aux_tf", str(args.lambda_aux_tf),
        ]

    if spdist1 and spdist1.exists():
        cmd += ["--spdist1_csv", str(spdist1)]
    if spdist2 and spdist2.exists():
        cmd += ["--spdist2_csv", str(spdist2)]
    if spdist3 and spdist3.exists():
        cmd += ["--spdist3_csv", str(spdist3)]

    run_cmd(cmd)
    return out_dir


# ------------------ main loop ------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", required=True)
    ap.add_argument("--train_script", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model_prefix", type=str, default="hgt_xformer_structbias")

    # features
    ap.add_argument("--cell_h5ad", required=True)
    ap.add_argument("--spot_h5ad", required=True)
    ap.add_argument("--lr_h5ad",   required=True)

    # initial edges
    ap.add_argument("--secrete_csv", required=True)
    ap.add_argument("--bind_csv",    required=True)
    ap.add_argument("--present_csv", required=True)
    ap.add_argument("--bind_spot_csv", required=True)
    ap.add_argument("--at_csv", required=True)
    ap.add_argument("--nei_csv", required=True)
    ap.add_argument("--spdist1_csv", default="")
    ap.add_argument("--spdist2_csv", default="")
    ap.add_argument("--spdist3_csv", default="")

    # train hyperparams
    ap.add_argument("--epochs_pre", type=int, default=3)
    ap.add_argument("--epochs_ft",  type=int, default=20)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--fanout", type=int, default=8)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out_dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--train_rels", type=str, default="secrete,bind,present,at")

    # ✅ aux / focal / zscore pass-through
    ap.add_argument("--block_zscore", action="store_true")
    ap.add_argument("--lambda_aux", type=float, default=0.0)
    ap.add_argument("--lambda_aux_path", type=float, default=1.0)
    ap.add_argument("--lambda_aux_tf", type=float, default=1.0)
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    # loop keep policy
    ap.add_argument("--keep_fraction", type=float, default=0.95)
    ap.add_argument("--keep_metric", choices=["triad_max", "triad_mean"], default="triad_max")

    # spatial ranking params
    ap.add_argument("--relay_mode", choices=["kernel", "none"], default="kernel")
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--embed_smooth_steps", type=int, default=1)
    ap.add_argument("--at_prob_col", type=str, default="prob_tangram")
    ap.add_argument("--at_topk", type=int, default=10)
    ap.add_argument("--lrspot_topk", type=int, default=50)
    ap.add_argument("--perlr_topS", type=int, default=200)
    ap.add_argument("--perlr_topT", type=int, default=200)
    ap.add_argument("--perlr_topM", type=int, default=1500)
    ap.add_argument("--dot_temp", type=float, default=3.0)

    args = ap.parse_args()
    base = Path(args.base).resolve()

    edges_in = {
        "sender":   pjoin(base, args.secrete_csv),
        "receiver": pjoin(base, args.bind_csv),
        "present":  pjoin(base, args.present_csv),
        "bind_spot":pjoin(base, args.bind_spot_csv),
    }
    for k, p in edges_in.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"missing {k} edge: {p}")

    for r in range(1, int(args.rounds) + 1):
        print("\n" + "=" * 10 + f" ROUND {r} " + "=" * 10, flush=True)

        # 1) train
        emb_dir = run_train_one_round(args, r, edges_in)

        # 2) spatial LR ranking (use current round edges)
        out_round_dir = base / "edges" / "qc_eval_rankreport" / f"round{r}"
        ensure_dir(out_round_dir)

        present_for_rank = edges_in.get("present", pjoin(base, args.present_csv))
        bindspot_for_rank = edges_in.get("bind_spot", pjoin(base, args.bind_spot_csv))

        lr_rank, usable = spatial_rank_lrs(
            base=base,
            emb_dir=emb_dir,
            sender_csv=edges_in["sender"],
            receiver_csv=edges_in["receiver"],
            present_csv=present_for_rank,
            bind_spot_csv=bindspot_for_rank,
            at_csv=pjoin(base, args.at_csv),
            nei_csv=pjoin(base, args.nei_csv),
            out_round_dir=out_round_dir,
            relay_mode=args.relay_mode,
            knn=args.knn,
            sigma=args.sigma,
            embed_smooth_steps=args.embed_smooth_steps,
            at_prob_col=args.at_prob_col,
            at_topk=args.at_topk,
            lrspot_topk=args.lrspot_topk,
            perlr_topS=args.perlr_topS,
            perlr_topT=args.perlr_topT,
            perlr_topM=args.perlr_topM,
            dot_temp=args.dot_temp,
        )

        if lr_rank.empty:
            print("[stop] lr_rank empty -> break.")
            break

        metric = args.keep_metric
        total = len(lr_rank)
        keep_n = int(math.ceil(total * float(args.keep_fraction)))
        keep_n = max(1, min(total, keep_n))

        keep_lr = lr_rank.sort_values(metric, ascending=False)["lr_idx"].head(keep_n).astype(int).tolist()
        print(f"[round {r}] LR total={total}, keep={keep_n} (keep_fraction={args.keep_fraction})")

        keep_txt = out_round_dir / "kept_lr_ids.txt"
        keep_txt.write_text("\n".join(str(x) for x in keep_lr) + "\n", encoding="utf-8")
        print(f"[save] {keep_txt}")

        # 3) build next round edges
        if r < int(args.rounds):
            next_edge_dir = base / "hgt_graph" / "edges" / f"iter_round{r+1}"
            out_map = filter_edges_by_lr(
                sender_csv=edges_in["sender"],
                receiver_csv=edges_in["receiver"],
                present_csv=present_for_rank,
                bind_spot_csv=bindspot_for_rank,
                keep_lr=keep_lr,
                out_dir=next_edge_dir,
            )
            edges_in = {
                "sender": out_map["sender"],
                "receiver": out_map["receiver"],
                "present": out_map.get("present", present_for_rank),
                "bind_spot": out_map.get("bind_spot", bindspot_for_rank),
            }
            print(f"[round {r}] build edges for round {r+1} -> {next_edge_dir}")

    print("\n[done] loop finished.")

if __name__ == "__main__":
    main()
