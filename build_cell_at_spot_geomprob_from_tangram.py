#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Tangram 映射 + ST 坐标，构建几何增强版 cell→spot 边：

输出 CSV 列：
  src_cell_idx, dst_spot_idx,
  weight, prob_tangram,
  dx, dy, dist,
  row_softmax, col_softmax,
  rbf_d_0 ... rbf_d_5

- weight        : 筛选后按 cell 归一化的映射概率，cell 的所有出边加起来 = 1
- prob_tangram  : Tangram 原始概率（筛选前；此脚本会先做 row_normalize 再筛边）
- dx, dy, dist  : 几何偏置（spot 坐标 - cell 重心坐标）
- row_softmax   : 对同一 cell 的所有边按 -dist 做 softmax
- col_softmax   : 对同一 spot 的所有边按 -dist 做 softmax
- rbf_d_0..5    : 用 6 个自适应尺度的 RBF(dist) 形成的几何 basis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


def get_spot_xy(ad_sp):
    """优先从 obsm['spatial'] 取坐标，否则尝试常见 obs 列。"""
    if "spatial" in ad_sp.obsm.keys():
        xy = np.asarray(ad_sp.obsm["spatial"])
        if xy.ndim >= 2 and xy.shape[1] >= 2:
            return xy[:, :2].astype(float)
    for a, b in [("array_col", "array_row"), ("x", "y"), ("aligned_x", "aligned_y")]:
        if a in ad_sp.obs.columns and b in ad_sp.obs.columns:
            return ad_sp.obs[[a, b]].to_numpy(dtype=float)
    raise KeyError("在 ST h5ad 中找不到坐标（obsm['spatial'] 或 obs['array_col','array_row'] 等）")


def get_tangram_matrix(ad_map):
    """
    从 Tangram 输出里取 cell×spot 的概率矩阵：
    - 优先 ad_map.X
    - 或 obsm['tangram_probs']（如果你以后这么存）

    返回：np.ndarray(float32), shape=(cells, spots)

    说明：
    - 如果 X 是 scipy.sparse（CSR/CSC/COO 等），不要 np.asarray(X)，
      应用 X.toarray() / X.A 转成 dense ndarray，否则容易得到 object array 并报错。
    """
    if ad_map.X is not None:
        X = ad_map.X
    elif "tangram_probs" in ad_map.obsm_keys():
        X = ad_map.obsm["tangram_probs"]
    else:
        raise KeyError("Tangram h5ad 里既没有 X，也没有 obsm['tangram_probs']，不知道概率矩阵在哪。")

    if sp.issparse(X):
        # ✅ 正确：稀疏 -> dense ndarray
        X = X.tocsr().astype(np.float32).toarray()
    else:
        X = np.asarray(X, dtype=np.float32)

    # 防御：确保二维
    if X.ndim != 2:
        X = np.asarray(X, dtype=np.float32).reshape(ad_map.n_obs, ad_map.n_vars)

    return X


def row_normalize(mat):
    """按行归一化，使每行和为 1（有 0 行则保持 0）。支持 dense 或 sparse。"""
    if sp.issparse(mat):
        mat = mat.tocsr()
        sums = np.asarray(mat.sum(axis=1)).ravel()
        sums[sums == 0.0] = 1.0
        inv = 1.0 / sums
        D = sp.diags(inv)
        return (D @ mat).astype(np.float32)
    else:
        mat = np.asarray(mat, dtype=np.float32)
        sums = mat.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0
        return mat / sums


def softmax_by_group(ids, values, tau):
    """
    对每个 group（ids 相同的一组）内的 values 做 softmax(-value / tau)。
    返回和 values 同 shape 的 numpy 数组。
    """
    ids = np.asarray(ids)
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)

    df = pd.DataFrame({"id": ids, "v": values})
    for _, sub in df.groupby("id", sort=False):
        v = sub["v"].to_numpy(dtype=np.float32)
        if v.size == 0:
            continue
        x = -v / max(float(tau), 1e-6)
        x = x - x.max()
        e = np.exp(x)
        s = e.sum()
        if s <= 0:
            out[sub.index.to_numpy()] = 0.0
        else:
            out[sub.index.to_numpy()] = e / s
    return out


def build_rbf_features(dist, n_rbf=6):
    """
    根据所有 dist 的分位数自动选 n_rbf 个尺度，构造 RBF 特征：
      rbf_d_k = exp( - (dist / sigma_k)^2 )
    """
    dist = np.asarray(dist, dtype=np.float32)
    positive = dist[dist > 0]
    if positive.size == 0:
        return np.zeros((dist.shape[0], n_rbf), dtype=np.float32)

    qs = np.linspace(0.1, 0.9, n_rbf)
    sigmas = np.quantile(positive, qs)
    sigmas = np.maximum(sigmas, np.percentile(positive, 5))

    rbf = []
    for s in sigmas:
        s = float(s) + 1e-6
        rbf.append(np.exp(-(dist / s) ** 2))
    return np.stack(rbf, axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tangram-map-h5ad", required=True,
                    help="Tangram cell→spot 映射结果（cells×spots 概率矩阵）")
    ap.add_argument("--spot-h5ad", required=True,
                    help="ST h5ad，用于取 spot 坐标（obs_names 必须和 Tangram 的列一致）")
    ap.add_argument("--cell-h5ad", required=True,
                    help="cell_features.final.h5ad，用于确认 cell 顺序与数量（obs_names 必须和 Tangram 的行一致）")

    ap.add_argument("--min-prob", type=float, default=0.01,
                    help="筛选边时的最小 Tangram 概率（默认 0.01）")
    ap.add_argument("--topm-per-cell", type=int, default=3,
                    help="每个 cell 最多保留多少个 spot（默认 3）")

    ap.add_argument("--out-csv", required=True,
                    help="输出 CSV 路径，比如 edges_cell_at_spot.geomprob.csv")

    args = ap.parse_args()

    map_path = Path(args.tangram_map_h5ad)
    spot_path = Path(args.spot_h5ad)
    cell_path = Path(args.cell_h5ad)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[IO] reading tangram map:", map_path)
    ad_map = ad.read_h5ad(map_path)

    print("[IO] reading spot h5ad:", spot_path)
    ad_sp = ad.read_h5ad(spot_path)

    print("[IO] reading cell h5ad:", cell_path)
    ad_cell = ad.read_h5ad(cell_path)

    # -------- 对齐 cell / spot 顺序 --------
    if list(ad_map.obs_names) != list(ad_cell.obs_names):
        print("[align] reindex tangram rows to cell_h5ad.obs_names")
        ad_map = ad_map[ad_cell.obs_names, :].copy()

    if list(ad_map.var_names) != list(ad_sp.obs_names):
        print("[align] reindex tangram cols to spot_h5ad.obs_names")
        ad_map = ad_map[:, ad_sp.obs_names].copy()

    n_cells = ad_cell.n_obs
    n_spots = ad_sp.n_obs
    print(f"[info] cells={n_cells}, spots={n_spots}")

    # -------- Tangram 概率矩阵（✅ 已修复稀疏转换）--------
    P_raw = get_tangram_matrix(ad_map)          # dense float32 ndarray
    print("[info] P_raw shape:", P_raw.shape, "dtype:", P_raw.dtype)

    # 行归一化，避免数值漂移
    P_raw = row_normalize(P_raw)

    # backup 原始 prob（注意：此处是 row_normalize 后的概率）
    prob_tangram = P_raw.copy()

    # -------- 取 spot 坐标 & cell 重心坐标 --------
    XY_spot = get_spot_xy(ad_sp)               # (spots × 2)
    print("[info] spot XY shape:", XY_spot.shape)

    # cell 重心坐标 = P_raw * XY_spot
    cell_x = (P_raw @ XY_spot[:, 0]).astype(np.float32)
    cell_y = (P_raw @ XY_spot[:, 1]).astype(np.float32)

    # -------- 逐 cell 选 topM 边 --------
    src_list, dst_list = [], []
    w_list, prob_list = [], []
    dx_list, dy_list, dist_list = [], [], []

    min_prob = float(args.min_prob)
    topM = int(args.topm_per_cell)

    for i in range(n_cells):
        row = P_raw[i, :]  # 1D ndarray length=n_spots

        cand = np.where(row >= min_prob)[0]
        if cand.size == 0:
            j = int(row.argmax())
            cand = np.array([j], dtype=int)

        if cand.size > topM:
            idx_sort = np.argsort(-row[cand])[:topM]
            cand = cand[idx_sort]

        p_sel = row[cand]
        p_orig = prob_tangram[i, cand]

        s = float(p_sel.sum())
        if s > 0:
            w_sel = (p_sel / s).astype(np.float32)
        else:
            w_sel = np.zeros_like(p_sel, dtype=np.float32)

        cx, cy = cell_x[i], cell_y[i]
        sx = XY_spot[cand, 0]
        sy = XY_spot[cand, 1]
        dx = (sx - cx).astype(np.float32)
        dy = (sy - cy).astype(np.float32)
        dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        src_list.append(np.full(cand.shape[0], i, dtype=np.int32))
        dst_list.append(cand.astype(np.int32))
        w_list.append(w_sel)
        prob_list.append(p_orig.astype(np.float32))
        dx_list.append(dx)
        dy_list.append(dy)
        dist_list.append(dist)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    weight = np.concatenate(w_list)
    prob_t = np.concatenate(prob_list)
    dx = np.concatenate(dx_list)
    dy = np.concatenate(dy_list)
    dist = np.concatenate(dist_list)

    print(f"[edges] after filter: {len(src)} cell→spot edges")

    # -------- row_softmax / col_softmax --------
    if dist.size > 0:
        tau_row = float(np.median(dist))
        tau_col = tau_row
    else:
        tau_row = tau_col = 1.0

    row_soft = softmax_by_group(src, dist, tau=tau_row)
    col_soft = softmax_by_group(dst, dist, tau=tau_col)

    # -------- RBF(dist) 特征 --------
    rbf = build_rbf_features(dist, n_rbf=6)   # (E × 6)

    # -------- 组装 DataFrame --------
    df = pd.DataFrame({
        "src_cell_idx": src.astype(np.int32),
        "dst_spot_idx": dst.astype(np.int32),
        "weight":       weight.astype(np.float32),
        "prob_tangram": prob_t.astype(np.float32),
        "dx":           dx.astype(np.float32),
        "dy":           dy.astype(np.float32),
        "dist":         dist.astype(np.float32),
        "row_softmax":  row_soft.astype(np.float32),
        "col_softmax":  col_soft.astype(np.float32),
    })

    for k in range(rbf.shape[1]):
        df[f"rbf_d_{k}"] = rbf[:, k].astype(np.float32)

    df.to_csv(out_path, index=False)
    print(f"[save] {out_path} | rows={len(df)}")
    print("[done] cell→spot geomprob edges 完成。")


if __name__ == "__main__":
    main()
