#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 receiver 侧边：LR -> Cell
- 复合体聚合：min-subunit（--require-all-subunits 不开则用 max）
- 群体阈值：按 cell_type 的检测率 >= --min-detect-r
- 单细胞阈值：per-cell 聚合表达 >= --per-cell-expr-cutoff
- 分位筛选：--top-cell-quantile>0 时，对每个 (LR, cell_type) 仅保留群内分位以上的细胞
- 输出：src_lr_idx, dst_cell_idx, weight

同时默认额外输出：*.with_lrname.PAIRNAME.csv
增加列: lr, pair_name, lr_id, lr_key
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from tqdm import tqdm


def split_genes(s: str):
    toks = re.split(r"[^A-Za-z0-9_+-]+", str(s or ""))
    return [t for t in toks if t]


def pick_col(obs: pd.DataFrame, cands):
    low = [c.lower() for c in obs.columns]
    for k in cands:
        if k in low:
            return obs.columns[low.index(k)]
    return None


def _mk_pair_name_from_obs(obs: pd.DataFrame) -> pd.Series:
    cols = {c.lower(): c for c in obs.columns}

    def has(*names):
        return all(n in cols for n in names)

    if "pair_name" in obs.columns:
        pn = obs["pair_name"].astype("string")
    elif has("ligand_subunits", "receptor_subunits"):
        pn = (obs[cols["ligand_subunits"]].astype("string")
              + "__" + obs[cols["receptor_subunits"]].astype("string"))
    elif has("l_subunits", "r_subunits"):
        pn = (obs[cols["l_subunits"]].astype("string")
              + "__" + obs[cols["r_subunits"]].astype("string"))
    elif has("l_genes", "r_genes"):
        pn = (obs[cols["l_genes"]].astype("string")
              + "__" + obs[cols["r_genes"]].astype("string"))
    else:
        pn = pd.Series([f"LR{i:05d}" for i in range(obs.shape[0])], dtype="string")

    pn = pn.fillna("").astype("string").str.upper()
    return pn


def _vector_add_lr_meta(df_edges: pd.DataFrame, lr_obs: pd.DataFrame, lr_idx_col: str) -> pd.DataFrame:
    pn = _mk_pair_name_from_obs(lr_obs).to_numpy()

    lr_id = lr_obs["lr_id"].astype("string").fillna("").astype("string").to_numpy() if "lr_id" in lr_obs.columns \
        else np.array([""] * lr_obs.shape[0], dtype=object)
    lr_key = lr_obs["lr_key"].astype("string").fillna("").astype("string").to_numpy() if "lr_key" in lr_obs.columns \
        else np.array([""] * lr_obs.shape[0], dtype=object)

    idx = df_edges[lr_idx_col].to_numpy(dtype=np.int64, copy=False)

    out = df_edges.copy()
    out["pair_name"] = pd.Series(pn[idx], index=out.index, dtype="string")
    out["lr"] = out["pair_name"]
    out["lr_id"] = pd.Series(lr_id[idx], index=out.index, dtype="string")
    out["lr_key"] = pd.Series(lr_key[idx], index=out.index, dtype="string")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrna-h5ad", required=True)
    ap.add_argument("--cell-h5ad", required=True)
    ap.add_argument("--lr-h5ad", required=True)
    ap.add_argument("--celltype-key", default="cell_type")

    ap.add_argument("--min-detect-r", type=float, default=0.20)
    ap.add_argument("--per-cell-expr-cutoff", type=float, default=0.0)
    ap.add_argument("--require-all-subunits", action="store_true")
    ap.add_argument("--min-cells-per-type", type=int, default=25)
    ap.add_argument("--top-cell-quantile", type=float, default=0.0,
                    help="每个(LR, cell_type)仅保留该群内分位以上的细胞；0=关闭，典型 0.7~0.85")

    ap.add_argument("--lr-dict-csv", default=None)
    ap.add_argument("--require-ccc-usable", action="store_true")

    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--no-write-with-lrname", action="store_true",
                    help="默认会额外写出 *.with_lrname.PAIRNAME.csv；加此参数关闭")
    ap.add_argument("--out-with-lrname", default="",
                    help="可选：手动指定 with_lrname 文件路径；不填则自动在 out-csv 旁边生成")

    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ============== 载入并对齐 ==============
    ad_sc = ad.read_h5ad(args.scrna_h5ad)
    ad_cell = ad.read_h5ad(args.cell_h5ad)
    ad_lr = ad.read_h5ad(args.lr_h5ad)

    if not np.array_equal(ad_cell.obs_names.values, ad_sc.obs_names.values):
        ad_sc = ad_sc[ad_cell.obs_names, :].copy()

    if args.celltype_key not in ad_cell.obs.columns:
        raise KeyError(f"cell_h5ad.obs 缺少列: {args.celltype_key}")

    cell_type = ad_cell.obs[args.celltype_key].astype(str).to_numpy()

    # 最小样本量过滤（注意：过滤后 cell idx 变成子集索引）
    ct_all, ct_counts = np.unique(cell_type, return_counts=True)
    valid_ct = set(ct_all[ct_counts >= args.min_cells_per_type])
    keep_mask = np.array([c in valid_ct for c in cell_type], dtype=bool)
    ad_sc = ad_sc[keep_mask, :].copy()
    cell_type = cell_type[keep_mask]

    ct_vals = np.array(sorted(list(valid_ct)))
    ct2idx = {c: i for i, c in enumerate(ct_vals)}
    ct_idx = np.array([ct2idx[c] for c in cell_type], dtype=np.int32)
    n_types = len(ct_vals)

    # LR 侧字段
    obs = ad_lr.obs.copy()
    rec_col = pick_col(obs, ["receptor_subunits", "receptor_genes", "receptor", "r_genes", "r"])
    if not rec_col:
        raise KeyError(f"lr_h5ad.obs 缺 receptor_* 列，当前列: {obs.columns.tolist()}")

    LR = ad_lr.n_obs
    rec_list = [split_genes(obs[rec_col].astype(str).iloc[i]) for i in range(LR)]

    # 可选：用 LR 字典限制 is_ccc_usable
    lr_keep_mask = np.ones(LR, dtype=bool)
    if args.lr_dict_csv:
        df_dict = pd.read_csv(args.lr_dict_csv)
        if args.require_ccc_usable and "is_ccc_usable" in df_dict.columns:
            if "lr_idx" in df_dict.columns:
                m = df_dict["is_ccc_usable"].astype(int).to_numpy().astype(bool)
                lr_keep_mask = m[:LR]
            elif "lr_key" in df_dict.columns and "lr_id" in ad_lr.obs.columns:
                m = df_dict.set_index("lr_key")["is_ccc_usable"].astype(int).to_dict()
                lr_keep_mask = np.array([bool(m.get(ad_lr.obs["lr_id"].iloc[i], 1)) for i in range(LR)], dtype=bool)

    # 仅计算所需基因
    needed_genes = sorted({g.upper() for sub in rec_list for g in sub})
    var2idx = {str(g).upper(): i for i, g in enumerate(map(str, ad_sc.var_names))}
    gene_idx = np.array([var2idx.get(g, -1) for g in needed_genes], dtype=np.int32)
    ok = gene_idx >= 0
    needed_genes = [g for g, m in zip(needed_genes, ok) if m]
    gene_idx = gene_idx[ok]
    gene2row = {g: i for i, g in enumerate(needed_genes)}

    X = ad_sc.X
    is_sparse = sp.issparse(X)
    E = X[:, gene_idx].toarray().astype(np.float32) if is_sparse else X[:, gene_idx].astype(np.float32)

    # ============== 1) gene×type 检测率 ==============
    tau = float(args.per_cell_expr_cutoff)
    detect_frac = np.zeros((len(needed_genes), n_types), dtype=np.float32)
    for c, ci in ct2idx.items():
        rows = np.where(ct_idx == ci)[0]
        if rows.size == 0:
            continue
        sub = E[rows, :]
        det = (sub > tau).astype(np.float32)
        detect_frac[:, ci] = det.mean(axis=0).astype(np.float32)

    # ============== 2) LR×type 是否达标 ==============
    R_pass_type = np.zeros((LR, n_types), dtype=bool)
    for k in range(LR):
        if not lr_keep_mask[k]:
            continue
        rows = [gene2row.get(g.upper(), None) for g in rec_list[k]]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        d = detect_frac[np.array(rows, dtype=int), :]
        if args.require_all_subunits:
            R_pass_type[k, :] = (d.min(axis=0) >= float(args.min_detect_r))
        else:
            R_pass_type[k, :] = (d.max(axis=0) >= float(args.min_detect_r))

    # ============== 3) 逐 LR 生成 per-cell 边 ==============
    out_rows = []
    q = float(args.top_cell_quantile)

    for k in tqdm(range(LR), desc="[build] LR->cell edges"):
        if not lr_keep_mask[k]:
            continue

        sub_idx = [gene2row.get(g.upper(), None) for g in rec_list[k]]
        sub_idx = [i for i in sub_idx if i is not None]
        if not sub_idx:
            continue

        subE = E[:, np.array(sub_idx, dtype=int)]
        agg = np.min(subE, axis=1) if args.require_all_subunits else np.max(subE, axis=1)

        pass_type = R_pass_type[k, ct_idx]
        pass_cell = (agg >= tau)
        mask = pass_type & pass_cell

        # (LR, cell_type) 内分位筛选
        if q > 0 and mask.any():
            keep_fine = np.zeros_like(mask)
            types_hit = np.unique(ct_idx[mask])
            for ti in types_hit:
                rows_ti = mask & (ct_idx == ti)
                thr_q = float(np.quantile(agg[rows_ti], q))
                thr_use = max(thr_q, tau)
                keep_fine |= rows_ti & (agg >= thr_use)
            mask = keep_fine

        if not np.any(mask):
            continue

        cells = np.where(mask)[0]
        weights = agg[cells]
        out_rows.append(pd.DataFrame({
            "src_lr_idx": k,
            "dst_cell_idx": cells.astype(np.int64),
            "weight": weights.astype(np.float32)
        }))

    if not out_rows:
        pd.DataFrame(columns=["src_lr_idx", "dst_cell_idx", "weight"]).to_csv(out_csv, index=False)
        print("[save]", out_csv, "| rows=0")
        return

    df_out = pd.concat(out_rows, ignore_index=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[save] {out_csv} | rows={len(df_out)} | uniqueLR={df_out['src_lr_idx'].nunique()}")

    # ============== 额外写出 with_lrname 文件 ==============
    if not args.no_write_with_lrname:
        if args.out_with_lrname:
            out_with = Path(args.out_with_lrname)
        else:
            out_with = out_csv.with_name(out_csv.stem + ".with_lrname.PAIRNAME.csv")

        df2 = _vector_add_lr_meta(df_out, ad_lr.obs, lr_idx_col="src_lr_idx")
        df2.to_csv(out_with, index=False)
        print(f"[save] {out_with} | with lr/pair_name/lr_id/lr_key")

    # 简报（可选）
    pair_name = _mk_pair_name_from_obs(ad_lr.obs)
    top_lr = df_out["src_lr_idx"].value_counts().head(15)
    print("\n[top LR by edges]")
    for lr_idx, cnt in top_lr.items():
        name = pair_name.iloc[int(lr_idx)] if int(lr_idx) < len(pair_name) else f"LR{int(lr_idx):05d}"
        print(f"{str(name):>24s}  {cnt}")


if __name__ == "__main__":
    main()
