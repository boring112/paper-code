#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build sender-side edges: cell -> LR (per-cell)
- 群体占比阈值: --min-detect-l （默认 0.20）
- 单细胞聚合（复合体）阈: --per-cell-expr-cutoff （默认 0.20）
- 复合体聚合: --require-all-subunits 时取 min-subunit，否则取 max-subunit
- 可选：按群内分位筛最强细胞 --top-cell-quantile （默认 0 关闭）
- 可选：对极小群体剔除 --min-cells-per-type
- 可选：只保留 LR 词典里 is_ccc_usable=1 的条目（开关 --require-ccc-usable）

输出:
  1) --out-csv
     默认列: src_cell_idx, dst_lr_idx, weight
  2) 自动额外输出（可关）: *.with_lrname.PAIRNAME.csv
     增加列: lr, pair_name, lr_id, lr_key
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from tqdm import tqdm


def split_genes(s):
    toks = re.split(r"[^A-Za-z0-9_+-]+", str(s or ""))
    return [t for t in toks if t]


def pick_col(cols, cands):
    low = [c.lower() for c in cols]
    for k in cands:
        if k in low:
            return cols[low.index(k)]
    return None


def _mk_pair_name_from_obs(obs: pd.DataFrame) -> pd.Series:
    """
    优先使用 obs['pair_name']；
    否则尝试 ligand/receptor_subunits 拼接；
    再否则 L_subunits/R_subunits；
    再否则 L_genes/R_genes；
    最后 fallback: LRxxxxx
    """
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

    pn = pn.fillna("").astype("string")
    # 统一大小写（你的 CellChat/过滤脚本一般都用大写 LR）
    pn = pn.str.upper()
    return pn


def _vector_add_lr_meta(df_edges: pd.DataFrame, lr_obs: pd.DataFrame, lr_idx_col: str) -> pd.DataFrame:
    """
    df_edges: 必须有 lr_idx_col (dst_lr_idx)
    lr_obs:   ad_lr.obs
    返回：增加 lr/pair_name/lr_id/lr_key 列（向量化索引）
    """
    pn = _mk_pair_name_from_obs(lr_obs).to_numpy()

    lr_id = lr_obs["lr_id"].astype("string").fillna("").astype("string").to_numpy() if "lr_id" in lr_obs.columns \
        else np.array([""] * lr_obs.shape[0], dtype=object)
    lr_key = lr_obs["lr_key"].astype("string").fillna("").astype("string").to_numpy() if "lr_key" in lr_obs.columns \
        else np.array([""] * lr_obs.shape[0], dtype=object)

    idx = df_edges[lr_idx_col].to_numpy(dtype=np.int64, copy=False)

    out = df_edges.copy()
    out["pair_name"] = pd.Series(pn[idx], index=out.index, dtype="string")
    out["lr"] = out["pair_name"]  # 兼容你后续过滤脚本常用列名
    out["lr_id"] = pd.Series(lr_id[idx], index=out.index, dtype="string")
    out["lr_key"] = pd.Series(lr_key[idx], index=out.index, dtype="string")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrna-h5ad", required=True)
    ap.add_argument("--cell-h5ad", required=True)
    ap.add_argument("--lr-h5ad", required=True)

    ap.add_argument("--celltype-key", default="cell_type")
    ap.add_argument("--min-detect-l", type=float, default=0.20)
    ap.add_argument("--per-cell-expr-cutoff", type=float, default=0.20)
    ap.add_argument("--top-cell-quantile", type=float, default=0.0, help="0 代表关闭")
    ap.add_argument("--require-all-subunits", action="store_true")
    ap.add_argument("--min-cells-per-type", type=int, default=50)

    ap.add_argument("--lr-dict-csv", default="", help="可选，用于 is_ccc_usable 过滤")
    ap.add_argument("--require-ccc-usable", action="store_true")

    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--no-write-with-lrname", action="store_true",
                    help="默认会额外写出 *.with_lrname.PAIRNAME.csv；加此参数关闭")
    ap.add_argument("--out-with-lrname", default="",
                    help="可选：手动指定 with_lrname 文件路径；不填则自动在 out-csv 旁边生成")

    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ============== 读数据并对齐 ==============
    ad_sc = ad.read_h5ad(args.scrna_h5ad)
    ad_cell = ad.read_h5ad(args.cell_h5ad)
    ad_lr = ad.read_h5ad(args.lr_h5ad)

    if not np.array_equal(ad_cell.obs_names.values, ad_sc.obs_names.values):
        ad_sc = ad_sc[ad_cell.obs_names, :].copy()

    if args.celltype_key not in ad_cell.obs.columns:
        raise KeyError(f"cell.obs 缺少列: {args.celltype_key}")

    cell_type = ad_cell.obs[args.celltype_key].astype(str).to_numpy()

    # 剔除小群体（但注意：一旦剔除，cell idx 会变成“子集索引”）
    ct, cnt = np.unique(cell_type, return_counts=True)
    kept = set(ct[cnt >= args.min_cells_per_type])
    keep_mask = np.array([c in kept for c in cell_type], dtype=bool)

    ad_sc = ad_sc[keep_mask, :].copy()
    cell_type = cell_type[keep_mask]

    # LR: ligand_subunits & pair_name
    obs_lr = ad_lr.obs.copy()
    lig_col = pick_col(obs_lr.columns, ["ligand_subunits", "ligand_genes", "ligand", "l_genes", "l"])
    if not lig_col:
        raise KeyError(f"lr_h5ad.obs 缺 ligand_* 列; 可用列={obs_lr.columns.tolist()}")

    LR = ad_lr.n_obs
    lig_list = [split_genes(obs_lr[lig_col].astype(str).iloc[i]) for i in range(LR)]

    # 可选：从 LR 字典拿 is_ccc_usable 过滤
    ok_lr_mask = None
    if args.lr_dict_csv and Path(args.lr_dict_csv).exists() and args.require_ccc_usable:
        d = pd.read_csv(args.lr_dict_csv)
        key = "is_ccc_usable" if "is_ccc_usable" in d.columns else None
        if key is not None:
            if "lr_key" in d.columns and "lr_id" in obs_lr.columns:
                ok = set(d.loc[d[key] == 1, "lr_key"].astype(str))
                ok_lr_mask = np.array(
                    [(str(obs_lr["lr_id"].iloc[i]) in ok) or (("lr_key" in obs_lr.columns) and (str(obs_lr["lr_key"].iloc[i]) in ok))
                     for i in range(LR)],
                    dtype=bool
                )
            else:
                ok_lr_mask = np.array(d[key].values[:LR], dtype=bool)

    # 基因索引与表达矩阵切片
    var2idx = {str(g).upper(): i for i, g in enumerate(map(str, ad_sc.var_names))}
    needed = sorted({g.upper() for sub in lig_list for g in sub})
    idx = np.array([var2idx.get(g, -1) for g in needed], dtype=int)
    ok = idx >= 0
    needed = [g for g, m in zip(needed, ok) if m]
    idx = idx[ok]
    gene2col = {g: i for i, g in enumerate(needed)}

    X = ad_sc.X
    is_sparse = sp.issparse(X)
    E = X[:, idx].toarray().astype(np.float32) if is_sparse else X[:, idx].astype(np.float32)

    # 群体索引
    ct_vals = np.array(sorted(list(kept)))
    ct2idx = {c: i for i, c in enumerate(ct_vals)}
    ct_idx = np.array([ct2idx[c] for c in cell_type], dtype=np.int32)
    n_types = len(ct_vals)

    # 预先把每个 cell_type 的细胞编号列出来
    ids_by_type = {ti: np.where(ct_idx == ti)[0] for ti in range(n_types)}

    # ============== 逐 LR 构建 per-cell 边 ==============
    rows_out = []
    min_detect = float(args.min_detect_l)
    cutoff = float(args.per_cell_expr_cutoff)
    q = float(args.top_cell_quantile)

    print("[build] Cell->LR edges:")
    for k in tqdm(range(LR)):
        if ok_lr_mask is not None and not ok_lr_mask[k]:
            continue

        rows = [gene2col.get(g.upper(), None) for g in lig_list[k]]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue

        sub = E[:, np.array(rows, dtype=int)]
        agg = np.min(sub, axis=1) if args.require_all_subunits else np.max(sub, axis=1)

        for ti in range(n_types):
            ids = ids_by_type[ti]
            v = agg[ids]

            detect_mask = (v > 0) if cutoff <= 0 else (v >= cutoff)
            detect_frac = float(detect_mask.mean()) if ids.size > 0 else 0.0
            if detect_frac < min_detect:
                continue

            keep_ids = ids.copy()
            if q > 0:
                thr = float(np.quantile(v, q))
                keep_ids = keep_ids[v >= thr]

            if cutoff > 0:
                keep_ids = keep_ids[agg[keep_ids] >= cutoff]

            if keep_ids.size == 0:
                continue

            for ci in keep_ids:
                rows_out.append((ci, k, float(agg[ci])))

    if not rows_out:
        pd.DataFrame(columns=["src_cell_idx", "dst_lr_idx", "weight"]).to_csv(out_csv, index=False)
        print("[save]", out_csv, "| rows=0")
        return

    df = pd.DataFrame(rows_out, columns=["src_cell_idx", "dst_lr_idx", "weight"])
    df.to_csv(out_csv, index=False)
    print(f"[save] {out_csv} | rows={len(df)} | uniqueLR={df['dst_lr_idx'].nunique()}")

    # ============== 额外写出 with_lrname 文件 ==============
    if not args.no_write_with_lrname:
        if args.out_with_lrname:
            out_with = Path(args.out_with_lrname)
        else:
            out_with = out_csv.with_name(out_csv.stem + ".with_lrname.PAIRNAME.csv")

        df2 = _vector_add_lr_meta(df, ad_lr.obs, lr_idx_col="dst_lr_idx")
        df2.to_csv(out_with, index=False)
        print(f"[save] {out_with} | with lr/pair_name/lr_id/lr_key")

    # ============== 简单诊断 ==============
    pair_name = _mk_pair_name_from_obs(ad_lr.obs)
    lig_counts = df.groupby("dst_lr_idx")["src_cell_idx"].count().sort_values(ascending=False)

    print("\n[top LR by edges]")
    for lr_idx, cnt in lig_counts.head(15).items():
        name = pair_name.iloc[lr_idx] if lr_idx < len(pair_name) else f"LR{lr_idx:05d}"
        print(f"{str(name):>24s}  {cnt}")

    hit_ct = df.merge(
        pd.DataFrame({"cell_idx": np.arange(len(cell_type)), "cell_type": cell_type}),
        left_on="src_cell_idx", right_on="cell_idx", how="left"
    ).groupby("dst_lr_idx")["cell_type"].nunique()

    broad = int((hit_ct >= max(1, n_types // 2)).sum())
    print(f"\n[diagnostic] broad LR count = {broad} / {hit_ct.shape[0]} (threshold = half cell types)")


if __name__ == "__main__":
    main()
