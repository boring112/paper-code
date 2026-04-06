#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从现有结果直接构建 cell_features.final.h5ad

输入：
  1) scRNA h5ad（带 cell_type 等 obs 信息）
  2) cell_pathway_tf_acts.h5ad（上一步 decoupler_waggr_to_spot_cli.py 的输出）

步骤：
  - 按细胞 ID 对齐这两个 AnnData
  - 在 sc 上跑 PCA 得到 X_expr_pca
  - 根据 obs[cell_type_key] 生成 X_ct_onehot
  - 从 pathway_tf h5ad 里拷贝 X_pathway_progeny14 / X_tfact_dorothea
  - 把 4 块特征横向拼接到 .X，生成 cell_features.final.h5ad
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import warnings


def zscore_block(M):
    M = np.asarray(M, dtype=float)
    mu = np.nanmean(M, axis=0, keepdims=True)
    sd = np.nanstd(M, axis=0, keepdims=True) + 1e-8
    return (M - mu) / sd


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build cell_features.final.h5ad directly from scRNA + cell_pathway_tf_acts.h5ad"
    )
    ap.add_argument("--sc-h5ad", type=str, required=True,
                    help="单细胞 h5ad，例：PAAD_GSE111672_patientA_sc_p6.h5ad")
    ap.add_argument("--pathway-tf-h5ad", type=str, required=True,
                    help="cell_pathway_tf_acts.h5ad（decoupler WAGGR 的输出）")
    ap.add_argument("--out-h5ad", type=str, required=True,
                    help="输出：cell_features.final.h5ad")
    ap.add_argument("--csv-columns", type=str, default="",
                    help="列名清单 CSV（默认用 out-h5ad 改后缀生成）")
    ap.add_argument("--cell-type-key", type=str, default="cell_type",
                    help="obs 里表示细胞类型的列名（默认 cell_type）")
    ap.add_argument("--n-pcs", type=int, default=50,
                    help="PCA 维度（默认 50）")
    ap.add_argument("--center-onehot", action="store_true",
                    help="是否对 cell_type one-hot 做 z-score（默认 False）")

    return ap.parse_args()


def main():
    args = parse_args()

    SC_AD   = args.sc_h5ad
    PT_AD   = args.pathway_tf_h5ad
    OUT_H5  = args.out_h5ad
    CT_KEY  = args.cell_type_key
    N_PCS   = args.n_pcs
    CENTER_ONEHOT = args.center_onehot

    if args.csv_columns:
        CSV_FN = args.csv_columns
    else:
        base, ext = os.path.splitext(OUT_H5)
        CSV_FN = base + ".columns.csv"

    os.makedirs(os.path.dirname(OUT_H5), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_FN), exist_ok=True)

    # -------- 1) 读取输入 --------
    print("[info] read scRNA:", SC_AD)
    ad_sc = sc.read_h5ad(SC_AD)

    print("[info] read pathway/TF acts:", PT_AD)
    ad_pt = sc.read_h5ad(PT_AD)

    # -------- 2) 按 cell ID 对齐 --------
    sc_ids = ad_sc.obs_names.astype(str)
    pt_ids = ad_pt.obs_names.astype(str)
    common = sc_ids.intersection(pt_ids)

    if len(common) == 0:
        raise RuntimeError("scRNA 与 cell_pathway_tf_acts 没有共有的细胞 ID，请检查。")

    if len(common) < len(sc_ids) or len(common) < len(pt_ids):
        print(f"[warn] 细胞 ID 不完全一致：共有 {len(common)} / sc {len(sc_ids)} / pt {len(pt_ids)}，按交集对齐。")

    # 以 sc 为主，但仅保留交集部分；顺序按 sc_ids
    ad_sc = ad_sc[common].copy()
    ad_pt = ad_pt[common].copy()
    n = ad_sc.n_obs
    print(f"[info] aligned cells: n={n}")

    # -------- 3) 表达 PCA：X_expr_pca --------
    print("[step] PCA on scRNA ...")
    ad_expr = ad_sc.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 若已经log/归一，可以直接 PCA；这里给出通用流程
        if "highly_variable" not in ad_expr.var.columns:
            sc.pp.highly_variable_genes(ad_expr, n_top_genes=2000, flavor="seurat_v3")
        sc.pp.scale(ad_expr, max_value=10)
        sc.tl.pca(ad_expr, n_comps=N_PCS, use_highly_variable=True)

    X_expr_pca = ad_expr.obsm["X_pca"]
    X_expr_pca = zscore_block(X_expr_pca)
    expr_cols  = [f"expr_pca_{i:02d}" for i in range(1, X_expr_pca.shape[1]+1)]
    print("[ok] X_expr_pca:", X_expr_pca.shape)

    # -------- 4) cell_type one-hot：X_ct_onehot --------
    if CT_KEY not in ad_sc.obs.columns:
        raise KeyError(f"obs 里找不到 cell_type 列：{CT_KEY}")

    print(f"[step] one-hot encode cell type from obs['{CT_KEY}'] ...")
    ct_ser = ad_sc.obs[CT_KEY].astype("category")
    ct_dum = pd.get_dummies(ct_ser, drop_first=False)
    X_ct = ct_dum.to_numpy(dtype=np.float32)
    if CENTER_ONEHOT:
        X_ct = zscore_block(X_ct)
    ct_cols = [f"ct_{c}" for c in ct_dum.columns.astype(str)]
    print("[ok] X_ct_onehot:", X_ct.shape)

    # -------- 5) pathway / TF：从 ad_pt 直接拷贝 --------
    def get_block_from_pt(ad, key, default_prefix):
        if key not in ad.obsm_keys():
            print(f"[warn] pathway/tf block {key} 不存在，跳过。")
            return None, []
        X = np.asarray(ad.obsm[key])
        if X.ndim == 1:
            X = X.reshape(n, 1)
        else:
            X = X.reshape(n, -1)
        fb = (ad.uns.get("feature_blocks", {}) or {}).get(key, {})
        if fb and "cols" in fb and len(fb["cols"]) == X.shape[1]:
            cols = list(map(str, fb["cols"]))
        else:
            cols = [f"{default_prefix}_{i:02d}" for i in range(X.shape[1])]
        X = X.astype(np.float32)
        return X, cols

    X_pw, pw_cols = get_block_from_pt(ad_pt, "X_pathway_progeny14", "progeny")
    X_tf, tf_cols = get_block_from_pt(ad_pt, "X_tfact_dorothea", "tfact")

    if X_pw is None or X_tf is None:
        raise RuntimeError("X_pathway_progeny14 或 X_tfact_dorothea 缺失，请检查 cell_pathway_tf_acts.h5ad。")

    print("[ok] X_pathway_progeny14:", X_pw.shape)
    print("[ok] X_tfact_dorothea   :", X_tf.shape)

    # -------- 6) 拼接所有 feature block -> X_all --------
    mats   = [X_expr_pca, X_ct, X_pw, X_tf]
    names  = expr_cols + ct_cols + [f"progeny_{c}" for c in pw_cols] + [f"tfact_{c}" for c in tf_cols]
    blocks = (["X_expr_pca"] * X_expr_pca.shape[1]
             + ["X_ct_onehot"] * X_ct.shape[1]
             + ["X_pathway_progeny14"] * X_pw.shape[1]
             + ["X_tfact_dorothea"] * X_tf.shape[1])

    X_all = np.concatenate(mats, axis=1).astype(np.float32)
    print("[concat] X shape:", X_all.shape)

    # -------- 7) 构造 AnnData：cell_features.final.h5ad --------
    var_df = pd.DataFrame(index=pd.Index(names, name="feature"))
    var_df["block"] = blocks

    ad_final = sc.AnnData(
        X=X_all,
        obs=ad_sc.obs.copy(),
        var=var_df
    )

    # 把 pathway/TF 的 block 信息带过去（方便追溯）
    ad_final.obsm["X_expr_pca"]         = X_expr_pca
    ad_final.obsm["X_ct_onehot"]        = X_ct
    ad_final.obsm["X_pathway_progeny14"] = X_pw
    ad_final.obsm["X_tfact_dorothea"]    = X_tf

    ad_final.uns["feature_blocks"] = {
        "X_expr_pca": {
            "shape": [ad_final.n_obs, X_expr_pca.shape[1]],
            "cols": expr_cols,
            "source": "PCA on scRNA (HVGs, scaled)"
        },
        "X_ct_onehot": {
            "shape": [ad_final.n_obs, X_ct.shape[1]],
            "cols": ct_cols,
            "source": f"one-hot from obs['{CT_KEY}'], center={CENTER_ONEHOT}"
        },
        "X_pathway_progeny14": {
            "shape": [ad_final.n_obs, X_pw.shape[1]],
            "cols": [f"progeny_{c}" for c in pw_cols],
            "source": f"decoupler WAGGR PROGENy (from {os.path.basename(PT_AD)})"
        },
        "X_tfact_dorothea": {
            "shape": [ad_final.n_obs, X_tf.shape[1]],
            "cols": [f"tfact_{c}" for c in tf_cols],
            "source": f"decoupler WAGGR DoRothEA (from {os.path.basename(PT_AD)})"
        }
    }

    # 记录每个 block 在 X 中的起止位置
    offsets = {}
    s = 0
    for key in ["X_expr_pca", "X_ct_onehot", "X_pathway_progeny14", "X_tfact_dorothea"]:
        d = sum(1 for b in blocks if b == key)
        if d > 0:
            offsets[key] = {"start": int(s), "end": int(s + d), "dim": int(d)}
            s += d
    ad_final.uns["X_blocks"] = offsets
    ad_final.uns["X_summary"] = {
        "n_cells": int(ad_final.n_obs),
        "n_features": int(ad_final.n_vars),
        "blocks": list(offsets.keys()),
        "note": "X = [PCA || CT One-Hot || PROGENy || DoRothEA]"
                + (" (one-hot z-scored)" if CENTER_ONEHOT else "")
    }

    # -------- 8) 导出列名 CSV + 保存 h5ad --------
    pd.DataFrame({"feature": names, "block": blocks}).to_csv(CSV_FN, index=False)
    print("[save]", CSV_FN)

    ad_final.write(OUT_H5)
    print("[save]", OUT_H5)
    print("[done] cell_features.final.h5ad 构建完成。")


if __name__ == "__main__":
    main()
