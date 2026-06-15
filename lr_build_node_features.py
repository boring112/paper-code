import os, re, argparse
import numpy as np
import pandas as pd
import anndata as ad

# ---------- 工具函数 ----------

def parse_list(x: str):
    """把 'A;B' / 'A+B' / 'A,B' 等形式拆成基因列表（大写）"""
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [t.strip().upper()
            for t in re.split(r"[;|,+]|\s*\+\s*", str(x))
            if t.strip()]

def canonical_side_from_genes(genes):
    """去重 + 排序 + ';' 连接，做成规范的亚基串"""
    genes = sorted(set(g for g in genes if g))
    return ";".join(genes)

def pick_col(df: pd.DataFrame, candidates, fuzzy_key=None):
    cols = [c.lower() for c in df.columns]
    for c in candidates:
        if c in cols:
            return c
    if fuzzy_key:
        for c in cols:
            if fuzzy_key in c:
                return c
    return None

def ensure_lr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保生成标准列:
      ligand_subunits / receptor_subunits /
      ligand_location / receptor_location / receptor_class / interaction_mode
    若原表缺这些列则通过其他列推断/补 'other'。
    """
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]

    # --- 找到配体/受体列 ---
    L_cand = [
        "ligand_subunits","ligand_genes","ligand_symbols",
        "ligand","partner_a","source","from","a"
    ]
    R_cand = [
        "receptor_subunits","receptor_genes","receptor_symbols",
        "receptor","partner_b","target","to","b"
    ]

    cL = pick_col(d, L_cand, "ligand")
    cR = pick_col(d, R_cand, "receptor")

    if cL is None or cR is None:
        raise ValueError(
            f"无法在输入表中识别配体/受体列；现有列: {list(d.columns)}"
        )

    # --- 归一化为 *_subunits ---
    def to_subunits_col(series):
        return series.astype(str).apply(
            lambda s: canonical_side_from_genes(parse_list(s))
        )

    d["ligand_subunits"]   = to_subunits_col(d[cL])
    d["receptor_subunits"] = to_subunits_col(d[cR])

    # --- 其他先验列，缺失则补 'other' ---
    if "ligand_location" not in d.columns:
        d["ligand_location"] = "other"
    if "receptor_location" not in d.columns:
        d["receptor_location"] = "other"
    if "receptor_class" not in d.columns:
        d["receptor_class"] = "other"
    if "interaction_mode" not in d.columns:
        d["interaction_mode"] = "other"

    # 去重
    d = d.drop_duplicates(
        subset=["ligand_subunits","receptor_subunits"]
    ).reset_index(drop=True)
    return d

def one_hot_from_cats(series: pd.Series, categories, prefix: str):
    """
    按给定类别做 one-hot 编码，未知类全部归到 'other'（若存在）。
    返回: (矩阵(n×C), 列名列表)
    """
    x = series.astype(str).str.lower().fillna("other")
    vals = x.values
    mat = np.zeros((len(vals), len(categories)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(categories)}
    for i, v in enumerate(vals):
        v = v.lower()
        if v in idx:
            mat[i, idx[v]] = 1.0
        elif "other" in idx:
            mat[i, idx["other"]] = 1.0
    cols = [f"{prefix}:{c}" for c in categories]
    return mat, cols

def log1p_scale(arr):
    return np.log1p(np.asarray(arr, dtype=np.float32))


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(
        description="从 cleaned LR CSV 构建 LR 节点特征 (h5ad) + 标准 LR CSV"
    )
    ap.add_argument(
        "--lr_csv", type=str, required=True,
        help="输入 LR 列表 CSV (例如 lr_candidates.with_complexes.filtered.csv)"
    )
    ap.add_argument(
        "--out_dir", type=str, required=True,
        help="输出目录 (会自动创建)"
    )
    args = ap.parse_args()

    lr_csv  = os.path.abspath(args.lr_csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_normalized = os.path.join(out_dir, "lr_pairs.normalized.csv")
    out_h5         = os.path.join(out_dir, "lr_features.h5ad")
    out_cols       = os.path.join(out_dir, "lr_features.columns.csv")
    out_map        = os.path.join(out_dir, "lr_nodes.table.csv")

    print(f"[info] read LR CSV: {lr_csv}")
    df_raw = pd.read_csv(lr_csv, sep=None, engine="python")
    print(f"[info] raw rows = {len(df_raw)}")

    # 1) 规范列 & 填补缺失
    df = ensure_lr_columns(df_raw)
    print(f"[info] normalized unique pairs = {len(df)}")

    # 2) 为 relay_eval 等导出“标准列”CSV
    cols_needed = [
        "ligand_subunits","receptor_subunits",
        "ligand_location","receptor_location",
        "receptor_class","interaction_mode"
    ]
    df[cols_needed].to_csv(
        out_normalized, index=False, encoding="utf-8-sig"
    )
    print(f"[save] normalized LR CSV (for eval): {out_normalized}")

    # 3) 构建粗粒度 LR 节点特征
    df["node_id"] = np.arange(len(df), dtype=int)

    # 结构先验
    nL = df["ligand_subunits"].apply(
        lambda s: len(parse_list(s))
    ).astype(np.int32)
    nR = df["receptor_subunits"].apply(
        lambda s: len(parse_list(s))
    ).astype(np.int32)
    is_complex = ((nL > 1) | (nR > 1)).astype(np.float32)
    n_total_log = log1p_scale(nL + nR)

    X_struct = np.vstack([
        is_complex.values,
        nL.values.astype(np.float32),
        nR.values.astype(np.float32),
        n_total_log
    ]).T
    cols_struct = [
        "struct:is_complex",
        "struct:n_subunits_lig",
        "struct:n_subunits_rec",
        "struct:n_subunits_total_log1p",
    ]

    # 类别 One-hot
    LOC_CATS  = ["secreted","membrane","ecm","other"]
    MODE_CATS = ["paracrine","juxtacrine","ecm_mediated","other"]

    X_lig_loc, cols_lig_loc = one_hot_from_cats(
        df["ligand_location"], LOC_CATS, "lig_loc"
    )
    X_rec_loc, cols_rec_loc = one_hot_from_cats(
        df["receptor_location"], LOC_CATS, "rec_loc"
    )

    # 受体类别：按数据中出现的集合 + 兜底 other
    rc_all = sorted(
        set(df["receptor_class"].astype(str).str.lower().tolist())
        - {"", "nan"}
    )
    if "other" not in rc_all:
        rc_all.append("other")
    X_rc, cols_rc = one_hot_from_cats(
        df["receptor_class"], rc_all, "rc"
    )

    # 互动模式
    X_mode, cols_mode = one_hot_from_cats(
        df["interaction_mode"], MODE_CATS, "mode"
    )

    # 拼接为总特征
    blocks = [
        ("X_lig_loc",      X_lig_loc,  cols_lig_loc),
        ("X_rec_loc",      X_rec_loc,  cols_rec_loc),
        ("X_mode",         X_mode,     cols_mode),
        ("X_receptor_cls", X_rc,       cols_rc),
        ("X_struct",       X_struct,   cols_struct),
    ]
    X_all = np.concatenate([b[1] for b in blocks], axis=1).astype(np.float32)
    cols_all = sum([b[2] for b in blocks], [])

    # 4) AnnData: 特征矩阵 + 元信息
    ad_lr = ad.AnnData(
        X = X_all,
        obs = pd.DataFrame({
            "node_id": df["node_id"].values,
            "ligand_subunits": df["ligand_subunits"].values,
            "receptor_subunits": df["receptor_subunits"].values,
            "ligand_location": df["ligand_location"].astype(str).values,
            "receptor_location": df["receptor_location"].astype(str).values,
            "receptor_class": df["receptor_class"].astype(str).values,
            "interaction_mode": df["interaction_mode"].astype(str).values,
            "n_subunits_lig": nL.values,
            "n_subunits_rec": nR.values,
            "is_complex": is_complex.astype(bool).values,
        }).set_index("node_id")
    )

    for key, mat, cols in blocks:
        ad_lr.obsm[key] = mat
        ad_lr.uns[f"{key}_columns"] = cols

    # 5) 保存各种输出
    pd.Series(cols_all, name="feature").to_csv(
        out_cols, index=False, encoding="utf-8-sig"
    )
    df[[
        "node_id", "ligand_subunits", "receptor_subunits",
        "ligand_location", "receptor_location",
        "receptor_class", "interaction_mode"
    ]].to_csv(out_map, index=False, encoding="utf-8-sig")

    ad_lr.write(out_h5)

    print(f"[save] h5ad:     {out_h5}")
    print(f"[save] columns:  {out_cols}")
    print(f"[save] node map: {out_map}")
    print(f"[done] LR 节点特征构建完成。")

if __name__ == "__main__":
    main()
