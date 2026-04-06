#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build spot→LR (present) edges with spatial smoothing and controllable sparsification.

Fixes
-----
- --only-triad-lr now prefers using lr-index columns from edges:
    sender:   dst_lr_idx
    receiver: src_lr_idx
  This avoids the common mismatch where lr_h5ad.obs has no pair_name/lr column
  (often obs_names are node_id), which previously could make the candidate set empty.

- Memory optimization: only slice ST matrix by ligand genes needed for candidate LRs.

Outputs
-------
CSV columns:
  src_spot_idx, dst_lr_idx, weight, lr
plus optional metadata columns if present in LR.obs:
  lr_id, lr_key, pair_name (etc.)

"""

from __future__ import annotations
import argparse, warnings, math, ast, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# ---- optional deps
try:
    import squidpy as sq  # only needed for mode=squidpy
    _HAS_SQ = True
except Exception:
    _HAS_SQ = False

try:
    from sklearn.neighbors import NearestNeighbors  # used in mode=knn
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ========== small utils ==========

def read_csv_any(p: Path) -> pd.DataFrame:
    for enc in (None, "utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(p, encoding="latin-1", engine="python", low_memory=False)

def to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)

def row_normalize_csr(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.tocsr(copy=False)
    rs = np.array(A.sum(1)).ravel()
    rs[rs == 0.0] = 1.0
    return sp.diags(1.0/rs) @ A

def iterative_smooth(A: sp.csr_matrix, v: np.ndarray, alpha: float, k: int) -> np.ndarray:
    if k <= 0:
        return v.astype(np.float32, copy=False)
    Ahat = row_normalize_csr(A)
    x = v.astype(np.float32, copy=True)
    for _ in range(k):
        x = (1.0 - alpha) * x + alpha * (Ahat @ x)
    return x

def parse_listish(s: str) -> List[str]:
    """Accept 'A|B', 'A,B', 'A;B', '["A","B"]' etc. Return UPPER gene-like tokens."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x).strip().upper() for x in v if str(x).strip()]
        except Exception:
            pass
    # treat common separators
    toks = re.split(r"[;|,+]|\s*\+\s*|/|\s+", s.replace("__", "|"))
    toks = [t.strip().upper() for t in toks if t.strip()]
    return toks

def find_first(df_or_cols, candidates: List[str]) -> Optional[str]:
    cols = df_or_cols.columns if hasattr(df_or_cols, "columns") else list(df_or_cols)
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k.lower() in low:
            return low[k.lower()]
    for c in cols:
        cl = c.lower()
        for k in candidates:
            if k.lower() in cl:
                return c
    return None

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ========== LR helpers ==========

def detect_lig_rec_cols(obs: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    lig = find_first(obs, ["ligand_subunits", "ligand_genes", "ligand", "l_genes", "l"])
    rec = find_first(obs, ["receptor_subunits", "receptor_genes", "receptor", "r_genes", "r"])
    return lig, rec

def lrname_from_subunits(lig_sub: str, rec_sub: str) -> str:
    """
    Convert 'A;B' to 'A_B' within each side, then 'L__R'.
    Keep ordering as in the string if provided; if you want strict canonical ordering,
    sort tokens before joining.
    """
    L = [t for t in re.split(r"[;|,+]|\s*\+\s*", str(lig_sub)) if t.strip()]
    R = [t for t in re.split(r"[;|,+]|\s*\+\s*", str(rec_sub)) if t.strip()]
    L = [t.strip().upper() for t in L if t.strip()]
    R = [t.strip().upper() for t in R if t.strip()]
    if not L or not R:
        return ""
    return "_".join(L) + "__" + "_".join(R)

def detect_lr_display_array(LR: ad.AnnData) -> np.ndarray:
    """
    Preferred display name column order:
      pair_name > lr > lr_id > lr_key
    Else derive from ligand_subunits/receptor_subunits.
    Else fallback to obs_names.
    """
    obs = LR.obs.copy()
    for key in ["pair_name", "lr", "lr_id", "lr_key"]:
        if key in obs.columns:
            return obs[key].astype(str).fillna("").values

    lig_col, rec_col = detect_lig_rec_cols(obs)
    if lig_col is not None and rec_col is not None:
        lig = obs[lig_col].astype(str).fillna("").values
        rec = obs[rec_col].astype(str).fillna("").values
        out = np.array([lrname_from_subunits(lig[i], rec[i]) for i in range(LR.n_obs)], dtype=object)
        # some rows might still be empty
        miss = np.sum(out == "")
        if miss:
            # fallback those to obs_names
            out2 = out.astype(object)
            names = np.array([str(x) for x in LR.obs_names], dtype=object)
            out2[out2 == ""] = names[out2 == ""]
            return out2
        return out

    return np.array([str(x) for x in LR.obs_names], dtype=object)


# ========== adjacency builders ==========

def build_adj_squidpy(ad_sp, n_neighbors: int = 6, radius: Optional[float] = None) -> sp.csr_matrix:
    if not _HAS_SQ:
        raise RuntimeError("mode=squidpy requires squidpy to be installed.")
    sq.gr.spatial_neighbors(
        ad_sp,
        n_neighs=n_neighbors if radius is None else None,
        radius=radius,
        coord_type="generic",
    )
    A = ad_sp.obsp.get("spatial_connectivities", None)
    if A is None:
        raise RuntimeError("squidpy did not produce obsp['spatial_connectivities'].")
    return A.tocsr()

def build_adj_knn(ad_sp, k: int = 8, radius: Optional[float] = None) -> sp.csr_matrix:
    if not _HAS_SK:
        raise RuntimeError("mode=knn requires scikit-learn.")
    coords = None
    for key in ("spatial", "X_spatial", "coords", "X_coords"):
        if key in ad_sp.obsm and ad_sp.obsm[key] is not None:
            arr = to_dense(ad_sp.obsm[key])
            if arr.ndim >= 2 and arr.shape[1] >= 2 and arr.shape[0] == ad_sp.n_obs:
                coords = arr[:, :2]
                break
    if coords is None:
        raise RuntimeError("Cannot find spot coordinates in ad_sp.obsm (need obsm['spatial'] or similar).")

    if radius is not None:
        nn = NearestNeighbors(radius=radius, algorithm="ball_tree").fit(coords)
        A = nn.radius_neighbors_graph(coords, mode="connectivity")
    else:
        nn = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(coords)
        A = nn.kneighbors_graph(coords, mode="connectivity")
        A.setdiag(0)
        A.eliminate_zeros()
    return A.tocsr()

def build_adj(ad_sp, mode: str, n_neighbors: int, knn_k: int, radius: Optional[float]) -> sp.csr_matrix:
    mode = mode.lower()
    if mode == "squidpy":
        return build_adj_squidpy(ad_sp, n_neighbors=n_neighbors, radius=radius)
    if mode == "knn":
        return build_adj_knn(ad_sp, k=knn_k, radius=radius)
    raise ValueError("mode must be one of: squidpy, knn")


# ========== sparsify ==========

def sparsify_edges(df: pd.DataFrame,
                   min_weight: Optional[float] = None,
                   q_per_lr: Optional[float] = None,
                   topk_per_lr: Optional[int] = None,
                   topk_per_spot: Optional[int] = None) -> pd.DataFrame:
    out = df
    if min_weight is not None:
        out = out.loc[out["weight"] >= float(min_weight)]
    if q_per_lr is not None and len(out):
        thr = out.groupby("dst_lr_idx")["weight"].transform(lambda s: s.quantile(q_per_lr))
        out = out.loc[out["weight"] >= thr]
    if topk_per_lr is not None and len(out):
        out = (out.sort_values(["dst_lr_idx", "weight"], ascending=[True, False])
                 .groupby("dst_lr_idx", group_keys=False)
                 .head(int(topk_per_lr)))
    if topk_per_spot is not None and len(out):
        out = (out.sort_values(["src_spot_idx", "weight"], ascending=[True, False])
                 .groupby("src_spot_idx", group_keys=False)
                 .head(int(topk_per_spot)))
    return out


# ========== triad whitelist (idx-first) ==========

def get_triad_lr_idx_set(sender_edges_csv: str, receiver_edges_csv: str) -> Tuple[Optional[Set[int]], int]:
    S = read_csv_any(Path(sender_edges_csv))
    R = read_csv_any(Path(receiver_edges_csv))

    s_idx_col = find_first(S, ["dst_lr_idx", "lr_idx", "dst_lr_index", "dst_lr"])
    r_idx_col = find_first(R, ["src_lr_idx", "lr_idx", "src_lr_index", "src_lr"])

    if s_idx_col in S.columns and r_idx_col in R.columns:
        s_idx = pd.to_numeric(S[s_idx_col], errors="coerce").dropna().astype(int).unique()
        r_idx = pd.to_numeric(R[r_idx_col], errors="coerce").dropna().astype(int).unique()
        triad = set(s_idx).intersection(set(r_idx))
        return triad, len(triad)

    return None, 0


# ========== main pipeline ==========

def build_edges(args):
    ST = ad.read_h5ad(args.st_h5ad)
    LR = ad.read_h5ad(args.lr_h5ad)

    print(f"[ST] {ST.n_obs} spots × {ST.n_vars} genes")

    # uppercase gene names
    st_genes = np.array([str(g).upper() for g in ST.var_names], dtype=object)
    g2i = {g: i for i, g in enumerate(st_genes)}

    # LR ligand subunits column
    lig_col, rec_col = detect_lig_rec_cols(LR.obs)
    if lig_col is None:
        raise KeyError("LR.obs missing ligand subunits column (need 'ligand_subunits' or similar).")

    # LR display names (for output column 'lr')
    lr_display = detect_lr_display_array(LR)

    # ---- triad whitelist (idx-first) ----
    triad_lr_idx: Optional[Set[int]] = None
    if args.only_triad_lr:
        if not (args.sender_edges and args.receiver_edges):
            raise ValueError("--only-triad-lr requires --sender-edges and --receiver-edges")
        triad_lr_idx, ntri = get_triad_lr_idx_set(args.sender_edges, args.receiver_edges)

        if triad_lr_idx is not None:
            print(f"[LR] candidate for spot→LR edges (by lr_idx intersection): {len(triad_lr_idx)}")
        else:
            # fallback: use LR name strings intersection
            S = read_csv_any(Path(args.sender_edges))
            R = read_csv_any(Path(args.receiver_edges))
            s_lr_col = find_first(S, ["lr", "pair_name", "pair", "lr_name"]) or "lr"
            r_lr_col = find_first(R, ["lr", "pair_name", "pair", "lr_name"]) or "lr"
            if s_lr_col not in S.columns or r_lr_col not in R.columns:
                raise KeyError("Cannot find LR string column in sender/receiver edges (expected lr/pair_name).")
            s_lrs = set(S[s_lr_col].dropna().astype(str).str.upper().str.replace(" ", "", regex=False))
            r_lrs = set(R[r_lr_col].dropna().astype(str).str.upper().str.replace(" ", "", regex=False))
            lr_whitelist = s_lrs & r_lrs
            # map lr_display -> idx
            disp2idx = {str(lr_display[i]).upper().replace(" ", ""): i for i in range(LR.n_obs)}
            triad_lr_idx = set()
            for name in lr_whitelist:
                if name in disp2idx:
                    triad_lr_idx.add(disp2idx[name])
            print(f"[LR] candidate for spot→LR edges (fallback name→idx): {len(triad_lr_idx)}")

    # ---- adjacency ----
    A = build_adj(ST, mode=args.mode, n_neighbors=args.n_neighbors,
                  knn_k=args.knn_k, radius=args.radius)
    print(f"[adj] nnz={A.nnz}  shape={A.shape}")

    # ---- choose LR indices to iterate ----
    if triad_lr_idx is not None:
        lr_indices = np.array(sorted(list(triad_lr_idx)), dtype=int)
    else:
        lr_indices = np.arange(LR.n_obs, dtype=int)

    # ---- collect needed ligand genes (only for candidate LRs) ----
    lig_strs = LR.obs[lig_col].astype(str).values
    needed = set()
    for i in lr_indices:
        for g in parse_listish(lig_strs[i]):
            needed.add(g)
    needed = sorted(needed)

    st_idx = np.array([g2i.get(g, -1) for g in needed], dtype=int)
    ok = st_idx >= 0
    missing_genes = {g for g, m in zip(needed, ok) if not m}
    needed = [g for g, m in zip(needed, ok) if m]
    st_idx = st_idx[ok]
    gene2col = {g: j for j, g in enumerate(needed)}

    # slice ST.X only for needed genes
    Xst = ST.X
    if sp.issparse(Xst):
        E = Xst[:, st_idx].toarray().astype(np.float32)
    else:
        E = np.asarray(Xst[:, st_idx], dtype=np.float32)

    # smoothing
    alpha = float(args.smooth_alpha)
    kstep = int(args.smooth_k)

    # aggregation
    agg = args.ligand_agg.lower()

    rows = []
    n_spots = ST.n_obs

    # optional extra LR metadata columns (if exist)
    meta_cols = []
    for k in ["pair_name", "lr_id", "lr_key"]:
        if k in LR.obs.columns:
            meta_cols.append(k)
    meta_arrays = {k: LR.obs[k].astype(str).fillna("").values for k in meta_cols}

    for lr_idx in lr_indices:
        ligs = parse_listish(lig_strs[lr_idx])
        if not ligs:
            continue

        # build subunit matrix (n_spots × n_subunits)
        sub = np.zeros((n_spots, len(ligs)), dtype=np.float32)
        for j, g in enumerate(ligs):
            c = gene2col.get(g, None)
            if c is None:
                # missing gene -> zeros
                continue
            sub[:, j] = E[:, c]

        if agg == "min":
            v = sub.min(axis=1)
        elif agg == "mean":
            v = sub.mean(axis=1)
        elif agg == "geom":
            with np.errstate(divide="ignore"):
                logM = np.where(sub > 0, np.log(sub), -np.inf)
                v = np.exp(np.mean(logM, axis=1))
                v[~np.isfinite(v)] = 0.0
        else:
            raise ValueError("--ligand-agg must be one of: min, mean, geom")

        if kstep > 0:
            v = iterative_smooth(A, v, alpha=alpha, k=kstep)

        nz = np.flatnonzero(v > 0)
        if nz.size == 0:
            continue

        dfk = pd.DataFrame({
            "src_spot_idx": nz.astype(np.int32),
            "dst_lr_idx": np.full(nz.size, lr_idx, dtype=np.int32),
            "weight": v[nz].astype(np.float32),
            "lr": np.full(nz.size, str(lr_display[lr_idx]), dtype=object),
        })
        for k in meta_cols:
            dfk[k] = np.full(nz.size, meta_arrays[k][lr_idx], dtype=object)
        rows.append(dfk)

    if not rows:
        print("[warn] no edges constructed.")
        return pd.DataFrame(columns=["src_spot_idx", "dst_lr_idx", "weight", "lr"])

    df = pd.concat(rows, ignore_index=True)

    if missing_genes:
        print(f"[note] {len(missing_genes)} ligand genes not found in ST matrix (treated as zeros).")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st-h5ad", required=True)
    ap.add_argument("--lr-h5ad", required=True)

    # Triad LR whitelist (sender & receiver both exist)
    ap.add_argument("--sender-edges", type=str, default="", help="cell→LR cpdbLite CSV (preferably has dst_lr_idx)")
    ap.add_argument("--receiver-edges", type=str, default="", help="LR→cell cpdbLite CSV (preferably has src_lr_idx)")
    ap.add_argument("--only-triad-lr", action="store_true", help="only keep LR present on both sides")

    # Spatial graph
    ap.add_argument("--mode", choices=["squidpy", "knn"], default="squidpy")
    ap.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors", type=int, default=6, help="for squidpy")
    ap.add_argument("--knn-k", "--knn_k", dest="knn_k", type=int, default=8, help="for knn")
    ap.add_argument("--radius", type=float, default=None, help="if set, use radius graph")

    # Smoothing
    ap.add_argument("--smooth-k", "--smooth_k", dest="smooth_k", type=int, default=2, help="iterations; 0 to disable")
    ap.add_argument("--smooth-alpha", "--smooth_alpha", dest="smooth_alpha", type=float, default=0.6)

    # Ligand aggregation across subunits
    ap.add_argument("--ligand-agg", "--ligand_agg", dest="ligand_agg", choices=["min", "mean", "geom"], default="min")

    # Sparsify
    ap.add_argument("--min-weight", "--min_weight", dest="min_weight", type=float, default=None)
    ap.add_argument("--quantile-per-lr", "--quantile_per_lr", dest="quantile_per_lr", type=float, default=None)
    ap.add_argument("--topk-per-lr", "--topk_per_lr", dest="topk_per_lr", type=int, default=None)
    ap.add_argument("--topk-per-spot", "--topk_per_spot", dest="topk_per_spot", type=int, default=None)

    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    print("[info] building edges ...")
    df = build_edges(args)
    print(f"[pre-sparsify] edges={len(df):,}")

    df = sparsify_edges(
        df,
        min_weight=args.min_weight,
        q_per_lr=args.quantile_per_lr,
        topk_per_lr=args.topk_per_lr,
        topk_per_spot=args.topk_per_spot,
    )
    print(f"[post-sparsify] edges={len(df):,}")

    outp = Path(args.out_csv)
    safe_mkdir(outp.parent)
    df.to_csv(outp, index=False)
    print(f"[save] {outp} | rows={len(df):,}")


if __name__ == "__main__":
    main()
