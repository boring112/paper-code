#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mirror_spot_lr_edges.py

把 spot→LR 的边表镜像成 LR→spot 的边表。

典型输入（spot→LR）：
    src_spot_idx, dst_lr_idx, weight[, lr]

典型输出（LR→spot）：
    src_lr_idx, dst_spot_idx, weight[, lr]

用法示例：
    python mirror_spot_lr_edges.py \
        --in-csv  /mnt/c/jieguo/GSE111672/PDAC_A/hgt_graph/edges/edges_spot_present_lr.NEW.smoothTriad.csv \
        --out-csv /mnt/c/jieguo/GSE111672/PDAC_A/hgt_graph/edges/edges_lr_bind_spot.from_smoothTriad.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def detect_cols(df: pd.DataFrame):
    """简单自动检测 spot / lr 列名。"""
    cols_lower = {c.lower(): c for c in df.columns}

    # spot 源列
    spot_candidates = ["src_spot_idx", "spot_idx", "spot", "src"]
    spot_col = None
    for key in spot_candidates:
        if key in cols_lower:
            spot_col = cols_lower[key]
            break

    # lr 目标列
    lr_candidates = ["dst_lr_idx", "lr_idx", "dst"]
    lr_col = None
    for key in lr_candidates:
        if key in cols_lower:
            lr_col = cols_lower[key]
            break

    if spot_col is None or lr_col is None:
        raise KeyError(
            f"无法在列中找到 spot / lr 索引列，现有列：{list(df.columns)}\n"
            f"期待类似: src_spot_idx / dst_lr_idx"
        )

    return spot_col, lr_col


def main():
    ap = argparse.ArgumentParser(
        description="Mirror spot→LR edges into LR→spot edges."
    )
    ap.add_argument(
        "--in-csv",
        required=True,
        help="输入 spot→LR 边 CSV（如 edges_spot_present_lr.NEW.smoothTriad.csv）",
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        help="输出 LR→spot 边 CSV（如 edges_lr_bind_spot.from_smoothTriad.csv）",
    )
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[IO] read: {in_path}")
    df = pd.read_csv(in_path)

    spot_col, lr_col = detect_cols(df)
    print(f"[detect] spot_col = {spot_col}, lr_col = {lr_col}")

    # 如果没有 weight 列，就默认全 1
    if "weight" in df.columns:
        w = df["weight"].astype(float)
    else:
        w = pd.Series(1.0, index=df.index, name="weight")

    out = pd.DataFrame(
        {
            "src_lr_idx": df[lr_col].astype("int64"),
            "dst_spot_idx": df[spot_col].astype("int64"),
            "weight": w,
        }
    )

    # 保留 lr 名称（如果有的话）
    if "lr" in df.columns:
        out["lr"] = df["lr"].astype(str)

    out.to_csv(out_path, index=False)
    print(f"[save] {out_path} | rows={len(out)}")


if __name__ == "__main__":
    main()
