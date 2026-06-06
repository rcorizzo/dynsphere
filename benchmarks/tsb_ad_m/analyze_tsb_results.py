#!/usr/bin/env python3
"""Aggregate TSB-AD-M results by source family and diagnostic subgroup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


STATIC_METHODS = ["iforest", "hbos", "copod", "pca"]


def source_name(path: str) -> str:
    base = os.path.basename(path)
    return base.split("_id_")[0].split("_", 1)[1]


def summarize_source_families(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"].eq("ok")].copy()
    ok["source"] = ok["file"].map(source_name)
    grouped = (
        ok.groupby(["source", "method"])
        .agg(
            files=("file", "nunique"),
            anomaly_fraction=("anomaly_fraction", "mean"),
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
        )
        .reset_index()
    )
    grouped["auprc_lift"] = grouped["auprc"] / grouped["anomaly_fraction"].clip(lower=1e-12)
    return grouped


def source_winners(source_summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    idx = source_summary.groupby("source")[metric].idxmax()
    return source_summary.loc[idx].sort_values("source").reset_index(drop=True)


def static_hard_subset(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = df[df["status"].eq("ok")].copy()
    ok["base"] = ok["file"].map(os.path.basename)
    wide = ok.pivot_table(index="base", columns="method", values="auprc", aggfunc="first")
    present_static = [m for m in STATIC_METHODS if m in wide.columns]
    wide["static_max"] = wide[present_static].max(axis=1, skipna=True)
    selected = wide[wide["static_max"] < threshold].index
    sub = ok[ok["base"].isin(selected)].copy()
    summary = (
        sub.groupby("method")
        .agg(
            files=("file", "nunique"),
            anomaly_fraction=("anomaly_fraction", "mean"),
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
        )
        .sort_values("auprc", ascending=False)
    )
    summary["auprc_lift"] = summary["auprc"] / summary["anomaly_fraction"].clip(lower=1e-12)
    return sub, summary.reset_index()


def latex_table(df: pd.DataFrame, columns: list[str], float_cols: set[str]) -> str:
    lines = []
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if col in float_cols:
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(int(value)) if isinstance(value, (np.integer, int)) else str(value))
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="benchmarks/tsb_ad_m/results_full_all/tsb_ad_m_results.csv")
    parser.add_argument("--out-dir", default="benchmarks/tsb_ad_m/analysis")
    parser.add_argument("--static-hard-threshold", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)

    source_summary = summarize_source_families(df)
    auprc_winners = source_winners(source_summary, "auprc")
    auroc_winners = source_winners(source_summary, "auroc")
    static_hard_rows, static_hard_summary = static_hard_subset(df, args.static_hard_threshold)

    source_summary.to_csv(out_dir / "source_family_summary.csv", index=False)
    auprc_winners.to_csv(out_dir / "source_family_auprc_winners.csv", index=False)
    auroc_winners.to_csv(out_dir / "source_family_auroc_winners.csv", index=False)
    static_hard_rows.to_csv(out_dir / "static_hard_rows.csv", index=False)
    static_hard_summary.to_csv(out_dir / "static_hard_summary.csv", index=False)

    latex = {
        "source_family_auprc_winners": latex_table(
            auprc_winners,
            ["source", "method", "files", "anomaly_fraction", "auprc", "auprc_lift"],
            {"anomaly_fraction", "auprc", "auprc_lift"},
        ),
        "static_hard_summary": latex_table(
            static_hard_summary,
            ["method", "files", "anomaly_fraction", "auprc", "auprc_lift"],
            {"anomaly_fraction", "auprc", "auprc_lift"},
        ),
    }
    (out_dir / "tables_latex.json").write_text(json.dumps(latex, indent=2) + "\n")

    print(f"Wrote analysis tables to {out_dir}")
    print("\nAUPRC winners by source family:")
    print(auprc_winners[["source", "method", "files", "anomaly_fraction", "auprc", "auprc_lift"]].to_string(index=False))
    print("\nStatic-hard summary:")
    print(static_hard_summary[["method", "files", "anomaly_fraction", "auprc", "auprc_lift"]].to_string(index=False))


if __name__ == "__main__":
    main()
