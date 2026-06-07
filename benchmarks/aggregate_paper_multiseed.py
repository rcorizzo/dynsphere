#!/usr/bin/env python3
"""Aggregate paper benchmark result files across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "auroc",
    "vus_roc",
    "auprc",
    "vus_pr",
    "pa_f1",
    "pauc_fpr_05",
    "event_recall_top_1",
    "event_recall_top_3",
    "event_recall_top_5",
    "seconds",
]


def normalize_contextual(path: Path, seed: int) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    rows = []
    results = payload.get("results", {})
    iterable = results.items() if isinstance(results, dict) else ((item.get("method"), item) for item in results)
    for method, result in iterable:
        metrics = result.get("overall", result) if isinstance(result, dict) else {}
        row = {
            "dataset": "DynSphere-Contextual",
            "source": "contextual",
            "method": method,
            "seed": seed,
            "file": "",
            "status": "ok",
        }
        for metric in METRICS:
            if metric in metrics:
                row[metric] = metrics[metric]
        rows.append(row)
    return rows


def normalize_csv(path: Path, seed: int) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].fillna("ok") == "ok"]
    rows = []
    for _, item in df.iterrows():
        dataset = item.get("dataset")
        if pd.isna(dataset) and "source" in item:
            dataset = item.get("source")
        source = item.get("source", "")
        file_value = str(item.get("file", ""))
        dataset_text = str(dataset)
        if source == "" or pd.isna(source):
            source = ""
            for candidate in ("MSL", "SMAP", "SMD"):
                if candidate in file_value or candidate in dataset_text:
                    source = "tsb"
                    dataset = candidate
                    break
        method = item.get("method")
        row = {
            "dataset": str(dataset),
            "source": source,
            "method": str(method),
            "seed": int(item.get("seed", seed)) if not pd.isna(item.get("seed", seed)) else seed,
            "file": file_value,
            "status": "ok",
        }
        for metric in METRICS:
            if metric in df.columns and not pd.isna(item.get(metric)):
                row[metric] = float(item.get(metric))
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw

    group_cols = ["dataset", "method"]
    summaries = []
    for keys, group in raw.groupby(group_cols, sort=True):
        dataset, method = keys
        out = {
            "dataset": dataset,
            "method": method,
            "runs": int(len(group)),
            "seeds": int(group["seed"].nunique()) if "seed" in group else 0,
            "files": int(group["file"].nunique()) if "file" in group else 1,
        }
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(values):
                out[f"{metric}_mean"] = float(values.mean())
                out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(out)

    by_method = []
    for method, group in raw.groupby("method", sort=True):
        out = {
            "dataset": "__overall__",
            "method": method,
            "runs": int(len(group)),
            "seeds": int(group["seed"].nunique()) if "seed" in group else 0,
            "files": int(group["file"].nunique()) if "file" in group else 1,
        }
        for metric in METRICS:
            if metric not in group.columns:
                continue
            dataset_means = (
                group.groupby("dataset")[metric]
                .apply(lambda values: pd.to_numeric(values, errors="coerce").dropna().mean())
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(dataset_means):
                out[f"{metric}_mean"] = float(dataset_means.mean())
                out[f"{metric}_std"] = float(dataset_means.std(ddof=1)) if len(dataset_means) > 1 else 0.0
        by_method.append(out)

    return raw, pd.concat([pd.DataFrame(summaries), pd.DataFrame(by_method)], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contextual-json", nargs="*", default=[])
    parser.add_argument("--csv", nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path_str in args.contextual_json:
        path = Path(path_str)
        seed = int(path.stem.split("_seed")[-1]) if "_seed" in path.stem else -1
        rows.extend(normalize_contextual(path, seed))
    for path_str in args.csv:
        path = Path(path_str)
        seed = int(path.stem.split("_seed")[-1]) if "_seed" in path.stem else -1
        rows.extend(normalize_csv(path, seed))

    raw, summary = aggregate(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out_dir / "paper_multiseed_raw.csv", index=False)
    summary.to_csv(args.out_dir / "paper_multiseed_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
