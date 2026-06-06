#!/usr/bin/env python3
"""Run subsampled real-world benchmark datasets with existing detectors."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
TSB_DIR = REPO_ROOT / "benchmarks" / "tsb_ad_m"
for path in (REPO_ROOT, TSB_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_tsb_ad_m import (  # noqa: E402
    ENHANCED_ABLATIONS,
    PENDULUM_ABLATIONS,
    score_deep_svdd_kmeans,
    score_forecast_mlp,
    score_enhanced_pendulum,
    score_hybrid,
    score_pendulum,
    score_pyod,
    score_sequence_ae,
    score_static_ae,
    score_window_knn,
)


METHODS = {
    "pendulum": score_pendulum,
    **{
        name: (lambda kwargs: lambda x_train, x_all, train_idx, random_state: score_pendulum(
            x_train, x_all, train_idx, random_state, **kwargs
        ))(kwargs)
        for name, kwargs in PENDULUM_ABLATIONS.items()
    },
    "enhanced_pendulum": score_enhanced_pendulum,
    **{
        name: (lambda kwargs: lambda x_train, x_all, train_idx, random_state: score_enhanced_pendulum(
            x_train, x_all, train_idx, random_state, **kwargs
        ))(kwargs)
        for name, kwargs in ENHANCED_ABLATIONS.items()
    },
    "sequence_ae": lambda x_train, x_all, train_idx, random_state: score_sequence_ae(
        x_train, x_all, random_state
    ),
    "static_ae": lambda x_train, x_all, train_idx, random_state: score_static_ae(
        x_train, x_all, random_state
    ),
    "deep_svdd_kmeans": lambda x_train, x_all, train_idx, random_state: score_deep_svdd_kmeans(
        x_train, x_all, random_state
    ),
    "pendulum_seqae": score_hybrid,
    "hybrid": score_hybrid,
    "window_knn": lambda x_train, x_all, train_idx, random_state: score_window_knn(
        x_train, x_all, random_state
    ),
    "forecast_mlp": lambda x_train, x_all, train_idx, random_state: score_forecast_mlp(
        x_train, x_all, random_state
    ),
    "iforest": lambda x_train, x_all, train_idx, random_state: score_pyod(
        "iforest", x_train, x_all, random_state
    ),
    "hbos": lambda x_train, x_all, train_idx, random_state: score_pyod(
        "hbos", x_train, x_all, random_state
    ),
    "copod": lambda x_train, x_all, train_idx, random_state: score_pyod(
        "copod", x_train, x_all, random_state
    ),
    "lof": lambda x_train, x_all, train_idx, random_state: score_pyod(
        "lof", x_train, x_all, random_state
    ),
    "knn": lambda x_train, x_all, train_idx, random_state: score_pyod(
        "knn", x_train, x_all, random_state
    ),
}


LABEL_CANDIDATES = [
    "is_anomaly",
    "Label",
    "label",
    "attack",
    "Attack",
    "ATT_FLAG",
    "Normal/Attack",
    "class",
    "Class",
    "marker",
    "result",
]


def read_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".arff":
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].map(
                lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
            )
        df.columns = [str(col).strip("'\" ") for col in df.columns]
        return df
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    kwargs = {"sep": sep}
    if path.suffix == ".gz":
        kwargs["compression"] = "gzip"
    df = pd.read_csv(path, **kwargs)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def find_label(df: pd.DataFrame, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def normalize_labels(values: pd.Series | None, default_normal: bool) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=int) if default_normal else np.array([], dtype=int)
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy()
        return (numeric > 0).astype(int)
    text = values.astype(str).str.strip().str.lower()
    normal = {"0", "normal", "natural", "benign", "false", "none", "no", "negative"}
    return (~text.isin(normal)).astype(int).to_numpy()


def features_and_labels(df: pd.DataFrame, label_col: str | None, default_normal: bool) -> tuple[pd.DataFrame, np.ndarray]:
    labels = (
        np.zeros(len(df), dtype=int)
        if label_col is None
        else normalize_labels(df[label_col], default_normal)
    )
    drop_cols = set()
    if label_col is not None:
        drop_cols.add(label_col)
    for col in df.columns:
        low = col.lower()
        if low in {"time", "timestamp", "datetime", "date", "datetime"}:
            drop_cols.add(col)
        if low.startswith("attack_") or low in {"attack_p1", "attack_p2", "attack_p3"}:
            drop_cols.add(col)
    numeric = df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number])
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    return numeric, labels


def downsample(x: np.ndarray, y: np.ndarray, max_rows: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    if max_rows is None or len(y) <= max_rows:
        return x, y, 1
    stride = int(math.ceil(len(y) / max_rows))
    return x[::stride], y[::stride], stride


def metrics(labels: np.ndarray, scores: np.ndarray, include_vus: bool) -> dict[str, float | int | str | None]:
    positives = int(labels.sum())
    out = {"n": int(len(labels)), "positives": positives, "auroc": None, "auprc": None}
    if positives and positives < len(labels):
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        if include_vus:
            from range_metrics import range_vus_scores

            out.update(range_vus_scores(labels, scores))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-label", default=None)
    parser.add_argument("--test-label", default=None)
    parser.add_argument("--max-train-rows", type=int, default=20000)
    parser.add_argument("--max-test-rows", type=int, default=30000)
    parser.add_argument("--methods", nargs="+", default=["pendulum", "enhanced_pendulum", "iforest", "hbos", "copod"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=37)
    parser.add_argument("--include-vus", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Available: {sorted(METHODS)}")

    train_df = read_csv(args.train)
    test_df = read_csv(args.test)
    train_label = find_label(train_df, args.train_label)
    test_label = find_label(test_df, args.test_label)
    x_train_df, y_train_all = features_and_labels(train_df, train_label, default_normal=True)
    x_test_df, y_test = features_and_labels(test_df, test_label, default_normal=False)
    common_features = [col for col in x_train_df.columns if col in set(x_test_df.columns)]
    if not common_features:
        raise ValueError("No common numeric feature columns between train and test files")
    x_train_all = x_train_df[common_features].to_numpy(dtype=float)
    x_test = x_test_df[common_features].to_numpy(dtype=float)

    train_idx = np.flatnonzero(y_train_all == 0) if len(y_train_all) else np.arange(len(x_train_all))
    x_train = x_train_all[train_idx]
    x_train, _, train_stride = downsample(x_train, np.zeros(len(x_train), dtype=int), args.max_train_rows)
    x_test, y_test, test_stride = downsample(x_test, y_test, args.max_test_rows)

    rows = []
    for method in args.methods:
        started = time.time()
        try:
            scores = METHODS[method](x_train, x_test, np.arange(len(x_train)), args.random_state)
            row = {
                "dataset": args.dataset,
                "method": method,
                "status": "ok",
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "features": int(x_train.shape[1]),
                "train_stride": int(train_stride),
                "test_stride": int(test_stride),
                "seconds": float(time.time() - started),
                **metrics(y_test, scores, args.include_vus),
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "dataset": args.dataset,
                "method": method,
                "status": "failed",
                "seconds": float(time.time() - started),
                "error": str(exc),
            }
        rows.append(row)
        print(row, flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    summary = {
        "dataset": args.dataset,
        "train": str(args.train),
        "test": str(args.test),
        "train_label": train_label,
        "test_label": test_label,
        "rows": rows,
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
