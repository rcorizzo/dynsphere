#!/usr/bin/env python3
"""Multi-seed deep TSAD candidate runs on the paper benchmark protocol."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
TSB_DIR = REPO_ROOT / "benchmarks" / "tsb_ad_m"
REAL_WORLD_DIR = REPO_ROOT / "benchmarks" / "real_world_datasets"
for path in (REPO_ROOT, TSB_DIR, REAL_WORLD_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deep_tsad_smoke import (  # noqa: E402
    OmniAnomalyAdapter,
    TransformerReconAdapter,
    fit_and_score,
    make_windows,
    subsample,
)
from run_real_world_datasets import (  # noqa: E402
    downsample,
    features_and_labels,
    find_label,
    read_csv,
)
from run_tsb_ad_m import (  # noqa: E402
    discover_csvs,
    downsample_sequence,
    load_tsb_csv,
    source_name,
)


METHOD_BUILDERS = {
    "OmniAnomaly": lambda dim: OmniAnomalyAdapter(dim),
    "AnomalyTransformer": lambda dim: TransformerReconAdapter(dim, tranad=False),
    "TranAD": lambda dim: TransformerReconAdapter(dim, tranad=True),
}


@dataclass
class DatasetSpec:
    dataset: str
    source: str
    x_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_entities: np.ndarray
    test_entities: np.ndarray
    train_rows: int
    test_rows: int
    features: int
    train_stride: int = 1
    test_stride: int = 1
    file: str = ""


def metric_dict(labels: np.ndarray, scores: np.ndarray, include_vus: bool) -> dict[str, float | int | None]:
    positives = int(labels.sum())
    out: dict[str, float | int | None] = {
        "n": int(len(labels)),
        "positives": positives,
        "auroc": None,
        "auprc": None,
    }
    if positives and positives < len(labels):
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        from range_metrics import best_point_adjusted_f1, practical_range_metrics

        out.update(best_point_adjusted_f1(labels, scores))
        out.update(practical_range_metrics(labels, scores))
        if include_vus:
            from range_metrics import range_vus_scores

            out.update(range_vus_scores(labels, scores))
    return out


def load_contextual(args: argparse.Namespace) -> list[DatasetSpec]:
    train = pd.read_csv(args.contextual_train).sort_values(["entity_id", "time"]).reset_index(drop=True)
    test = pd.read_csv(args.contextual_test).sort_values(["entity_id", "time"]).reset_index(drop=True)
    feature_cols = [col for col in train.columns if col.startswith(args.feature_prefix)]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test[feature_cols].to_numpy(dtype=float))
    return [
        DatasetSpec(
            dataset="DynSphere-Contextual",
            source="contextual",
            x_train=x_train,
            x_test=x_test,
            y_test=test["is_anomaly"].to_numpy(dtype=int),
            train_entities=train["entity_id"].to_numpy(),
            test_entities=test["entity_id"].to_numpy(),
            train_rows=len(train),
            test_rows=len(test),
            features=len(feature_cols),
        )
    ]


def load_tsb_sources(args: argparse.Namespace) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    wanted = set(args.tsb_sources)
    for path in discover_csvs(Path(args.tsb_data_dir)):
        source = source_name(path)
        if source not in wanted:
            continue
        x_all, labels = load_tsb_csv(path)
        x_all, labels, _, stride = downsample_sequence(x_all, labels, args.tsb_downsample_max_rows)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_all)
        entities = np.zeros(len(labels), dtype=int)
        specs.append(
            DatasetSpec(
                dataset=source,
                source="tsb",
                x_train=x_scaled,
                x_test=x_scaled,
                y_test=labels.astype(int),
                train_entities=entities,
                test_entities=entities,
                train_rows=len(x_scaled),
                test_rows=len(x_scaled),
                features=x_scaled.shape[1],
                train_stride=stride,
                test_stride=stride,
                file=str(path),
            )
        )
    return specs


def load_real_world(args: argparse.Namespace) -> list[DatasetSpec]:
    raw_specs = [
        (
            "BATADAL",
            Path("benchmarks/real_world_datasets/data/batadal/BATADAL_dataset03.csv"),
            Path("benchmarks/real_world_datasets/data/batadal/BATADAL_dataset04.csv"),
            "ATT_FLAG",
            "ATT_FLAG",
        ),
        (
            "HAI-20.07",
            Path("benchmarks/real_world_datasets/data/hai_20_07/train1.csv.gz"),
            Path("benchmarks/real_world_datasets/data/hai_20_07/test1.csv.gz"),
            "attack",
            "attack",
        ),
        (
            "PowerSystem",
            Path("benchmarks/real_world_datasets/data/power_system/binary/combined_binary.csv"),
            Path("benchmarks/real_world_datasets/data/power_system/binary/combined_binary.csv"),
            "marker",
            "marker",
        ),
    ]
    specs: list[DatasetSpec] = []
    for dataset, train_path, test_path, train_label_arg, test_label_arg in raw_specs:
        if dataset not in args.real_world_sources:
            continue
        train_df = read_csv(REPO_ROOT / train_path)
        test_df = read_csv(REPO_ROOT / test_path)
        train_label = find_label(train_df, train_label_arg)
        test_label = find_label(test_df, test_label_arg)
        x_train_df, y_train_all = features_and_labels(train_df, train_label, default_normal=True)
        x_test_df, y_test = features_and_labels(test_df, test_label, default_normal=False)
        common_features = [col for col in x_train_df.columns if col in set(x_test_df.columns)]
        if not common_features:
            raise ValueError(f"No common numeric features for {dataset}")
        x_train_all = x_train_df[common_features].to_numpy(dtype=float)
        x_test = x_test_df[common_features].to_numpy(dtype=float)
        train_idx = np.flatnonzero(y_train_all == 0) if len(y_train_all) else np.arange(len(x_train_all))
        x_train = x_train_all[train_idx]
        x_train, _, train_stride = downsample(x_train, np.zeros(len(x_train), dtype=int), args.real_max_train_rows)
        x_test, y_test, test_stride = downsample(x_test, y_test, args.real_max_test_rows)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        specs.append(
            DatasetSpec(
                dataset=dataset,
                source="real_world",
                x_train=x_train_scaled,
                x_test=x_test_scaled,
                y_test=y_test.astype(int),
                train_entities=np.zeros(len(x_train_scaled), dtype=int),
                test_entities=np.zeros(len(x_test_scaled), dtype=int),
                train_rows=len(x_train_scaled),
                test_rows=len(x_test_scaled),
                features=x_train_scaled.shape[1],
                train_stride=train_stride,
                test_stride=test_stride,
                file=f"{train_path}|{test_path}",
            )
        )
    return specs


def row_scores_for_spec(
    spec: DatasetSpec,
    method: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    started = time.perf_counter()
    try:
        train_windows, _, train_labels = make_windows(
            spec.x_train,
            spec.train_entities,
            None,
            args.window,
        )
        test_windows, centers, y_window = make_windows(
            spec.x_test,
            spec.test_entities,
            spec.y_test,
            args.window,
        )
        if len(train_windows) == 0 or len(test_windows) == 0:
            raise ValueError("not enough rows to form windows")
        train_windows, train_labels = subsample(train_windows, train_labels, args.max_train_windows, seed + 101)
        original_test_windows = len(test_windows)
        test_windows, y_window = subsample(test_windows, y_window, args.max_test_windows, seed + 303)
        adapter = METHOD_BUILDERS[method](train_windows.shape[2])
        scores = fit_and_score(
            adapter,
            train_windows,
            test_windows,
            args.epochs,
            args.batch_size,
            args.lr,
            seed,
        )
        metrics = metric_dict(y_window, scores, args.include_vus)
        return {
            "dataset": spec.dataset,
            "source": spec.source,
            "file": spec.file,
            "method": method,
            "seed": seed,
            "status": "ok",
            "seconds": time.perf_counter() - started,
            "train_rows": spec.train_rows,
            "test_rows": spec.test_rows,
            "features": spec.features,
            "train_stride": spec.train_stride,
            "test_stride": spec.test_stride,
            "train_windows": int(len(train_windows)),
            "test_windows": int(len(test_windows)),
            "original_test_windows": int(original_test_windows),
            **metrics,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "dataset": spec.dataset,
            "source": spec.source,
            "file": spec.file,
            "method": method,
            "seed": seed,
            "status": "failed",
            "seconds": time.perf_counter() - started,
            "error": str(exc),
        }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    df = pd.DataFrame([row for row in rows if row.get("status") == "ok"])
    if df.empty:
        return {}
    metric_cols = [
        col
        for col in [
            "auroc",
            "auprc",
            "vus_roc",
            "vus_pr",
            "pa_f1",
            "pauc_fpr_05",
            "event_recall_top_1",
            "event_recall_top_3",
            "event_recall_top_5",
            "seconds",
        ]
        if col in df.columns
    ]
    by_dataset_method = {}
    for (dataset, method), group in df.groupby(["dataset", "method"], sort=True):
        item = {
            "runs": int(len(group)),
            "files": int(group["file"].nunique()) if "file" in group else 1,
        }
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            item[f"mean_{col}"] = float(values.mean()) if len(values) else None
            item[f"std_{col}"] = float(values.std(ddof=0)) if len(values) else None
        by_dataset_method[f"{dataset}/{method}"] = item

    by_method = {}
    for method, group in df.groupby("method", sort=True):
        item = {"runs": int(len(group))}
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            item[f"mean_{col}"] = float(values.mean()) if len(values) else None
            item[f"std_{col}"] = float(values.std(ddof=0)) if len(values) else None
        by_method[method] = item
    return {"by_dataset_method": by_dataset_method, "by_method": by_method}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/deep_tsad_paper_3seed"))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_BUILDERS), choices=list(METHOD_BUILDERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[37, 38, 39])
    parser.add_argument("--datasets", nargs="+", default=["contextual", "tsb", "real_world"], choices=["contextual", "tsb", "real_world"])
    parser.add_argument("--contextual-train", default="data/contextual/pendulum_contextual_train.csv")
    parser.add_argument("--contextual-test", default="data/contextual/pendulum_contextual_test.csv")
    parser.add_argument("--feature-prefix", default="x_")
    parser.add_argument("--tsb-data-dir", default=TSB_DIR / "data" / "TSB-AD-M")
    parser.add_argument("--tsb-sources", nargs="+", default=["MSL", "SMAP", "SMD"])
    parser.add_argument("--tsb-downsample-max-rows", type=int, default=None)
    parser.add_argument("--real-world-sources", nargs="+", default=["BATADAL", "HAI-20.07", "PowerSystem"])
    parser.add_argument("--real-max-train-rows", type=int, default=20000)
    parser.add_argument("--real-max-test-rows", type=int, default=30000)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    parser.add_argument("--include-vus", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    specs: list[DatasetSpec] = []
    if "contextual" in args.datasets:
        specs.extend(load_contextual(args))
    if "tsb" in args.datasets:
        specs.extend(load_tsb_sources(args))
    if "real_world" in args.datasets:
        specs.extend(load_real_world(args))
    if not specs:
        raise SystemExit("No dataset specs selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "deep_tsad_paper_results.csv"
    rows: list[dict[str, object]] = []
    done = set()
    if args.resume and results_path.exists():
        rows = pd.read_csv(results_path).where(pd.notnull(pd.read_csv(results_path)), None).to_dict(orient="records")
        done = {(row.get("dataset"), row.get("file"), row.get("method"), int(row.get("seed"))) for row in rows}

    for spec in specs:
        for method in args.methods:
            for seed in args.seeds:
                key = (spec.dataset, spec.file, method, seed)
                if key in done:
                    print(f"skip {spec.dataset} {Path(spec.file).name if spec.file else ''} {method} seed={seed}", flush=True)
                    continue
                row = row_scores_for_spec(spec, method, seed, args)
                rows.append(row)
                done.add(key)
                print(row, flush=True)
                pd.DataFrame(rows).to_csv(results_path, index=False)
                (args.out_dir / "deep_tsad_paper_summary.json").write_text(
                    json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "summary": summarize(rows)}, indent=2)
                    + "\n"
                )

    summary = summarize(rows)
    (args.out_dir / "deep_tsad_paper_summary.json").write_text(
        json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "summary": summary}, indent=2)
        + "\n"
    )
    print(f"Wrote {results_path}")
    print(json.dumps(summary.get("by_method", {}), indent=2))


if __name__ == "__main__":
    main()
