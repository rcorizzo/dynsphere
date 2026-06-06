#!/usr/bin/env python3
"""Run pendulum and baseline detectors on TSB-AD-M CSV files."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pendulum_detector import MultiConceptPendulumDetector  # noqa: E402


PYOD_METHODS = {"lof", "knn", "iforest", "hbos", "copod", "pca"}
ENHANCED_ABLATIONS = {
    "enhanced_no_pendulum": {"pendulum_weight": 0.0},
    "enhanced_no_sequence": {"sequence_weight": 0.0},
    "enhanced_no_window": {"window_knn_weight": 0.0},
    "enhanced_no_static": {"static_weight": 0.0},
}
NEURAL_METHODS = {
    "static_ae",
    "deep_svdd_kmeans",
    "sequence_ae",
    "hybrid",
    "pendulum_seqae",
    "window_knn",
    "forecast_mlp",
    "enhanced_pendulum",
    *ENHANCED_ABLATIONS,
}
PENDULUM_ABLATIONS = {
    "pendulum_no_transition": {"transition_weight": 0.0},
    "pendulum_no_energy": {"energy_weight": 0.0},
    "pendulum_no_raw": {"raw_dynamic_weight": 0.0},
    "pendulum_no_concepts": {"n_concepts": 1, "transition_weight": 0.0},
}
ALL_METHODS = [
    "pendulum",
    *sorted(PENDULUM_ABLATIONS),
    "static_ae",
    "deep_svdd_kmeans",
    "sequence_ae",
    "pendulum_seqae",
    "hybrid",
    "window_knn",
    "forecast_mlp",
    "enhanced_pendulum",
    *sorted(ENHANCED_ABLATIONS),
    "lof",
    "knn",
    "iforest",
    "hbos",
    "copod",
    "pca",
]


def discover_csvs(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*.csv") if p.is_file())


def filter_csvs_by_file_list(csvs: list[Path], file_list: Path) -> list[Path]:
    wanted = {
        line.strip()
        for line in file_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    by_name = {p.name: p for p in csvs}
    by_stem = {p.stem: p for p in csvs}
    by_path = {str(p): p for p in csvs}
    selected: list[Path] = []
    missing: list[str] = []
    for item in wanted:
        match = by_path.get(item) or by_name.get(item) or by_stem.get(item)
        if match is None:
            missing.append(item)
        else:
            selected.append(match)
    if missing:
        raise ValueError(f"{len(missing)} file-list entries were not found: {missing[:5]}")
    return sorted(set(selected))


def source_name(path: Path) -> str:
    return path.name.split("_id_")[0].split("_", 1)[1]


def row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def feature_count(path: Path) -> int:
    df = pd.read_csv(path, nrows=5)
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    return len(df.drop(columns=[label_col]).select_dtypes(include=[np.number]).columns)


def exclude_sources_with_large_files(csvs: list[Path], max_rows: int) -> tuple[list[Path], dict[str, int]]:
    source_max_rows: dict[str, int] = {}
    for path in csvs:
        source = source_name(path)
        source_max_rows[source] = max(source_max_rows.get(source, 0), row_count(path))
    excluded = {source for source, rows in source_max_rows.items() if rows > max_rows}
    kept = [path for path in csvs if source_name(path) not in excluded]
    return kept, {source: source_max_rows[source] for source in sorted(excluded)}


def exclude_sources_with_large_matrices(csvs: list[Path], max_cells: int) -> tuple[list[Path], dict[str, int]]:
    source_max_cells: dict[str, int] = {}
    for path in csvs:
        source = source_name(path)
        cells = row_count(path) * feature_count(path)
        source_max_cells[source] = max(source_max_cells.get(source, 0), cells)
    excluded = {source for source, cells in source_max_cells.items() if cells > max_cells}
    kept = [path for path in csvs if source_name(path) not in excluded]
    return kept, {source: source_max_cells[source] for source in sorted(excluded)}


def load_tsb_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).dropna()
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    labels = df[label_col].astype(int).to_numpy()
    features = df.drop(columns=[label_col]).select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if features.ndim != 2 or features.shape[1] < 2:
        raise ValueError(f"{path} is not multivariate after loading: shape={features.shape}")
    if len(np.unique(labels)) < 2:
        raise ValueError(f"{path} has only one label class; AUROC/AUPRC are undefined")
    return features, labels


def downsample_sequence(
    features: np.ndarray,
    labels: np.ndarray,
    max_rows: int | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Uniformly decimate a sequence without using labels to choose samples."""
    original_n = int(len(labels))
    if max_rows is None or original_n <= max_rows:
        return features, labels, original_n, 1
    stride = int(math.ceil(original_n / max_rows))
    return features[::stride], labels[::stride], original_n, stride


def train_indices(labels: np.ndarray, fit_mode: str, min_train: int) -> np.ndarray:
    if fit_mode == "all":
        idx = np.arange(len(labels))
    elif fit_mode == "prefix":
        positives = np.flatnonzero(labels == 1)
        end = int(positives[0]) if len(positives) else len(labels)
        if end < min_train:
            idx = np.flatnonzero(labels == 0)
        else:
            idx = np.arange(end)
    elif fit_mode == "oracle-clean":
        idx = np.flatnonzero(labels == 0)
    else:
        raise ValueError(f"Unknown fit mode: {fit_mode}")

    if len(idx) < min_train:
        raise ValueError(f"Only {len(idx)} training rows available; need at least {min_train}")
    return idx


def metric_dict(
    labels: np.ndarray,
    scores: np.ndarray,
    include_vus: bool = False,
    vus_window: int | None = None,
) -> dict[str, float | int | str | None]:
    labels = labels.astype(int)
    positives = int(labels.sum())
    out: dict[str, float | int | None] = {
        "n": int(len(labels)),
        "positives": positives,
        "anomaly_fraction": float(positives / len(labels)),
        "auroc": None,
        "auprc": None,
    }
    if positives == 0 or positives == len(labels):
        return out
    out["auroc"] = float(roc_auc_score(labels, scores))
    out["auprc"] = float(average_precision_score(labels, scores))
    if include_vus:
        from range_metrics import range_vus_scores

        out.update(range_vus_scores(labels, scores, vus_window))
    return out


def load_pyod(method: str, random_state: int):
    try:
        from pyod.models.copod import COPOD
        from pyod.models.hbos import HBOS
        from pyod.models.iforest import IForest
        from pyod.models.knn import KNN
        from pyod.models.lof import LOF
        from pyod.models.pca import PCA
    except ImportError as exc:
        raise SystemExit("PyOD is required for PyOD methods. Install requirements.txt first.") from exc

    if method == "lof":
        return LOF(n_neighbors=35)
    if method == "knn":
        return KNN(n_neighbors=35)
    if method == "iforest":
        return IForest(n_estimators=250, random_state=random_state)
    if method == "hbos":
        return HBOS()
    if method == "copod":
        return COPOD()
    if method == "pca":
        return PCA(random_state=random_state)
    raise ValueError(method)


def score_pyod(method: str, x_train: np.ndarray, x_all: np.ndarray, random_state: int) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    all_scaled = scaler.transform(x_all)
    model = load_pyod(method, random_state)
    model.fit(train_scaled)
    return np.asarray(model.decision_function(all_scaled), dtype=float)


def score_pendulum(
    x_train: np.ndarray,
    x_all: np.ndarray,
    train_idx: np.ndarray,
    random_state: int,
    **kwargs,
) -> np.ndarray:
    train_entities = np.zeros(len(x_train), dtype=int)
    all_entities = np.zeros(len(x_all), dtype=int)
    model = MultiConceptPendulumDetector(random_state=random_state, **kwargs)
    model.fit(x_train, train_entities, train_idx)
    return model.decision_function(x_all, all_entities, np.arange(len(x_all)))


def score_sequence_ae(x_train: np.ndarray, x_all: np.ndarray, random_state: int) -> np.ndarray:
    from neural_baselines import SequenceAutoencoderDetector

    model = SequenceAutoencoderDetector(random_state=random_state)
    model.fit(x_train, np.zeros(len(x_train), dtype=int))
    return model.decision_function(x_all, np.zeros(len(x_all), dtype=int))


def score_static_ae(x_train: np.ndarray, x_all: np.ndarray, random_state: int) -> np.ndarray:
    from neural_baselines import StaticAutoencoderDetector

    model = StaticAutoencoderDetector(random_state=random_state)
    model.fit(x_train)
    return model.decision_function(x_all)


def score_deep_svdd_kmeans(x_train: np.ndarray, x_all: np.ndarray, random_state: int) -> np.ndarray:
    from neural_baselines import MultiSphereDeepSVDDDetector

    model = MultiSphereDeepSVDDDetector(
        pretrain_epochs=8,
        svdd_epochs=12,
        random_state=random_state,
    )
    model.fit(x_train)
    return model.decision_function(x_all)


def score_hybrid(x_train: np.ndarray, x_all: np.ndarray, train_idx: np.ndarray, random_state: int) -> np.ndarray:
    from neural_baselines import HybridPendulumSequenceDetector

    model = HybridPendulumSequenceDetector(random_state=random_state)
    model.fit(x_train, np.zeros(len(x_train), dtype=int), train_idx)
    return model.decision_function(x_all, np.zeros(len(x_all), dtype=int), np.arange(len(x_all)))


def score_window_knn(x_train: np.ndarray, x_all: np.ndarray, random_state: int) -> np.ndarray:
    del random_state
    from neural_baselines import WindowKNNDetector

    model = WindowKNNDetector()
    model.fit(x_train, np.zeros(len(x_train), dtype=int))
    return model.decision_function(x_all, np.zeros(len(x_all), dtype=int))


def score_forecast_mlp(x_train: np.ndarray, x_all: np.ndarray, random_state: int) -> np.ndarray:
    from neural_baselines import ForecastingMLPDetector

    model = ForecastingMLPDetector(random_state=random_state)
    model.fit(x_train, np.zeros(len(x_train), dtype=int))
    return model.decision_function(x_all, np.zeros(len(x_all), dtype=int))


def score_enhanced_pendulum(
    x_train: np.ndarray,
    x_all: np.ndarray,
    train_idx: np.ndarray,
    random_state: int,
    **kwargs,
) -> np.ndarray:
    from neural_baselines import EnhancedPendulumEnsembleDetector

    model = EnhancedPendulumEnsembleDetector(random_state=random_state, **kwargs)
    model.fit(x_train, np.zeros(len(x_train), dtype=int), train_idx)
    return model.decision_function(x_all, np.zeros(len(x_all), dtype=int), np.arange(len(x_all)))


def method_scorer(method: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
    if method == "pendulum":
        return score_pendulum
    if method in PENDULUM_ABLATIONS:
        return lambda x_train, x_all, train_idx, random_state: score_pendulum(
            x_train, x_all, train_idx, random_state, **PENDULUM_ABLATIONS[method]
        )
    if method == "static_ae":
        return lambda x_train, x_all, train_idx, random_state: score_static_ae(x_train, x_all, random_state)
    if method == "deep_svdd_kmeans":
        return lambda x_train, x_all, train_idx, random_state: score_deep_svdd_kmeans(x_train, x_all, random_state)
    if method == "sequence_ae":
        return lambda x_train, x_all, train_idx, random_state: score_sequence_ae(x_train, x_all, random_state)
    if method in {"hybrid", "pendulum_seqae"}:
        return score_hybrid
    if method == "window_knn":
        return lambda x_train, x_all, train_idx, random_state: score_window_knn(x_train, x_all, random_state)
    if method == "forecast_mlp":
        return lambda x_train, x_all, train_idx, random_state: score_forecast_mlp(x_train, x_all, random_state)
    if method == "enhanced_pendulum":
        return score_enhanced_pendulum
    if method in ENHANCED_ABLATIONS:
        return lambda x_train, x_all, train_idx, random_state: score_enhanced_pendulum(
            x_train, x_all, train_idx, random_state, **ENHANCED_ABLATIONS[method]
        )
    if method in PYOD_METHODS:
        return lambda x_train, x_all, train_idx, random_state: score_pyod(method, x_train, x_all, random_state)
    raise ValueError(f"Unknown method: {method}")


def macro_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    ok = [r for r in rows if r["status"] == "ok"]
    by_method: dict[str, dict[str, object]] = {}
    for method in sorted({str(r["method"]) for r in ok}):
        group = [r for r in ok if r["method"] == method]
        aurocs = [float(r["auroc"]) for r in group if not math.isnan(float(r["auroc"]))]
        auprcs = [float(r["auprc"]) for r in group if not math.isnan(float(r["auprc"]))]
        by_method[method] = {
            "files": len(group),
            "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
            "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
            "median_auroc": float(np.median(aurocs)) if aurocs else None,
            "median_auprc": float(np.median(auprcs)) if auprcs else None,
        }
    return {
        "ok_runs": len(ok),
        "failed_runs": len(rows) - len(ok),
        "by_method": by_method,
    }


def write_outputs(
    rows: list[dict[str, object]],
    out_dir: Path,
    data_dir: Path,
    fit_mode: str,
    methods: list[str],
    limit: int,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "tsb_ad_m_results.csv"
    summary_path = out_dir / "tsb_ad_m_summary.json"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    summary = {
        "data_dir": str(data_dir),
        "fit_mode": fit_mode,
        "methods": methods,
        "limit": limit,
        "summary": macro_summary(rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return results_path, summary_path


def load_existing_rows(out_dir: Path) -> list[dict[str, object]]:
    results_path = out_dir / "tsb_ad_m_results.csv"
    if not results_path.exists():
        return []
    df = pd.read_csv(results_path)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def completed_keys(rows: list[dict[str, object]]) -> set[tuple[str, str, str]]:
    keys = set()
    for row in rows:
        if row.get("status") in {"ok", "failed", "load_failed", "skipped"}:
            keys.add((str(row.get("file")), str(row.get("method")), str(row.get("fit_mode"))))
    return keys


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=base_dir / "data" / "TSB-AD-M")
    parser.add_argument("--out-dir", default=base_dir / "results")
    parser.add_argument("--methods", nargs="+", default=["pendulum", "sequence_ae", "hybrid"])
    parser.add_argument("--fit-mode", choices=["all", "prefix", "oracle-clean"], default="all")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-train", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=37)
    parser.add_argument("--include-vus", action="store_true")
    parser.add_argument("--vus-window", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse completed ok rows in the output CSV.")
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help="Optional newline-separated CSV names, stems, or paths to run.",
    )
    parser.add_argument(
        "--exclude-sources",
        nargs="*",
        default=[],
        help="Optional TSB-AD-M source families to skip, e.g. MITDB GHL.",
    )
    parser.add_argument(
        "--include-sources",
        nargs="*",
        default=[],
        help="Optional TSB-AD-M source families to include, e.g. MSL SMD GHL.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Skip files with more rows than this after loading.",
    )
    parser.add_argument(
        "--downsample-max-rows",
        type=int,
        default=None,
        help=(
            "Uniformly decimate each loaded sequence to at most this many rows. "
            "The sampling stride is fixed from sequence length and does not use labels."
        ),
    )
    parser.add_argument(
        "--whole-source-max-rows",
        type=int,
        default=None,
        help=(
            "Exclude an entire source family if any of its files has more rows than this. "
            "Use this to avoid reporting partial source-family results."
        ),
    )
    parser.add_argument(
        "--whole-source-max-cells",
        type=int,
        default=None,
        help=(
            "Exclude an entire source family if any of its files has more "
            "rows times numeric features than this."
        ),
    )
    args = parser.parse_args()

    methods = [m.lower() for m in args.methods]
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Available: {ALL_METHODS}")

    csvs = discover_csvs(Path(args.data_dir))
    if args.file_list is not None:
        csvs = filter_csvs_by_file_list(csvs, args.file_list)
    included_sources = {s.lower() for s in args.include_sources}
    if included_sources:
        csvs = [p for p in csvs if source_name(p).lower() in included_sources]
    excluded_sources = {s.lower() for s in args.exclude_sources}
    if excluded_sources:
        csvs = [p for p in csvs if source_name(p).lower() not in excluded_sources]
    if args.whole_source_max_rows is not None:
        csvs, source_exclusions = exclude_sources_with_large_files(csvs, args.whole_source_max_rows)
        if source_exclusions:
            excluded_text = ", ".join(f"{source}({rows})" for source, rows in source_exclusions.items())
            print(
                "Excluding whole source families above "
                f"--whole-source-max-rows={args.whole_source_max_rows}: {excluded_text}",
                flush=True,
            )
    if args.whole_source_max_cells is not None:
        csvs, source_exclusions = exclude_sources_with_large_matrices(csvs, args.whole_source_max_cells)
        if source_exclusions:
            excluded_text = ", ".join(f"{source}({cells})" for source, cells in source_exclusions.items())
            print(
                "Excluding whole source families above "
                f"--whole-source-max-cells={args.whole_source_max_cells}: {excluded_text}",
                flush=True,
            )
    if args.limit:
        csvs = csvs[: args.limit]
    if not csvs:
        raise SystemExit(
            f"No CSV files found under {args.data_dir}. Run download_tsb_ad_m.py first or pass --data-dir."
        )

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    rows: list[dict[str, object]] = load_existing_rows(out_dir) if args.resume else []
    done = completed_keys(rows)
    if rows:
        print(f"Resuming from {len(rows)} existing rows in {out_dir}", flush=True)
    for file_idx, csv_path in enumerate(csvs, start=1):
        print(f"[{file_idx}/{len(csvs)}] {csv_path.name}", flush=True)
        try:
            x_all, labels = load_tsb_csv(csv_path)
            original_n = int(len(labels))
            downsample_stride = 1
            if args.max_rows is not None and len(labels) > args.max_rows:
                print(f"  skipped load: {len(labels)} rows exceeds --max-rows={args.max_rows}", flush=True)
                for method in methods:
                    key = (str(csv_path), method, args.fit_mode)
                    if key in done:
                        continue
                    rows.append(
                        {
                            "file": str(csv_path),
                            "dataset": csv_path.stem,
                            "method": method,
                            "fit_mode": args.fit_mode,
                            "status": "skipped",
                            "n": int(len(labels)),
                            "error": f"row count exceeds max_rows={args.max_rows}",
                        }
                    )
                    done.add(key)
                write_outputs(rows, out_dir, data_dir, args.fit_mode, methods, args.limit)
                continue
            x_all, labels, original_n, downsample_stride = downsample_sequence(
                x_all,
                labels,
                args.downsample_max_rows,
            )
            if downsample_stride > 1:
                print(
                    f"  subsampled {original_n} -> {len(labels)} rows "
                    f"(stride={downsample_stride})",
                    flush=True,
                )
            idx = train_indices(labels, args.fit_mode, args.min_train)
        except Exception as exc:
            for method in methods:
                rows.append({"file": str(csv_path), "method": method, "status": "load_failed", "error": str(exc)})
            write_outputs(rows, out_dir, data_dir, args.fit_mode, methods, args.limit)
            continue

        x_train = x_all[idx]
        for method in methods:
            key = (str(csv_path), method, args.fit_mode)
            if key in done:
                print(f"  {method:<12} skipped (resume)", flush=True)
                continue
            start = time.perf_counter()
            try:
                scores = method_scorer(method)(x_train, x_all, idx, args.random_state)
                metrics = metric_dict(labels, scores, args.include_vus, args.vus_window)
                rows.append(
                    {
                        "file": str(csv_path),
                        "dataset": csv_path.stem,
                        "method": method,
                        "fit_mode": args.fit_mode,
                        "status": "ok",
                        "n_features": int(x_all.shape[1]),
                        "train_rows": int(len(idx)),
                        "original_n": original_n,
                        "downsample_stride": downsample_stride,
                        "seconds": float(time.perf_counter() - start),
                        **metrics,
                    }
                )
                done.add(key)
                auroc = metrics["auroc"]
                auprc = metrics["auprc"]
                print(f"  {method:<12} AUROC={auroc:.4f} AUPRC={auprc:.4f}", flush=True)
                write_outputs(rows, out_dir, data_dir, args.fit_mode, methods, args.limit)
            except Exception as exc:
                rows.append(
                    {
                        "file": str(csv_path),
                        "dataset": csv_path.stem,
                        "method": method,
                        "fit_mode": args.fit_mode,
                        "status": "failed",
                        "error": str(exc),
                        "seconds": float(time.perf_counter() - start),
                    }
                )
                print(f"  {method:<12} failed: {exc}", flush=True)
                write_outputs(rows, out_dir, data_dir, args.fit_mode, methods, args.limit)

    results_path, summary_path = write_outputs(rows, out_dir, data_dir, args.fit_mode, methods, args.limit)
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
