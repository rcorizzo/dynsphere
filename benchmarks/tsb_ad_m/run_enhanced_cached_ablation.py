#!/usr/bin/env python3
"""Cached component ablation for DynSphere++ on TSB-AD-M source families."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neural_baselines import (  # noqa: E402
    SequenceAutoencoderDetector,
    WindowKNNDetector,
    _fit_score_scale,
    _robust_z,
)
from pendulum_detector import MultiConceptPendulumDetector  # noqa: E402
from run_tsb_ad_m import (  # noqa: E402
    discover_csvs,
    downsample_sequence,
    load_tsb_csv,
    metric_dict,
    source_name,
    train_indices,
)


DEFAULT_WEIGHTS = {
    "pendulum": 0.35,
    "sequence": 1.0,
    "window": 0.2,
    "static": 0.15,
}

VARIANTS = {
    "DynSphere++": ("pendulum", "sequence", "window", "static"),
    "w/o pendulum": ("sequence", "window", "static"),
    "w/o sequence": ("pendulum", "window", "static"),
    "w/o window": ("pendulum", "sequence", "static"),
    "w/o static": ("pendulum", "sequence", "window"),
}

SEQUENCE_FREE_VARIANTS = {
    "DynSphere++": ("pendulum", "window", "static"),
    "w/o pendulum": ("window", "static"),
    "w/o window": ("pendulum", "static"),
    "w/o static": ("pendulum", "window"),
}


def scaled_component(train_scores: np.ndarray, all_scores: np.ndarray) -> np.ndarray:
    median, scale = _fit_score_scale(np.asarray(train_scores, dtype=float))
    return _robust_z(np.asarray(all_scores, dtype=float), median, scale)


def fit_components(
    x_train: np.ndarray,
    x_all: np.ndarray,
    train_times: np.ndarray,
    random_state: int,
    needed_components: set[str],
) -> dict[str, np.ndarray]:
    del train_times
    train_entities = np.zeros(len(x_train), dtype=int)
    all_entities = np.zeros(len(x_all), dtype=int)
    components = {}

    if "pendulum" in needed_components:
        pendulum = MultiConceptPendulumDetector(random_state=random_state)
        pendulum.fit(x_train, train_entities)
        pendulum_train = pendulum.decision_function(x_train, train_entities)
        pendulum_all = pendulum.decision_function(x_all, all_entities)
        components["pendulum"] = scaled_component(pendulum_train, pendulum_all)

    if "sequence" in needed_components:
        sequence = SequenceAutoencoderDetector(random_state=random_state)
        sequence.fit(x_train, train_entities)
        sequence_train = sequence.decision_function(x_train, train_entities)
        sequence_all = sequence.decision_function(x_all, all_entities)
        components["sequence"] = scaled_component(sequence_train, sequence_all)

    if "window" in needed_components:
        window = WindowKNNDetector()
        window.fit(x_train, train_entities)
        window_train = window.decision_function(x_train, train_entities)
        window_all = window.decision_function(x_all, all_entities)
        components["window"] = scaled_component(window_train, window_all)

    if "static" in needed_components:
        scaler = StandardScaler()
        xs_train = scaler.fit_transform(x_train)
        xs_all = scaler.transform(x_all)
        static = IsolationForest(n_estimators=250, random_state=random_state)
        static.fit(xs_train)
        static_train = -static.score_samples(xs_train)
        static_all = -static.score_samples(xs_all)
        components["static"] = scaled_component(static_train, static_all)

    return components


def combine(components: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    total = np.zeros(len(next(iter(components.values()))), dtype=float)
    for name in names:
        total += DEFAULT_WEIGHTS[name] * components[name]
    return total


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    df = pd.DataFrame([row for row in rows if row.get("status") == "ok"])
    if df.empty:
        return {"ok_runs": 0, "failed_runs": len(rows), "by_method": {}}

    by_method = {}
    for method, group in df.groupby("method", sort=True):
        by_method[method] = {
            "files": int(len(group)),
            "macro_auroc": float(group["auroc"].mean()),
            "macro_vus_roc": float(group["vus_roc"].mean()) if "vus_roc" in group else None,
            "median_auroc": float(group["auroc"].median()),
            "median_vus_roc": float(group["vus_roc"].median()) if "vus_roc" in group else None,
        }

    by_source_method = {}
    for (source, method), group in df.groupby(["source", "method"], sort=True):
        by_source_method[f"{source}/{method}"] = {
            "files": int(len(group)),
            "macro_auroc": float(group["auroc"].mean()),
            "macro_vus_roc": float(group["vus_roc"].mean()) if "vus_roc" in group else None,
        }

    return {
        "ok_runs": int(len(df)),
        "failed_runs": len(rows) - int(len(df)),
        "by_method": by_method,
        "by_source_method": by_source_method,
    }


def write_outputs(rows: list[dict[str, object]], out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "tsb_ad_m_results.csv", index=False)
    summary = {
        "data_dir": str(args.data_dir),
        "sources": args.include_sources,
        "fit_mode": args.fit_mode,
        "weights": DEFAULT_WEIGHTS,
        "variant_set": args.variant_set,
        "variants": {key: list(value) for key, value in args.variants.items()},
        "summary": summarize(rows),
    }
    (out_dir / "tsb_ad_m_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data" / "TSB-AD-M")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "results_enhanced_cached_ablation_msl_smap_smd_official_vus",
    )
    parser.add_argument("--include-sources", nargs="+", default=["MSL", "SMAP", "SMD"])
    parser.add_argument("--fit-mode", choices=["all", "prefix", "oracle-clean"], default="all")
    parser.add_argument("--min-train", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=37)
    parser.add_argument("--include-vus", action="store_true")
    parser.add_argument("--vus-window", type=int, default=None)
    parser.add_argument("--downsample-max-rows", type=int, default=None)
    parser.add_argument(
        "--variant-set",
        choices=["full", "sequence-free"],
        default="full",
        help="Ablate the original four-component fusion or the winning sequence-free fusion.",
    )
    args = parser.parse_args()
    args.variants = SEQUENCE_FREE_VARIANTS if args.variant_set == "sequence-free" else VARIANTS

    wanted_sources = {source.lower() for source in args.include_sources}
    csvs = [path for path in discover_csvs(args.data_dir) if source_name(path).lower() in wanted_sources]
    if not csvs:
        raise SystemExit(f"No matching CSV files found under {args.data_dir}")

    rows: list[dict[str, object]] = []
    for file_idx, csv_path in enumerate(csvs, start=1):
        print(f"[{file_idx}/{len(csvs)}] {csv_path.name}", flush=True)
        source = source_name(csv_path)
        started = time.time()
        try:
            x_all, labels = load_tsb_csv(csv_path)
            x_all, labels, original_n, downsample_stride = downsample_sequence(
                x_all,
                labels,
                args.downsample_max_rows,
            )
            idx = train_indices(labels, args.fit_mode, args.min_train)
            x_train = x_all[idx]
            needed_components = {name for names in args.variants.values() for name in names}
            components = fit_components(x_train, x_all, idx, args.random_state, needed_components)
            seconds = time.time() - started

            for method, component_names in args.variants.items():
                scores = combine(components, component_names)
                metrics = metric_dict(labels, scores, args.include_vus, args.vus_window)
                rows.append(
                    {
                        "file": str(csv_path),
                        "dataset": csv_path.stem,
                        "source": source,
                        "method": method,
                        "fit_mode": args.fit_mode,
                        "status": "ok",
                        "n_features": int(x_all.shape[1]),
                        "train_rows": int(len(idx)),
                        "original_n": int(original_n),
                        "downsample_stride": int(downsample_stride),
                        "seconds": float(seconds),
                        **metrics,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            seconds = time.time() - started
            print(f"  failed: {exc}", flush=True)
            for method in args.variants:
                rows.append(
                    {
                        "file": str(csv_path),
                        "dataset": csv_path.stem,
                        "source": source,
                        "method": method,
                        "fit_mode": args.fit_mode,
                        "status": "failed",
                        "seconds": float(seconds),
                        "error": str(exc),
                    }
                )
        write_outputs(rows, args.out_dir, args)

    write_outputs(rows, args.out_dir, args)
    print(f"Wrote {args.out_dir / 'tsb_ad_m_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
