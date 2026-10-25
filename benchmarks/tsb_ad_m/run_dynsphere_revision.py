#!/usr/bin/env python3
"""Detector-preserving DynSphere revision experiments on complete TSB sources."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from pendulum_detector import MultiConceptPendulumDetector, _rolling_entity_max  # noqa: E402
from run_tsb_ad_m import (  # noqa: E402
    declared_prefix_indices,
    discover_csvs,
    filter_csvs_by_file_list,
    load_tsb_csv,
    metric_dict,
    source_name,
    train_indices,
)


SMD22 = "078_SMD_id_22_Facility_tr_500_1st_326.csv"
ADDITIVE_COMPONENTS = (
    "angular",
    "spherical_dynamics",
    "raw_dynamics",
    "state_speed",
    "velocity",
    "regime_distance",
    "transition",
)
WEIGHT_ATTRIBUTES = (
    "angular_weight",
    "dynamic_weight",
    "raw_dynamic_weight",
    "energy_weight",
    "velocity_weight",
    "cluster_weight",
    "transition_weight",
)


def revision_train_indices(
    path: Path,
    labels: np.ndarray,
    min_train: int,
    source_stride: int = 1,
) -> tuple[np.ndarray, str]:
    try:
        return (
            declared_prefix_indices(path, labels, min_train, source_stride=source_stride),
            "declared-prefix",
        )
    except ValueError:
        if path.name != SMD22:
            raise
        return train_indices(labels, "prefix", min_train), "maximal-clean-prefix"


def model_score(
    model: MultiConceptPendulumDetector,
    x: np.ndarray,
    entities: np.ndarray,
    **overrides: float | int,
) -> tuple[np.ndarray, float]:
    previous = {name: getattr(model, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(model, name, value)
        started = time.perf_counter()
        scores = model.decision_function(x, entities)
        return scores, time.perf_counter() - started
    finally:
        for name, value in previous.items():
            setattr(model, name, value)


def fit_model(
    x_train: np.ndarray,
    random_state: int,
    **configuration: float | int,
) -> tuple[MultiConceptPendulumDetector, float]:
    entities = np.zeros(len(x_train), dtype=int)
    model = MultiConceptPendulumDetector(random_state=random_state, **configuration)
    started = time.perf_counter()
    model.fit(x_train, entities)
    return model, time.perf_counter() - started


def result_row(
    *,
    path: Path,
    source: str,
    seed: int,
    experiment: str,
    scenario: str,
    method_id: str,
    fit_protocol: str,
    train_rows: int,
    labels: np.ndarray,
    scores: np.ndarray,
    fit_seconds: float,
    score_seconds: float,
) -> dict[str, object]:
    return {
        "file": str(path),
        "dataset": path.stem,
        "source": source,
        "seed": seed,
        "experiment": experiment,
        "scenario": scenario,
        "method_id": method_id,
        "fit_protocol": fit_protocol,
        "status": "ok",
        "train_rows": train_rows,
        "n": len(labels),
        "fit_seconds": fit_seconds,
        "score_seconds": score_seconds,
        "total_seconds": fit_seconds + score_seconds,
        **metric_dict(labels, scores),
    }


def source_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    frame = pd.DataFrame(row for row in rows if row.get("status") == "ok")
    metrics = [
        "auroc",
        "auprc",
        "pauc_fpr_05",
        "event_recall_top_1",
        "event_precision_top_1",
        "total_seconds",
    ]
    grouped = frame.groupby(
        ["experiment", "scenario", "source", "seed", "method_id"],
        dropna=False,
    )[metrics].agg(["count", "mean", "std"])
    grouped = grouped.reset_index()
    grouped.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    return grouped.where(pd.notnull(grouped), None).to_dict(orient="records")


def completed_keys(rows: list[dict[str, object]]) -> set[tuple[object, ...]]:
    return {
        (
            row.get("file"),
            row.get("seed"),
            row.get("experiment"),
            row.get("scenario"),
            row.get("method_id"),
        )
        for row in rows
        if row.get("status") == "ok"
    }


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, object]],
    component_rows: list[dict[str, object]],
    correlation_rows: list[dict[str, object]],
    example_rows: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(component_rows).to_csv(out_dir / "component_metrics.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(out_dir / "component_correlations.csv", index=False)
    pd.DataFrame(example_rows).to_csv(out_dir / "representative_cases.csv", index=False)
    summary = source_summary(rows)
    pd.DataFrame(summary).to_csv(out_dir / "source_summary.csv", index=False)
    payload = {
        "experiments": args.experiments,
        "seeds": args.seeds,
        "sources": args.include_sources,
        "exception": {
            "file": SMD22,
            "protocol": "maximal contiguous clean prefix",
        },
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def append_component_analysis(
    *,
    path: Path,
    source: str,
    seed: int,
    labels: np.ndarray,
    components: dict[str, np.ndarray],
    component_rows: list[dict[str, object]],
    correlation_rows: list[dict[str, object]],
    example_rows: list[dict[str, object]],
) -> None:
    analysis_names = (*ADDITIVE_COMPONENTS, "pre_propagation_score", "score")
    for name in analysis_names:
        values = np.asarray(components[name], dtype=float)
        correlation = spearmanr(values, components["score"]).statistic
        component_rows.append(
            {
                "file": str(path),
                "dataset": path.stem,
                "source": source,
                "seed": seed,
                "component": name,
                "auroc": float(roc_auc_score(labels, values)),
                "auprc": float(average_precision_score(labels, values)),
                "spearman_with_final": float(correlation) if np.isfinite(correlation) else None,
                "normal_mean": float(values[labels == 0].mean()),
                "anomaly_mean": float(values[labels == 1].mean()),
                **{
                    key: value
                    for key, value in metric_dict(labels, values).items()
                    if key.startswith("event_") or key == "pauc_fpr_05"
                },
            }
        )
    for left_index, left in enumerate(ADDITIVE_COMPONENTS):
        for right in ADDITIVE_COMPONENTS[left_index + 1 :]:
            value = spearmanr(components[left], components[right]).statistic
            correlation_rows.append(
                {
                    "file": str(path),
                    "dataset": path.stem,
                    "source": source,
                    "seed": seed,
                    "component_left": left,
                    "component_right": right,
                    "spearman": float(value) if np.isfinite(value) else None,
                }
            )

    final_score = np.asarray(components["score"], dtype=float)
    local_values = np.column_stack([components[name] for name in ADDITIVE_COMPONENTS])
    dominant = np.asarray(ADDITIVE_COMPONENTS)[np.argmax(local_values, axis=1)]
    for case_type, mask in (("true_positive", labels == 1), ("false_positive", labels == 0)):
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        selected = indices[np.argsort(final_score[indices])[-10:]][::-1]
        for rank, row_index in enumerate(selected, start=1):
            example_rows.append(
                {
                    "file": str(path),
                    "dataset": path.stem,
                    "source": source,
                    "seed": seed,
                    "case_type": case_type,
                    "rank": rank,
                    "row_index": int(row_index),
                    "label": int(labels[row_index]),
                    "score": float(final_score[row_index]),
                    "dominant_component": str(dominant[row_index]),
                    "selected_concept": int(components["selected_concept"][row_index]),
                    "assigned_concept": int(components["assigned_concept"][row_index]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data" / "TSB-AD-M")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["ablation", "components", "sensitivity", "robustness"],
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[37])
    parser.add_argument("--include-sources", nargs="+", default=["MSL", "SMAP", "SMD"])
    parser.add_argument("--file-list", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-train", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    wanted = {source.lower() for source in args.include_sources}
    paths = [path for path in discover_csvs(args.data_dir) if source_name(path).lower() in wanted]
    if args.file_list is not None:
        paths = filter_csvs_by_file_list(paths, args.file_list)
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No matching TSB-AD-M files found.")

    result_path = args.out_dir / "results.csv"
    rows = pd.read_csv(result_path).where(pd.notnull, None).to_dict(orient="records") if args.resume and result_path.exists() else []
    def load_optional_csv(name: str) -> list[dict[str, object]]:
        path = args.out_dir / name
        if not args.resume or not path.exists() or path.stat().st_size == 0:
            return []
        return pd.read_csv(path).where(pd.notnull, None).to_dict(orient="records")

    component_rows = load_optional_csv("component_metrics.csv")
    correlation_rows = load_optional_csv("component_correlations.csv")
    example_rows = load_optional_csv("representative_cases.csv")
    done = completed_keys(rows)
    component_done = {
        (str(row.get("file")), int(row.get("seed")))
        for row in component_rows
        if row.get("file") is not None and row.get("seed") is not None
    }

    for seed in args.seeds:
        for file_index, path in enumerate(paths, start=1):
            print(f"seed={seed} [{file_index}/{len(paths)}] {path.name}", flush=True)
            source = source_name(path)
            x_all, labels = load_tsb_csv(path)
            train_idx, fit_protocol = revision_train_indices(path, labels, args.min_train)
            x_train = x_all[train_idx]
            all_entities = np.zeros(len(x_all), dtype=int)

            base_model, base_fit_seconds = fit_model(x_train, seed)
            base_score_started = time.perf_counter()
            base_components = base_model.decision_components(x_all, all_entities)
            base_score_seconds = time.perf_counter() - base_score_started
            base_scores = base_components["score"]

            if "ablation" in args.experiments:
                variants: list[tuple[str, dict[str, float | int], MultiConceptPendulumDetector, float]] = [
                    ("dynsphere", {}, base_model, base_fit_seconds),
                    ("dynsphere_no_transition", {"transition_weight": 0.0}, base_model, base_fit_seconds),
                    ("dynsphere_no_energy", {"energy_weight": 0.0}, base_model, base_fit_seconds),
                    ("dynsphere_no_raw", {"raw_dynamic_weight": 0.0}, base_model, base_fit_seconds),
                ]
                one_model, one_fit_seconds = fit_model(
                    x_train,
                    seed,
                    n_concepts=1,
                    transition_weight=0.0,
                )
                variants.append(("dynsphere_no_concepts", {}, one_model, one_fit_seconds))
                for method_id, overrides, model, fit_seconds in variants:
                    key = (str(path), seed, "ablation", "default", method_id)
                    if key in done:
                        continue
                    if method_id == "dynsphere":
                        scores = base_scores
                        score_seconds = base_score_seconds
                    else:
                        scores, score_seconds = model_score(model, x_all, all_entities, **overrides)
                    rows.append(
                        result_row(
                            path=path,
                            source=source,
                            seed=seed,
                            experiment="ablation",
                            scenario="default",
                            method_id=method_id,
                            fit_protocol=fit_protocol,
                            train_rows=len(train_idx),
                            labels=labels,
                            scores=scores,
                            fit_seconds=fit_seconds,
                            score_seconds=score_seconds,
                        )
                    )
                    done.add(key)

            if "components" in args.experiments:
                component_key = (str(path), seed)
                if component_key not in component_done:
                    append_component_analysis(
                        path=path,
                        source=source,
                        seed=seed,
                        labels=labels,
                        components=base_components,
                        component_rows=component_rows,
                        correlation_rows=correlation_rows,
                        example_rows=example_rows,
                    )
                    component_done.add(component_key)

            if "sensitivity" in args.experiments:
                settings: list[tuple[str, str, dict[str, float | int], bool]] = [
                    ("default", "default", {}, False),
                    ("concept_count", "concepts_1", {"n_concepts": 1}, True),
                    ("concept_count", "concepts_2", {"n_concepts": 2}, True),
                    ("concept_count", "concepts_8", {"n_concepts": 8}, True),
                    ("ridge", "ridge_0.1", {"ridge_alpha": 0.1}, True),
                    ("ridge", "ridge_10", {"ridge_alpha": 10.0}, True),
                    ("propagation", "radius_0", {"propagation_radius": 0}, False),
                    ("propagation", "radius_4", {"propagation_radius": 4}, False),
                    ("propagation", "radius_16", {"propagation_radius": 16}, False),
                ]
                for attribute in WEIGHT_ATTRIBUTES:
                    default_value = float(getattr(base_model, attribute))
                    for factor in (0.5, 2.0):
                        settings.append(
                            (
                                "component_weight",
                                f"{attribute}_x{factor:g}",
                                {attribute: default_value * factor},
                                False,
                            )
                        )
                for family, setting, configuration, requires_fit in settings:
                    key = (str(path), seed, "sensitivity", family, setting)
                    if key in done:
                        continue
                    if setting == "default":
                        scores = base_scores
                        fit_seconds = base_fit_seconds
                        score_seconds = base_score_seconds
                    elif requires_fit:
                        model, fit_seconds = fit_model(x_train, seed, **configuration)
                        scores, score_seconds = model_score(model, x_all, all_entities)
                    else:
                        fit_seconds = base_fit_seconds
                        scores, score_seconds = model_score(
                            base_model,
                            x_all,
                            all_entities,
                            **configuration,
                        )
                    rows.append(
                        result_row(
                            path=path,
                            source=source,
                            seed=seed,
                            experiment="sensitivity",
                            scenario=family,
                            method_id=setting,
                            fit_protocol=fit_protocol,
                            train_rows=len(train_idx),
                            labels=labels,
                            scores=scores,
                            fit_seconds=fit_seconds,
                            score_seconds=score_seconds,
                        )
                    )
                    done.add(key)

            if "robustness" in args.experiments:
                robustness_settings = [
                    ("baseline", 0.0, 1),
                    ("noise_0.01", 0.01, 1),
                    ("noise_0.05", 0.05, 1),
                    ("noise_0.10", 0.10, 1),
                    ("decimation_2", 0.0, 2),
                    ("decimation_4", 0.0, 4),
                ]
                for setting, noise_level, stride in robustness_settings:
                    key = (str(path), seed, "robustness", setting, "dynsphere")
                    if key in done:
                        continue
                    if setting == "baseline":
                        robust_x, robust_labels = x_all, labels
                        robust_idx, robust_protocol = train_idx, fit_protocol
                        robust_model, fit_seconds = base_model, base_fit_seconds
                        scores, score_seconds = base_scores, base_score_seconds
                    else:
                        robust_x = x_all[::stride].copy()
                        robust_labels = labels[::stride].copy()
                        robust_idx, robust_protocol = revision_train_indices(
                            path,
                            robust_labels,
                            args.min_train,
                            source_stride=stride,
                        )
                        if noise_level > 0.0:
                            scale = np.std(robust_x[robust_idx], axis=0, ddof=0)
                            scale = np.where(scale > 1e-12, scale, 1.0)
                            file_seed = zlib.crc32(path.name.encode("utf-8"))
                            rng = np.random.default_rng(seed + file_seed)
                            robust_x += rng.normal(
                                0.0,
                                noise_level * scale,
                                size=robust_x.shape,
                            )
                        robust_model, fit_seconds = fit_model(robust_x[robust_idx], seed)
                        robust_entities = np.zeros(len(robust_x), dtype=int)
                        scores, score_seconds = model_score(
                            robust_model,
                            robust_x,
                            robust_entities,
                        )
                    rows.append(
                        result_row(
                            path=path,
                            source=source,
                            seed=seed,
                            experiment="robustness",
                            scenario=setting,
                            method_id="dynsphere",
                            fit_protocol=robust_protocol,
                            train_rows=len(robust_idx),
                            labels=robust_labels,
                            scores=scores,
                            fit_seconds=fit_seconds,
                            score_seconds=score_seconds,
                        )
                    )
                    done.add(key)

            write_outputs(
                args.out_dir,
                rows,
                component_rows,
                correlation_rows,
                example_rows,
                args,
            )

    write_outputs(
        args.out_dir,
        rows,
        component_rows,
        correlation_rows,
        example_rows,
        args,
    )
    print(f"Wrote revision artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
