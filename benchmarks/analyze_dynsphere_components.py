#!/usr/bin/env python3
"""Mechanism-level DynSphere component attribution on labeled CSV sequences."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pendulum_detector import MultiConceptPendulumDetector  # noqa: E402
from range_metrics import practical_range_metrics  # noqa: E402


COMPONENTS = (
    "angular",
    "spherical_dynamics",
    "raw_dynamics",
    "state_speed",
    "velocity",
    "regime_distance",
    "transition",
    "pre_propagation_score",
    "score",
)
ADDITIVE_COMPONENTS = COMPONENTS[:7]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame if column.startswith("x_")]
    if not columns:
        raise ValueError("No observed feature columns beginning with 'x_' were found.")
    return sorted(columns)


def entities(frame: pd.DataFrame) -> np.ndarray:
    if "entity_id" in frame:
        return frame["entity_id"].to_numpy()
    return np.zeros(len(frame), dtype=int)


def mechanism_masks(frame: pd.DataFrame, labels: np.ndarray) -> dict[str, np.ndarray]:
    masks = {"overall": np.ones(len(frame), dtype=bool)}
    if "anomaly_type" not in frame:
        return masks
    anomaly_types = sorted(
        value
        for value in frame.loc[labels == 1, "anomaly_type"].dropna().astype(str).unique()
        if value.lower() not in {"none", "normal", "nan"}
    )
    for anomaly_type in anomaly_types:
        masks[anomaly_type] = (labels == 0) | (
            (labels == 1) & frame["anomaly_type"].astype(str).eq(anomaly_type).to_numpy()
        )
    return masks


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    metrics = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }
    metrics.update(practical_range_metrics(labels, scores))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[37, 38, 39])
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    columns = feature_columns(train)
    if columns != feature_columns(test):
        raise ValueError("Training and test feature columns do not match.")
    x_train = train[columns].to_numpy(dtype=float)
    x_test = test[columns].to_numpy(dtype=float)
    train_entities = entities(train)
    test_entities = entities(test)
    labels = test["is_anomaly"].to_numpy(dtype=int)
    masks = mechanism_masks(test, labels)

    metric_rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    unique_top_rows: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []

    for seed in args.seeds:
        print(f"seed={seed}", flush=True)
        model = MultiConceptPendulumDetector(random_state=seed)
        started = time.perf_counter()
        model.fit(x_train, train_entities)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = model.decision_components(x_test, test_entities)
        score_seconds = time.perf_counter() - started
        runtimes.append(
            {"seed": seed, "fit_seconds": fit_seconds, "score_seconds": score_seconds}
        )

        for mechanism, mask in masks.items():
            subset_labels = labels[mask]
            anomaly_mask = subset_labels == 1
            for component in COMPONENTS:
                values = np.asarray(scores[component], dtype=float)[mask]
                metric_rows.append(
                    {
                        "seed": seed,
                        "mechanism": mechanism,
                        "component": component,
                        "n": int(mask.sum()),
                        "positives": int(subset_labels.sum()),
                        **evaluate(subset_labels, values),
                    }
                )
                contribution_rows.append(
                    {
                        "seed": seed,
                        "mechanism": mechanism,
                        "component": component,
                        "normal_mean": float(values[~anomaly_mask].mean()),
                        "anomaly_mean": float(values[anomaly_mask].mean()),
                        "anomaly_median": float(np.median(values[anomaly_mask])),
                    }
                )

            additive = np.column_stack(
                [np.asarray(scores[name], dtype=float)[mask] for name in ADDITIVE_COMPONENTS]
            )
            anomaly_additive = additive[anomaly_mask]
            dominant = np.argmax(anomaly_additive, axis=1)
            for component_index, component in enumerate(ADDITIVE_COMPONENTS):
                contribution_rows.append(
                    {
                        "seed": seed,
                        "mechanism": mechanism,
                        "component": f"dominant::{component}",
                        "normal_mean": None,
                        "anomaly_mean": float(np.mean(dominant == component_index)),
                        "anomaly_median": None,
                    }
                )

        for left_index, left in enumerate(ADDITIVE_COMPONENTS):
            for right in ADDITIVE_COMPONENTS[left_index + 1 :]:
                for subset_name, subset_mask in (
                    ("all", np.ones(len(labels), dtype=bool)),
                    ("normal", labels == 0),
                    ("anomaly", labels == 1),
                ):
                    correlation = spearmanr(
                        np.asarray(scores[left])[subset_mask],
                        np.asarray(scores[right])[subset_mask],
                    ).statistic
                    correlation_rows.append(
                        {
                            "seed": seed,
                            "subset": subset_name,
                            "component_left": left,
                            "component_right": right,
                            "spearman": float(correlation) if np.isfinite(correlation) else None,
                        }
                    )

        final_scores = np.asarray(scores["score"], dtype=float)
        additive_values = np.column_stack([scores[name] for name in ADDITIVE_COMPONENTS])
        dominant_names = np.asarray(ADDITIVE_COMPONENTS)[np.argmax(additive_values, axis=1)]
        for component in COMPONENTS:
            component_values = np.asarray(scores[component], dtype=float)
            for case_type, case_mask in (
                ("true_positive", labels == 1),
                ("false_positive", labels == 0),
            ):
                indices = np.flatnonzero(case_mask)
                selected = indices[np.argsort(component_values[indices])[-25:]][::-1]
                for rank, index in enumerate(selected, start=1):
                    example_rows.append(
                        {
                            "seed": seed,
                            "component": component,
                            "case_type": case_type,
                            "rank": rank,
                            "row_index": int(index),
                            "entity_id": test_entities[index],
                            "time": test.iloc[index]["time"] if "time" in test else index,
                            "anomaly_type": test.iloc[index]["anomaly_type"] if "anomaly_type" in test else None,
                            "component_score": float(component_values[index]),
                            "final_score": float(final_scores[index]),
                            "dominant_component": str(dominant_names[index]),
                        }
                    )

        top_count = max(1, int(np.ceil(0.01 * len(labels))))
        top_sets = {
            name: set(np.argsort(np.asarray(scores[name], dtype=float))[-top_count:])
            for name in ADDITIVE_COMPONENTS
        }
        for component in ADDITIVE_COMPONENTS:
            other_union: set[int] = set()
            for other, indices in top_sets.items():
                if other != component:
                    other_union.update(indices)
            unique = top_sets[component] - other_union
            unique_array = np.fromiter(unique, dtype=int)
            unique_top_rows.append(
                {
                    "seed": seed,
                    "component": component,
                    "top_fraction": 0.01,
                    "top_count": top_count,
                    "unique_count": len(unique),
                    "unique_fraction_of_component_top": len(unique) / top_count,
                    "unique_anomalies": int(labels[unique_array].sum()) if len(unique_array) else 0,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "component_metrics.csv": metric_rows,
        "component_contributions.csv": contribution_rows,
        "component_correlations.csv": correlation_rows,
        "representative_cases.csv": example_rows,
        "unique_top_detections.csv": unique_top_rows,
        "runtime.csv": runtimes,
    }
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(args.out_dir / name, index=False)
    payload = {
        "train": str(args.train),
        "test": str(args.test),
        "seeds": args.seeds,
        "features": columns,
        "mechanisms": list(masks),
        "runtime": runtimes,
    }
    (args.out_dir / "configuration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote component analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
