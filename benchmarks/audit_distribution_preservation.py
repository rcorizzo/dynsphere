#!/usr/bin/env python3
"""Audit whether labeled trajectory anomalies are marginally distinguishable.

This script is a diagnostic, not a detector comparison.  It deliberately uses
labels to test the strongest benchmark claim: anomalous rows should be hard to
separate from normal rows without temporal context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _subsample(values: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= maximum:
        return values
    return values[rng.choice(len(values), size=maximum, replace=False)]


def rbf_mmd_squared(
    normal: np.ndarray,
    anomalous: np.ndarray,
    rng: np.random.Generator,
    maximum: int = 1500,
) -> tuple[float, float]:
    """Return biased RBF MMD^2 and the median-heuristic bandwidth."""

    normal = _subsample(normal, maximum, rng)
    anomalous = _subsample(anomalous, maximum, rng)
    combined = np.vstack([normal, anomalous])
    bandwidth_sample = _subsample(combined, min(700, len(combined)), rng)
    squared = np.sum((bandwidth_sample[:, None] - bandwidth_sample[None, :]) ** 2, axis=2)
    positive = squared[squared > 0.0]
    bandwidth_squared = float(np.median(positive)) if len(positive) else 1.0
    bandwidth_squared = max(bandwidth_squared, 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distance = np.sum((a[:, None] - b[None, :]) ** 2, axis=2)
        return np.exp(-distance / (2.0 * bandwidth_squared))

    statistic = float(
        kernel(normal, normal).mean()
        + kernel(anomalous, anomalous).mean()
        - 2.0 * kernel(normal, anomalous).mean()
    )
    return statistic, float(np.sqrt(bandwidth_squared))


def grouped_row_classifier_scores(
    x: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two entities are required for the grouped classifier audit.")
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    scores = np.zeros(len(x), dtype=float)
    for fold, (fit_idx, score_idx) in enumerate(splitter.split(x, labels, groups)):
        model = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state + fold,
        )
        model.fit(x[fit_idx], labels[fit_idx])
        scores[score_idx] = model.predict_proba(x[score_idx])[:, 1]
    return scores


def exact_normal_duplicate_fraction(x: np.ndarray, labels: np.ndarray) -> float:
    normal_rows = {tuple(row) for row in x[labels == 0]}
    anomalous = x[labels == 1]
    if len(anomalous) == 0:
        return float("nan")
    return float(np.mean([tuple(row) in normal_rows for row in anomalous]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--feature-prefix", default="x_")
    parser.add_argument("--random-state", type=int, default=37)
    parser.add_argument("--max-mmd-rows", type=int, default=1500)
    args = parser.parse_args()

    train = pd.read_csv(args.train).sort_values(["entity_id", "time"]).reset_index(drop=True)
    test = pd.read_csv(args.test).sort_values(["entity_id", "time"]).reset_index(drop=True)
    feature_columns = [column for column in train if column.startswith(args.feature_prefix)]
    if not feature_columns:
        raise SystemExit(f"No feature columns found with prefix {args.feature_prefix!r}.")

    x_train = train[feature_columns].to_numpy(dtype=float)
    x_test = test[feature_columns].to_numpy(dtype=float)
    labels = test["is_anomaly"].to_numpy(dtype=int)
    groups = test["entity_id"].to_numpy()
    if labels.sum() == 0:
        raise SystemExit("The test file contains no anomalous rows.")

    scaler = StandardScaler().fit(x_train)
    train_scaled = scaler.transform(x_train)
    test_scaled = scaler.transform(x_test)
    rng = np.random.default_rng(args.random_state)

    neighbors = NearestNeighbors(n_neighbors=1).fit(train_scaled)
    distances = neighbors.kneighbors(test_scaled, return_distance=True)[0][:, 0]
    classifier_scores = grouped_row_classifier_scores(
        test_scaled,
        labels,
        groups,
        args.random_state,
    )
    mmd, bandwidth = rbf_mmd_squared(
        test_scaled[labels == 0],
        test_scaled[labels == 1],
        rng,
        maximum=args.max_mmd_rows,
    )

    result: dict[str, object] = {
        "train": str(Path(args.train)),
        "test": str(Path(args.test)),
        "rows": int(len(test)),
        "anomaly_rows": int(labels.sum()),
        "exact_duplicate_among_observed_normal_rows": exact_normal_duplicate_fraction(x_test, labels),
        "nearest_train_distance": {
            "normal_median": float(np.median(distances[labels == 0])),
            "anomaly_median": float(np.median(distances[labels == 1])),
            "rowwise_auroc": float(roc_auc_score(labels, distances)),
            "rowwise_auprc": float(average_precision_score(labels, distances)),
        },
        "grouped_supervised_row_classifier": {
            "auroc": float(roc_auc_score(labels, classifier_scores)),
            "auprc": float(average_precision_score(labels, classifier_scores)),
            "cross_validation": "GroupKFold by entity",
        },
        "rbf_mmd_squared": mmd,
        "rbf_bandwidth": bandwidth,
        "by_anomaly_type": {},
    }

    for anomaly_type in sorted(set(test.loc[labels == 1, "anomaly_type"])):
        subset = (labels == 0) | (test["anomaly_type"].to_numpy() == anomaly_type)
        subset_labels = labels[subset]
        result["by_anomaly_type"][str(anomaly_type)] = {
            "rows": int(np.sum(labels & (test["anomaly_type"].to_numpy() == anomaly_type))),
            "nearest_train_auroc": float(roc_auc_score(subset_labels, distances[subset])),
            "classifier_auroc": float(roc_auc_score(subset_labels, classifier_scores[subset])),
        }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
