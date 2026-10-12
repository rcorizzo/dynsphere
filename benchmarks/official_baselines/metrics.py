"""Metrics for private-label evaluation of validated anomaly scores."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from range_metrics import range_vus_scores


def _segments(values: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(values)))
    return segments


def _fixed_budget_events(
    labels: np.ndarray, scores: np.ndarray, budget: float
) -> tuple[int, float, float]:
    if not 0.0 < budget <= 1.0:
        raise ValueError("event_budget must lie in (0, 1]")
    count = max(1, int(math.ceil(len(scores) * budget)))
    ordering = np.argsort(scores, kind="mergesort")
    predictions = np.zeros(len(scores), dtype=int)
    predictions[ordering[-count:]] = 1
    true_events = _segments(labels)
    predicted_events = _segments(predictions)
    recall = (
        sum(1 for start, end in true_events if predictions[start:end].any())
        / len(true_events)
        if true_events
        else 0.0
    )
    precision = (
        sum(1 for start, end in predicted_events if labels[start:end].any())
        / len(predicted_events)
        if predicted_events
        else 0.0
    )
    return count, float(recall), float(precision)


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    event_budget: float,
    vus_window: int | None,
) -> dict[str, object]:
    """Compute threshold-free and fixed-budget metrics after strict validation."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be aligned one-dimensional arrays")
    if len(labels) == 0 or not np.isfinite(scores).all():
        raise ValueError("Metric inputs must be nonempty and finite")
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError("Labels must be binary")
    labels = labels.astype(int, copy=False)
    predictions, event_recall, event_precision = _fixed_budget_events(
        labels, scores, event_budget
    )

    binary_defined = unique == {0, 1}
    if binary_defined:
        auprc: float | None = float(average_precision_score(labels, scores))
        auroc: float | None = float(roc_auc_score(labels, scores))
        pauc: float | None = float(roc_auc_score(labels, scores, max_fpr=0.05))
    else:
        auprc = None
        auroc = None
        pauc = None

    vus = range_vus_scores(
        labels,
        scores,
        sliding_window=vus_window,
        prefer_official=True,
    )
    backend = str(vus.get("vus_backend", "unknown"))
    return {
        "auprc": auprc,
        "auroc": auroc,
        "pauc_fpr_05": pauc,
        "vus_pr": vus.get("vus_pr"),
        "vus_roc": vus.get("vus_roc"),
        "vus_window": vus.get("vus_window"),
        "vus_backend": backend,
        "event_budget": float(event_budget),
        "event_predictions": predictions,
        "event_recall": event_recall,
        "event_precision": event_precision,
        "manuscript_ready": binary_defined and backend == "official_vus",
    }
