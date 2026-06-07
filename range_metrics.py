"""Range/VUS metrics for time-series anomaly detection.

The preferred path uses the official `vus` package:
    pip install vus

When unavailable, the module falls back to a simple buffered-label average over
ROC-AUC and PR-AUC. The fallback is useful for smoke tests but is not an exact
implementation of the TSB/VUS metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def anomaly_segment_lengths(labels: np.ndarray) -> list[int]:
    labels = np.asarray(labels).astype(int)
    lengths: list[int] = []
    start = None
    for i, value in enumerate(labels):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            lengths.append(i - start)
            start = None
    if start is not None:
        lengths.append(len(labels) - start)
    return lengths


def default_sliding_window(labels: np.ndarray) -> int:
    lengths = anomaly_segment_lengths(labels)
    if not lengths:
        return 1
    return max(1, int(round(float(np.median(lengths)))))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    if not finite.all():
        scores = scores.copy()
        scores[~finite] = np.nanmedian(scores[finite]) if finite.any() else 0.0
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi <= lo:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def _segments(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels).astype(int)
    segments: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(labels):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


def point_adjust_predictions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Apply the common point-adjustment rule to binary predictions.

    If a detector fires at least once inside a labeled anomaly interval, the
    whole interval is counted as detected. Predictions outside labeled anomaly
    intervals remain unchanged and still contribute false positives.
    """
    labels = np.asarray(labels).astype(int)
    adjusted = np.asarray(predictions).astype(int).copy()
    for start, end in _segments(labels):
        if adjusted[start:end].any():
            adjusted[start:end] = 1
    return adjusted


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    tp = float(np.sum((labels == 1) & (predictions == 1)))
    fp = float(np.sum((labels == 0) & (predictions == 1)))
    fn = float(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1, precision, recall


def best_point_adjusted_f1(labels: np.ndarray, scores: np.ndarray, max_thresholds: int = 200) -> dict[str, float | None]:
    """Return best point-adjusted F1 over score thresholds.

    The threshold is selected with labeled evaluation data, so this metric is
    suitable as a diagnostic appendix measure rather than a deployable operating
    point unless a validation set is available.
    """
    labels = np.asarray(labels).astype(int)
    scores = normalize_scores(scores)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"pa_f1": None, "pa_precision": None, "pa_recall": None, "pa_threshold": None}

    unique_scores = np.unique(scores)
    if len(unique_scores) > max_thresholds:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, max_thresholds))
        thresholds = np.unique(thresholds)
    else:
        thresholds = unique_scores

    best = {"pa_f1": 0.0, "pa_precision": 0.0, "pa_recall": 0.0, "pa_threshold": float(thresholds[0])}
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        adjusted = point_adjust_predictions(labels, predictions)
        f1, precision, recall = _binary_f1(labels, adjusted)
        if f1 > float(best["pa_f1"]):
            best = {
                "pa_f1": float(f1),
                "pa_precision": float(precision),
                "pa_recall": float(recall),
                "pa_threshold": float(threshold),
            }
    return best


def _top_budget_predictions(scores: np.ndarray, fraction: float) -> np.ndarray:
    scores = normalize_scores(scores)
    k = max(1, int(np.ceil(len(scores) * fraction)))
    idx = np.argpartition(scores, -k)[-k:]
    predictions = np.zeros(len(scores), dtype=int)
    predictions[idx] = 1
    return predictions


def _event_recall(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    segments = _segments(labels)
    if not segments:
        return None
    detected = sum(1 for start, end in segments if predictions[start:end].any())
    return float(detected / len(segments))


def _event_precision(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    pred_segments = _segments(predictions)
    if not pred_segments:
        return 0.0
    labels = np.asarray(labels).astype(int)
    overlapping = sum(1 for start, end in pred_segments if labels[start:end].any())
    return float(overlapping / len(pred_segments))


def practical_range_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    budgets: tuple[float, ...] = (0.01, 0.03, 0.05),
) -> dict[str, float | None]:
    """Threshold-free and fixed-budget metrics for practical range detection."""
    labels = np.asarray(labels).astype(int)
    scores = normalize_scores(scores)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {
            "pauc_fpr_05": None,
            **{f"event_recall_top_{int(b * 100)}": None for b in budgets},
            **{f"event_precision_top_{int(b * 100)}": None for b in budgets},
        }

    out: dict[str, float | None] = {}
    try:
        out["pauc_fpr_05"] = float(roc_auc_score(labels, scores, max_fpr=0.05))
    except ValueError:
        out["pauc_fpr_05"] = None

    for budget in budgets:
        predictions = _top_budget_predictions(scores, budget)
        suffix = int(budget * 100)
        out[f"event_recall_top_{suffix}"] = _event_recall(labels, predictions)
        out[f"event_precision_top_{suffix}"] = _event_precision(labels, predictions)
    return out


def _dilate_labels(labels: np.ndarray, radius: int) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    if radius <= 0:
        return labels
    out = labels.copy()
    positives = np.flatnonzero(labels == 1)
    for idx in positives:
        lo = max(0, idx - radius)
        hi = min(len(labels), idx + radius + 1)
        out[lo:hi] = 1
    return out


def _fallback_vus(labels: np.ndarray, scores: np.ndarray, sliding_window: int) -> dict[str, float | str]:
    radii = np.unique(np.linspace(0, sliding_window, num=min(sliding_window + 1, 20), dtype=int))
    roc_values = []
    pr_values = []
    for radius in radii:
        buffered = _dilate_labels(labels, int(radius))
        if buffered.sum() == 0 or buffered.sum() == len(buffered):
            continue
        roc_values.append(roc_auc_score(buffered, scores))
        pr_values.append(average_precision_score(buffered, scores))
    return {
        "vus_roc": float(np.mean(roc_values)) if roc_values else None,
        "vus_pr": float(np.mean(pr_values)) if pr_values else None,
        "vus_backend": "buffered_fallback",
    }


def range_vus_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    sliding_window: int | None = None,
    prefer_official: bool = True,
) -> dict[str, float | int | str | None]:
    labels = np.asarray(labels).astype(int)
    scores = normalize_scores(scores)
    window = int(sliding_window or default_sliding_window(labels))
    out: dict[str, float | int | str | None] = {"vus_window": window}

    if labels.sum() == 0 or labels.sum() == len(labels):
        out.update({"vus_roc": None, "vus_pr": None, "vus_backend": "undefined_labels"})
        return out

    if prefer_official:
        try:
            from vus.metrics import get_metrics

            metrics = get_metrics(scores, labels, metric="vus", slidingWindow=window)
            out.update(
                {
                    "vus_roc": float(metrics.get("VUS_ROC")),
                    "vus_pr": float(metrics.get("VUS_PR")),
                    "vus_backend": "official_vus",
                }
            )
            return out
        except Exception as exc:
            out["vus_error"] = str(exc)

    out.update(_fallback_vus(labels, scores, window))
    return out


def entity_macro_vus(
    labels: np.ndarray,
    scores: np.ndarray,
    entity_ids: np.ndarray | None = None,
    sliding_window: int | None = None,
    prefer_official: bool = True,
) -> dict[str, float | int | str | None]:
    if entity_ids is None:
        return range_vus_scores(labels, scores, sliding_window, prefer_official)

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    entity_ids = np.asarray(entity_ids)
    rows = []
    for entity_id in np.unique(entity_ids):
        idx = np.flatnonzero(entity_ids == entity_id)
        yi = labels[idx]
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue
        rows.append(range_vus_scores(yi, scores[idx], sliding_window, prefer_official))
    if not rows:
        return {"vus_roc": None, "vus_pr": None, "vus_window": sliding_window, "vus_backend": "no_valid_entity"}
    return {
        "vus_roc": float(np.mean([r["vus_roc"] for r in rows if r["vus_roc"] is not None])),
        "vus_pr": float(np.mean([r["vus_pr"] for r in rows if r["vus_pr"] is not None])),
        "vus_window": int(round(float(np.median([r["vus_window"] for r in rows if r["vus_window"] is not None])))),
        "vus_backend": rows[0].get("vus_backend", "mixed"),
    }
