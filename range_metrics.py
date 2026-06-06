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
