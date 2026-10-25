#!/usr/bin/env python3
"""Paired seed and moving-block analysis for single-source DynSphere ablations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ABLATIONS = (
    "dynsphere_no_transition",
    "dynsphere_no_energy",
    "dynsphere_no_raw",
    "dynsphere_no_concepts",
)


def event_lengths(labels: np.ndarray) -> list[int]:
    padded = np.pad(np.asarray(labels, dtype=np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    return (stops - starts).astype(int).tolist()


def circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    count = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=count)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]


def metric_value(metric: str, labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.min() == labels.max():
        return np.nan
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    return float(average_precision_score(labels, scores))


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(values) - rank) * values[index])
        running = max(running, current)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=20260920)
    args = parser.parse_args()

    grouped: dict[str, list[tuple[int, dict[str, np.ndarray]]]] = {}
    for path in sorted(args.input_dir.glob("*_seed*_scores.npz")):
        match = re.match(r"(.+)_seed(\d+)_scores\.npz$", path.name)
        if match is None:
            continue
        with np.load(path) as payload:
            grouped.setdefault(match.group(1), []).append(
                (int(match.group(2)), {key: payload[key] for key in payload.files})
            )
    if not grouped:
        raise SystemExit(f"No *_seed*_scores.npz files found in {args.input_dir}")

    rng = np.random.default_rng(args.random_state)
    rows: list[dict[str, object]] = []
    for dataset, runs in sorted(grouped.items()):
        runs.sort(key=lambda item: item[0])
        labels = np.asarray(runs[0][1]["labels"], dtype=int)
        lengths = event_lengths(labels)
        block_length = max(1, int(round(np.median(lengths)))) if lengths else max(1, int(np.sqrt(len(labels))))
        for metric in ("auroc", "auprc"):
            dataset_rows = []
            for ablation in ABLATIONS:
                observed = []
                for _, payload in runs:
                    observed.append(
                        metric_value(metric, labels, payload["dynsphere"])
                        - metric_value(metric, labels, payload[ablation])
                    )
                boot = []
                for _ in range(args.bootstrap):
                    idx = circular_block_indices(len(labels), block_length, rng)
                    sampled_labels = labels[idx]
                    if sampled_labels.min() == sampled_labels.max():
                        continue
                    per_seed = []
                    for _, payload in runs:
                        per_seed.append(
                            metric_value(metric, sampled_labels, payload["dynsphere"][idx])
                            - metric_value(metric, sampled_labels, payload[ablation][idx])
                        )
                    boot.append(float(np.nanmean(per_seed)))
                boot_array = np.asarray(boot, dtype=float)
                negative = (np.count_nonzero(boot_array <= 0) + 1) / (len(boot_array) + 1)
                positive = (np.count_nonzero(boot_array >= 0) + 1) / (len(boot_array) + 1)
                dataset_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "ablation": ablation,
                        "interpretation": "full_minus_ablation",
                        "seeds": len(runs),
                        "block_length": block_length,
                        "observed_difference": float(np.mean(observed)),
                        "seed_std": float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0,
                        "fraction_seeds_full_better": float(np.mean(np.asarray(observed) > 0)),
                        "bootstrap_ci_low": float(np.quantile(boot_array, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(boot_array, 0.975)),
                        "bootstrap_p": float(min(1.0, 2 * min(negative, positive))),
                        "bootstrap_replicates": len(boot_array),
                    }
                )
            adjusted = holm_adjust([float(row["bootstrap_p"]) for row in dataset_rows])
            for row, value in zip(dataset_rows, adjusted):
                row["holm_p"] = value
                rows.append(row)

    output = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(output.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
