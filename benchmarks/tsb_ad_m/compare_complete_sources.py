#!/usr/bin/env python3
"""Paired complete-source comparison for TSB-AD-M result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


SOURCES = ("MSL", "SMAP", "SMD")


def source_from_dataset(dataset: str) -> str:
    for source in SOURCES:
        if f"_{source}_" in dataset:
            return source
    return "Other"


def load_results(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    data = data[data["status"] == "ok"].copy()
    data["file_name"] = data["file"].map(lambda value: Path(str(value)).name)
    data["source"] = data["dataset"].astype(str).map(source_from_dataset)
    # Later inputs intentionally replace a failed declared-prefix execution
    # with the explicitly documented maximal-clean-prefix execution.
    return data.drop_duplicates(subset=["file_name", "method_id"], keep="last")


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def bootstrap_mean_interval(
    differences: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> tuple[float, float]:
    draws = rng.choice(differences, size=(repetitions, len(differences)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = np.ones(len(p_values), dtype=float)
    order = np.argsort(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--reference-methods", nargs="+", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["auroc", "auprc", "pauc_fpr_05", "event_recall_top_1"],
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=37)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate = load_results(args.candidate_inputs)
    reference = load_results(args.reference_inputs)
    candidate = candidate[candidate["method_id"] == args.candidate_method]
    rng = np.random.default_rng(args.random_state)
    rows: list[dict[str, object]] = []

    for source in SOURCES:
        left = candidate[candidate["source"] == source]
        for reference_method in args.reference_methods:
            right = reference[
                (reference["source"] == source)
                & (reference["method_id"] == reference_method)
            ]
            paired = left.merge(
                right,
                on="file_name",
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            for metric in args.metrics:
                candidate_column = f"{metric}_candidate"
                reference_column = f"{metric}_reference"
                if candidate_column not in paired or reference_column not in paired:
                    continue
                values = paired[[candidate_column, reference_column]].dropna()
                candidate_values = values[candidate_column].to_numpy(dtype=float)
                reference_values = values[reference_column].to_numpy(dtype=float)
                differences = candidate_values - reference_values
                if len(differences) == 0:
                    continue
                low, high = bootstrap_mean_interval(
                    differences,
                    rng,
                    args.bootstrap_repetitions,
                )
                if np.allclose(differences, 0.0):
                    p_value = 1.0
                else:
                    p_value = float(
                        wilcoxon(
                            differences,
                            zero_method="pratt",
                            alternative="two-sided",
                            method="auto",
                        ).pvalue
                    )
                rows.append(
                    {
                        "source": source,
                        "candidate_method": args.candidate_method,
                        "reference_method": reference_method,
                        "metric": metric,
                        "n_files": int(len(differences)),
                        "candidate_mean": float(candidate_values.mean()),
                        "reference_mean": float(reference_values.mean()),
                        "mean_delta": float(differences.mean()),
                        "median_delta": float(np.median(differences)),
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "wins": int((differences > 1e-12).sum()),
                        "ties": int((np.abs(differences) <= 1e-12).sum()),
                        "losses": int((differences < -1e-12).sum()),
                        "rank_biserial": rank_biserial(differences),
                        "wilcoxon_p": p_value,
                    }
                )

    summary = pd.DataFrame(rows)
    if len(summary):
        summary["wilcoxon_p_holm"] = holm_adjust(summary["wilcoxon_p"].tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    payload = {
        "candidate_inputs": [str(path) for path in args.candidate_inputs],
        "reference_inputs": [str(path) for path in args.reference_inputs],
        "candidate_method": args.candidate_method,
        "reference_methods": args.reference_methods,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "random_state": args.random_state,
        "comparisons": summary.where(pd.notnull(summary), None).to_dict(orient="records"),
    }
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
