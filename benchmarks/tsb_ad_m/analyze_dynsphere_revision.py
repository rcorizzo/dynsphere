#!/usr/bin/env python3
"""Aggregate paired DynSphere revision experiments across files and seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


METRICS = ("auroc", "auprc", "pauc_fpr_05", "event_recall_top_1")


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(values: list[float]) -> list[float]:
    adjusted = np.ones(len(values), dtype=float)
    order = np.argsort(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_rows(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_name: str,
    family: str,
    rng: np.random.Generator,
    repetitions: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reference_mean = reference.groupby(["source", "file"], as_index=False)[list(METRICS)].mean()
    candidate_mean = candidate.groupby(["source", "file"], as_index=False)[list(METRICS)].mean()
    paired = reference_mean.merge(
        candidate_mean,
        on=["source", "file"],
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    for source, source_rows in paired.groupby("source", sort=True):
        for metric in METRICS:
            reference_values = source_rows[f"{metric}_reference"].to_numpy(dtype=float)
            candidate_values = source_rows[f"{metric}_candidate"].to_numpy(dtype=float)
            differences = candidate_values - reference_values
            draws = rng.choice(differences, size=(repetitions, len(differences)), replace=True)
            low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            p_value = (
                1.0
                if np.allclose(differences, 0.0)
                else float(wilcoxon(differences, zero_method="pratt", method="auto").pvalue)
            )
            rows.append(
                {
                    "family": family,
                    "candidate": candidate_name,
                    "source": source,
                    "metric": metric,
                    "n_files": len(differences),
                    "reference_mean": float(reference_values.mean()),
                    "candidate_mean": float(candidate_values.mean()),
                    "mean_delta": float(differences.mean()),
                    "bootstrap_ci_low": float(low),
                    "bootstrap_ci_high": float(high),
                    "wins": int((differences > 1e-12).sum()),
                    "ties": int((np.abs(differences) <= 1e-12).sum()),
                    "losses": int((differences < -1e-12).sum()),
                    "rank_biserial": rank_biserial(differences),
                    "wilcoxon_p": p_value,
                }
            )
    return rows


def sensitivity_rank_rows(reference: pd.DataFrame, sensitivity: pd.DataFrame) -> list[dict[str, object]]:
    reference_mean = reference.groupby(["source", "file"], as_index=False)[list(METRICS)].mean()
    rows: list[dict[str, object]] = []
    for (family, setting), subset in sensitivity[
        sensitivity["method_id"] != "default"
    ].groupby(["scenario", "method_id"], sort=True):
        candidate_mean = subset.groupby(["source", "file"], as_index=False)[list(METRICS)].mean()
        paired = reference_mean.merge(
            candidate_mean,
            on=["source", "file"],
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        for source, source_rows in paired.groupby("source", sort=True):
            for metric in METRICS:
                reference_values = source_rows[f"{metric}_reference"].to_numpy(dtype=float)
                candidate_values = source_rows[f"{metric}_candidate"].to_numpy(dtype=float)
                correlation = spearmanr(reference_values, candidate_values).statistic
                rows.append(
                    {
                        "family": family,
                        "setting": setting,
                        "source": source,
                        "metric": metric,
                        "n_files": len(source_rows),
                        "reference_mean": float(reference_values.mean()),
                        "candidate_mean": float(candidate_values.mean()),
                        "mean_absolute_file_delta": float(np.mean(np.abs(candidate_values - reference_values))),
                        "file_performance_spearman": float(correlation) if np.isfinite(correlation) else None,
                    }
                )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["candidate_mean_rank_within_family"] = frame.groupby(
            ["family", "source", "metric"]
        )["candidate_mean"].rank(method="min", ascending=False)
    return frame.to_dict(orient="records")


def write_robustness_plot(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = frame[frame["metric"].isin(["auroc", "auprc"])].copy()
    if selected.empty:
        return
    order = ["noise_0.01", "noise_0.05", "noise_0.10", "decimation_2", "decimation_4"]
    labels = ["noise 1%", "noise 5%", "noise 10%", "sample /2", "sample /4"]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharex=True)
    for axis, metric in zip(axes, ("auroc", "auprc")):
        metric_rows = selected[selected["metric"] == metric]
        for source, source_rows in metric_rows.groupby("source"):
            lookup = source_rows.set_index("candidate")
            relative = [
                100.0 * float(lookup.loc[item, "mean_delta"]) / max(
                    abs(float(lookup.loc[item, "reference_mean"])), 1e-12
                )
                for item in order
            ]
            axis.plot(labels, relative, marker="o", linewidth=1.5, label=source)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(metric.upper())
        axis.set_ylabel("relative metric change (%)")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def adjust_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["wilcoxon_p_holm"] = holm_adjust(frame["wilcoxon_p"].tolist())
    return frame


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--component-metrics", type=Path, default=None)
    parser.add_argument("--component-correlations", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=37)
    args = parser.parse_args()

    data = pd.read_csv(args.results)
    data = data[data["status"] == "ok"].copy()
    rng = np.random.default_rng(args.random_state)
    outputs: dict[str, pd.DataFrame] = {}

    ablation = data[data["experiment"] == "ablation"]
    if not ablation.empty:
        reference = ablation[ablation["method_id"] == "dynsphere"]
        rows: list[dict[str, object]] = []
        for method in sorted(set(ablation["method_id"]) - {"dynsphere"}):
            rows.extend(
                paired_rows(
                    reference,
                    ablation[ablation["method_id"] == method],
                    method,
                    "ablation",
                    rng,
                    args.bootstrap_repetitions,
                )
            )
        outputs["paired_ablation.csv"] = adjust_table(pd.DataFrame(rows))

    sensitivity = data[data["experiment"] == "sensitivity"]
    if not sensitivity.empty:
        reference = sensitivity[
            (sensitivity["scenario"] == "default")
            & (sensitivity["method_id"] == "default")
        ]
        rows = []
        for (family, setting), subset in sensitivity[
            sensitivity["method_id"] != "default"
        ].groupby(["scenario", "method_id"], sort=True):
            rows.extend(
                paired_rows(
                    reference,
                    subset,
                    setting,
                    family,
                    rng,
                    args.bootstrap_repetitions,
                )
            )
        outputs["paired_sensitivity.csv"] = adjust_table(pd.DataFrame(rows))
        outputs["sensitivity_rank_stability.csv"] = pd.DataFrame(
            sensitivity_rank_rows(reference, sensitivity)
        )

    robustness = data[data["experiment"] == "robustness"]
    if not robustness.empty:
        reference = robustness[robustness["scenario"] == "baseline"]
        rows = []
        for scenario, subset in robustness[
            robustness["scenario"] != "baseline"
        ].groupby("scenario", sort=True):
            rows.extend(
                paired_rows(
                    reference,
                    subset,
                    scenario,
                    "robustness",
                    rng,
                    args.bootstrap_repetitions,
                )
            )
        outputs["paired_robustness.csv"] = adjust_table(pd.DataFrame(rows))

    data["fit_rows_per_second"] = data["train_rows"] / data["fit_seconds"].clip(lower=1e-12)
    data["score_rows_per_second"] = data["n"] / data["score_seconds"].clip(lower=1e-12)
    runtime = data.groupby(
        ["experiment", "scenario", "method_id"]
    )[
        [
            "fit_seconds",
            "score_seconds",
            "total_seconds",
            "fit_rows_per_second",
            "score_rows_per_second",
        ]
    ].agg(["count", "mean", "std"]).reset_index()
    outputs["runtime_summary.csv"] = flatten_columns(runtime)

    if args.component_metrics and args.component_metrics.exists():
        component = pd.read_csv(args.component_metrics)
        component_summary = component.groupby(
            ["source", "component"]
        )[
            [
                "auroc",
                "auprc",
                "spearman_with_final",
                "normal_mean",
                "anomaly_mean",
                "pauc_fpr_05",
            ]
        ].agg(["count", "mean", "std"]).reset_index()
        outputs["component_summary.csv"] = flatten_columns(component_summary)

    if args.component_correlations and args.component_correlations.exists():
        correlations = pd.read_csv(args.component_correlations)
        correlation_summary = correlations.groupby(
            ["source", "component_left", "component_right"]
        )["spearman"].agg(["count", "mean", "std"]).reset_index()
        outputs["correlation_summary.csv"] = flatten_columns(correlation_summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "results": str(args.results),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "random_state": args.random_state,
        "outputs": {},
    }
    for name, frame in outputs.items():
        path = args.out_dir / name
        frame.to_csv(path, index=False)
        manifest["outputs"][name] = {"path": str(path), "rows": len(frame)}
    if "paired_robustness.csv" in outputs:
        plot_path = args.out_dir / "robustness_relative_degradation.pdf"
        write_robustness_plot(outputs["paired_robustness.csv"], plot_path)
        manifest["outputs"][plot_path.name] = {"path": str(plot_path), "rows": None}
    (args.out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote paired analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
