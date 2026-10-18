"""Hierarchical summaries for official baseline experiment records."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from benchmarks.official_baselines.contract import validate_scores
from benchmarks.official_baselines.metrics import compute_metrics


METRIC_COLUMNS = (
    "auprc",
    "auroc",
    "pauc_fpr_05",
    "vus_pr",
    "vus_roc",
    "event_recall",
    "event_precision",
)
PRIMARY_METRICS = ("auprc", "vus_pr")
PAIR_KEYS = ("dataset_id", "trajectory_id", "seed")
CAMPAIGN_KEY_FIELDS = (
    "case_id",
    "dataset_id",
    "source_family",
    "trajectory_id",
    "seed",
    "method_id",
    "adapter_revision",
)
PAPER_METHODS = (
    "pca_reconstruction",
    "kmeans_distance",
    "dagmm",
    "catch",
    "mtad_gat",
    "omni_anomaly_tsb_ad",
    "anomaly_transformer",
    "tranad",
    "predictive_euclidean_single",
    "dynsphere",
    "dynsphere_euclidean",
    "dynsphere_one_regime",
    "dynsphere_euclidean_one_regime",
)
FACTORIAL_CONTRASTS = (
    ("E4-S4", "dynsphere_euclidean", "dynsphere"),
    ("E1-S1", "dynsphere_euclidean_one_regime", "dynsphere_one_regime"),
    ("S4-S1", "dynsphere", "dynsphere_one_regime"),
    ("E4-E1", "dynsphere_euclidean", "dynsphere_euclidean_one_regime"),
)


@dataclass(frozen=True)
class AggregateArtifacts:
    per_run_csv: Path
    trajectory_csv: Path
    family_csv: Path
    family_macro_csv: Path
    paired_csv: Path
    geometry_factorial_csv: Path
    completion_csv: Path
    readiness_json: Path
    manuscript_table_tex: Path | None
    manuscript_ready: bool


def _read_records(
    result_root: Path,
    expected_adapter_revision: str | None = None,
    campaign_manifest: Path | None = None,
) -> tuple[
    list[tuple[Path, dict[str, object]]],
    int,
    int,
    str | None,
    set[tuple[object, ...]] | None,
]:
    records: list[tuple[Path, dict[str, object]]] = []
    excluded_stale = 0
    excluded_noncampaign = 0
    campaign_id: str | None = None
    expected_jobs: set[tuple[object, ...]] | None = None
    if campaign_manifest is not None:
        try:
            campaign = json.loads(campaign_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Campaign manifest is unreadable: {campaign_manifest}") from exc
        if campaign.get("schema_version") != "official-baseline-campaign-v1":
            raise ValueError("Campaign manifest has an unsupported schema")
        raw_jobs = campaign.get("expected_jobs")
        if not isinstance(raw_jobs, list):
            raise ValueError("Campaign manifest expected_jobs must be a list")
        if campaign.get("expected_job_count") != len(raw_jobs):
            raise ValueError("Campaign manifest expected_job_count is inconsistent")
        expected_jobs = set()
        for row in raw_jobs:
            if not isinstance(row, dict) or any(field not in row for field in CAMPAIGN_KEY_FIELDS):
                raise ValueError("Campaign manifest contains an incomplete job entry")
            expected_jobs.add(tuple(row[field] for field in CAMPAIGN_KEY_FIELDS))
        campaign_id = str(campaign.get("campaign_id", ""))
        if not campaign_id:
            raise ValueError("Campaign manifest must identify campaign_id")
    for path in sorted((result_root / "runs").rglob("run.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or "run_identity" not in record or "method_id" not in record:
            continue
        if (
            expected_adapter_revision is not None
            and record.get("adapter_revision") != expected_adapter_revision
        ):
            excluded_stale += 1
            continue
        if expected_jobs is not None:
            key = tuple(record.get(field) for field in CAMPAIGN_KEY_FIELDS)
            if key not in expected_jobs:
                excluded_noncampaign += 1
                continue
        records.append((path.parent, record))
    return records, excluded_stale, excluded_noncampaign, campaign_id, expected_jobs


def _exclude_superseded_records(
    records: list[tuple[Path, dict[str, object]]],
) -> tuple[list[tuple[Path, dict[str, object]]], int]:
    """Retain the newest terminal record for each campaign job key."""
    selected: dict[tuple[object, ...], tuple[Path, dict[str, object]]] = {}
    for run_dir, record in records:
        key = tuple(record.get(field) for field in CAMPAIGN_KEY_FIELDS)
        current = selected.get(key)
        if current is None:
            selected[key] = (run_dir, record)
            continue
        current_path, current_record = current
        candidate_order = (
            run_dir.joinpath("run.json").stat().st_mtime_ns,
            str(record.get("run_identity", "")),
        )
        current_order = (
            current_path.joinpath("run.json").stat().st_mtime_ns,
            str(current_record.get("run_identity", "")),
        )
        if candidate_order > current_order:
            selected[key] = (run_dir, record)
    ordered = sorted(
        selected.values(),
        key=lambda item: tuple(str(item[1].get(field, "")) for field in CAMPAIGN_KEY_FIELDS),
    )
    return ordered, len(records) - len(ordered)


def _metric_row(
    run_dir: Path,
    record: Mapping[str, object],
    event_budget: float = 0.01,
) -> dict[str, object]:
    labels = np.asarray(
        np.load(run_dir / "case" / "private" / "labels.npy", allow_pickle=False)
    )
    scores = validate_scores(run_dir / "execution" / "scores.npy", len(labels))
    values = compute_metrics(labels, scores, event_budget=event_budget, vus_window=None)
    return {
        "run_identity": record["run_identity"],
        "case_id": record["case_id"],
        "dataset_id": record["dataset_id"],
        "source_family": record["source_family"],
        "trajectory_id": record["trajectory_id"],
        "seed": int(record["seed"]),
        "method_id": record["method_id"],
        "executed_method_id": record.get("executed_method_id", record["method_id"]),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "peak_memory_mb": record.get("peak_memory_mb"),
        **values,
    }


def _mean_table(
    frame: pd.DataFrame,
    group_columns: list[str],
    count_name: str,
) -> pd.DataFrame:
    columns = group_columns + list(METRIC_COLUMNS) + [count_name]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    grouped = frame.groupby(group_columns, as_index=False, sort=True)
    means = grouped[list(METRIC_COLUMNS)].mean(numeric_only=True)
    counts = grouped.size().rename(columns={"size": count_name})
    return means.merge(counts, on=group_columns, validate="one_to_one")


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    denominator = float(ranks.sum())
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / denominator


def _bootstrap_interval(
    differences: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if len(differences) == 1:
        value = float(differences[0])
        return value, value
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _paired_table(
    per_run: pd.DataFrame,
    records: list[tuple[Path, dict[str, object]]],
    bootstrap_seed: int,
    bootstrap_replicates: int,
    method_pairs: Sequence[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    columns = [
        "method_a",
        "method_b",
        "metric",
        "paired_count",
        "incomplete_pair_count",
        "mean_difference",
        "ci_low",
        "ci_high",
        "win_rate",
        "rank_biserial",
    ]
    methods = sorted({str(record["method_id"]) for _, record in records})
    expected: dict[str, set[tuple[object, ...]]] = {method: set() for method in methods}
    for _, record in records:
        key = tuple(record.get(name) for name in PAIR_KEYS)
        expected[str(record["method_id"])].add(key)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(bootstrap_seed)
    pairs = tuple(method_pairs) if method_pairs is not None else tuple(combinations(methods, 2))
    for method_a, method_b in pairs:
        left = per_run[per_run.method_id == method_a]
        right = per_run[per_run.method_id == method_b]
        for metric in PRIMARY_METRICS:
            left_values = left[list(PAIR_KEYS) + [metric]].rename(columns={metric: "left"})
            right_values = right[list(PAIR_KEYS) + [metric]].rename(columns={metric: "right"})
            paired = left_values.merge(right_values, on=list(PAIR_KEYS), how="inner")
            paired = paired.dropna(subset=["left", "right"])
            differences = paired.left.to_numpy(dtype=float) - paired.right.to_numpy(dtype=float)
            paired_keys = {
                tuple(row[name] for name in PAIR_KEYS)
                for row in paired.to_dict(orient="records")
            }
            expected_union = expected.get(method_a, set()).union(
                expected.get(method_b, set())
            )
            incomplete = len(expected_union.difference(paired_keys))
            if len(differences) == 0:
                rows.append(
                    {
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "paired_count": 0,
                        "incomplete_pair_count": incomplete,
                        "mean_difference": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "win_rate": np.nan,
                        "rank_biserial": np.nan,
                    }
                )
                continue
            low, high = _bootstrap_interval(differences, rng, bootstrap_replicates)
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "paired_count": len(differences),
                    "incomplete_pair_count": incomplete,
                    "mean_difference": float(differences.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "win_rate": float(np.mean(differences > 0)),
                    "rank_biserial": _rank_biserial(differences),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _factorial_table(
    per_run: pd.DataFrame,
    records: list[tuple[Path, dict[str, object]]],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    pairs = tuple((method_a, method_b) for _, method_a, method_b in FACTORIAL_CONTRASTS)
    table = _paired_table(
        per_run,
        records,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        method_pairs=pairs,
    )
    labels = {
        (method_a, method_b): label
        for label, method_a, method_b in FACTORIAL_CONTRASTS
    }
    table.insert(
        0,
        "contrast",
        [labels[(row.method_a, row.method_b)] for row in table.itertuples()],
    )
    return table


def _identity_document(key: tuple[object, ...]) -> dict[str, object]:
    return dict(zip(CAMPAIGN_KEY_FIELDS, key))


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    metric_columns = [column for column in ("auprc", "vus_pr", "auroc", "vus_roc") if column in frame]
    lines = [
        "\\begin{tabular}{l" + "r" * len(metric_columns) + "}",
        "\\toprule",
        "Method & " + " & ".join(column.replace("_", "\\_") for column in metric_columns) + " \\\\",
        "\\midrule",
    ]
    for row in frame.to_dict(orient="records"):
        values = " & ".join(f"{float(row[column]):.4f}" for column in metric_columns)
        method = str(row["method_id"]).replace("_", "\\_")
        lines.append(f"{method} & {values} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_runs(
    result_root: Path,
    output_dir: Path,
    bootstrap_seed: int = 20261011,
    bootstrap_replicates: int = 10000,
    expected_adapter_revision: str | None = None,
    campaign_manifest: Path | None = None,
) -> AggregateArtifacts:
    """Compute run metrics and family-balanced summaries from terminal records."""
    (
        records,
        excluded_stale,
        excluded_noncampaign,
        campaign_id,
        expected_jobs,
    ) = _read_records(
        result_root,
        expected_adapter_revision,
        campaign_manifest,
    )
    records, excluded_superseded = _exclude_superseded_records(records)
    if not records:
        raise ValueError(f"No experiment records found under {result_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    completion = Counter(str(record.get("status", "unknown")) for _, record in records)
    completion_frame = pd.DataFrame(
        [{"status": status, "count": count} for status, count in sorted(completion.items())]
    )
    metric_rows: list[dict[str, object]] = []
    metric_errors: list[dict[str, str]] = []
    for run_dir, record in records:
        if record.get("status") != "complete":
            continue
        try:
            metric_rows.append(_metric_row(run_dir, record))
        except Exception as exc:
            metric_errors.append(
                {
                    "run_identity": str(record.get("run_identity", "unknown")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    per_run = pd.DataFrame(metric_rows)
    if per_run.empty:
        per_run = pd.DataFrame(
            columns=[
                "run_identity",
                "case_id",
                "dataset_id",
                "source_family",
                "trajectory_id",
                "seed",
                "method_id",
                *METRIC_COLUMNS,
                "vus_backend",
            ]
        )
    trajectory = _mean_table(
        per_run,
        ["method_id", "dataset_id", "source_family", "trajectory_id"],
        "seed_count",
    )
    family = _mean_table(
        trajectory,
        ["method_id", "dataset_id", "source_family"],
        "trajectory_count",
    )
    family_macro = _mean_table(family, ["method_id"], "family_count")
    paired = _paired_table(
        per_run,
        records,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    factorial = _factorial_table(
        per_run,
        records,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )

    per_run_csv = output_dir / "per_run.csv"
    trajectory_csv = output_dir / "per_trajectory.csv"
    family_csv = output_dir / "per_family.csv"
    family_macro_csv = output_dir / "family_macro.csv"
    paired_csv = output_dir / "paired.csv"
    geometry_factorial_csv = output_dir / "geometry_factorial.csv"
    completion_csv = output_dir / "completion.csv"
    readiness_json = output_dir / "readiness.json"
    per_run.to_csv(per_run_csv, index=False)
    trajectory.to_csv(trajectory_csv, index=False)
    family.to_csv(family_csv, index=False)
    family_macro.to_csv(family_macro_csv, index=False)
    paired.to_csv(paired_csv, index=False)
    factorial.to_csv(geometry_factorial_csv, index=False)
    completion_frame.to_csv(completion_csv, index=False)

    observed_methods = sorted({str(record["method_id"]) for _, record in records})
    expected_methods = sorted(PAPER_METHODS)
    missing_methods = sorted(set(expected_methods).difference(observed_methods))
    unexpected_methods = sorted(set(observed_methods).difference(expected_methods))
    valid_methods = set(per_run.method_id.astype(str))
    methods_without_valid = sorted(set(observed_methods).difference(valid_methods))
    metric_warnings = []
    if "vus_backend" in per_run:
        metric_warnings = [
            {
                "run_identity": str(row.run_identity),
                "warning": f"vus_backend={row.vus_backend}",
            }
            for row in per_run.loc[
                per_run.vus_backend != "official_vus",
                ["run_identity", "vus_backend"],
            ].itertuples(index=False)
        ]
    fallback_vus_runs = [warning["run_identity"] for warning in metric_warnings]
    observed_job_keys = {
        tuple(record.get(field) for field in CAMPAIGN_KEY_FIELDS)
        for _, record in records
    }
    missing_run_keys = (
        sorted(expected_jobs.difference(observed_job_keys), key=lambda row: tuple(map(str, row)))
        if expected_jobs is not None
        else []
    )
    missing_run_identities = [_identity_document(key) for key in missing_run_keys]
    noncomplete_identities = [
        {
            **_identity_document(
                tuple(record.get(field) for field in CAMPAIGN_KEY_FIELDS)
            ),
            "run_identity": str(record.get("run_identity", "unknown")),
            "status": str(record.get("status", "unknown")),
        }
        for _, record in records
        if record.get("status") != "complete"
    ]
    revisions: defaultdict[str, set[str]] = defaultdict(set)
    revision_sources = expected_jobs if expected_jobs is not None else observed_job_keys
    for key in revision_sources:
        method_id = str(key[CAMPAIGN_KEY_FIELDS.index("method_id")])
        revision = str(key[CAMPAIGN_KEY_FIELDS.index("adapter_revision")])
        revisions[method_id].add(revision)
    adapter_revisions_by_method = {
        method: sorted(values) for method, values in sorted(revisions.items())
    }
    methods_with_multiple_adapter_revisions = sorted(
        method for method, values in revisions.items() if len(values) > 1
    )
    campaign_adapter_revisions = sorted(
        {revision for values in revisions.values() for revision in values}
    )
    mixed_campaign_adapter_revisions = len(campaign_adapter_revisions) > 1
    noncomplete = int(sum(count for status, count in completion.items() if status != "complete"))
    manuscript_ready = bool(
        not missing_methods
        and not missing_run_identities
        and not methods_without_valid
        and not fallback_vus_runs
        and not metric_errors
        and not methods_with_multiple_adapter_revisions
        and not mixed_campaign_adapter_revisions
        and noncomplete == 0
    )
    readiness = {
        "manuscript_ready": manuscript_ready,
        "expected_methods": expected_methods,
        "observed_methods": observed_methods,
        "missing_methods": missing_methods,
        "unexpected_methods": unexpected_methods,
        "methods_without_valid_runs": methods_without_valid,
        "missing_run_identities": missing_run_identities,
        "noncomplete_identities": noncomplete_identities,
        "noncomplete_run_count": noncomplete,
        "fallback_vus_run_count": len(fallback_vus_runs),
        "fallback_vus_runs": fallback_vus_runs,
        "metric_warnings": metric_warnings,
        "metric_errors": metric_errors,
        "adapter_revisions_by_method": adapter_revisions_by_method,
        "methods_with_multiple_adapter_revisions": methods_with_multiple_adapter_revisions,
        "campaign_adapter_revisions": campaign_adapter_revisions,
        "mixed_campaign_adapter_revisions": mixed_campaign_adapter_revisions,
        "geometry_factorial_csv": str(geometry_factorial_csv),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "expected_adapter_revision": expected_adapter_revision,
        "excluded_stale_record_count": excluded_stale,
        "campaign_id": campaign_id,
        "campaign_manifest": str(campaign_manifest) if campaign_manifest is not None else None,
        "excluded_noncampaign_record_count": excluded_noncampaign,
        "excluded_superseded_record_count": excluded_superseded,
        "superseded_record_selection": "newest-run-record-mtime-with-identity-tiebreak",
        "weighting": "trajectory mean within source family, then unweighted source-family mean",
    }
    readiness_json.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manuscript_table: Path | None = None
    if manuscript_ready:
        manuscript_table = output_dir / "manuscript_family_macro.tex"
        _write_table(manuscript_table, family_macro)
    return AggregateArtifacts(
        per_run_csv=per_run_csv,
        trajectory_csv=trajectory_csv,
        family_csv=family_csv,
        family_macro_csv=family_macro_csv,
        paired_csv=paired_csv,
        geometry_factorial_csv=geometry_factorial_csv,
        completion_csv=completion_csv,
        readiness_json=readiness_json,
        manuscript_table_tex=manuscript_table,
        manuscript_ready=manuscript_ready,
    )
