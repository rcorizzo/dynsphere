from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.official_baselines import aggregate, metrics
from benchmarks.official_baselines.aggregate import aggregate_runs
from benchmarks.official_baselines.contract import write_run_record
from benchmarks.official_baselines.metrics import compute_metrics


def test_metrics_match_sklearn_and_require_official_vus(monkeypatch) -> None:
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=int)
    scores = np.array([0.05, 0.2, 0.8, 0.7, 0.4, 0.9, 0.1, 0.3])
    monkeypatch.setattr(
        metrics,
        "range_vus_scores",
        lambda labels, scores, sliding_window, prefer_official: {
            "vus_roc": 0.81,
            "vus_pr": 0.72,
            "vus_window": sliding_window,
            "vus_backend": "official_vus",
        },
    )

    result = compute_metrics(labels, scores, event_budget=0.25, vus_window=7)

    assert result["auprc"] == pytest.approx(average_precision_score(labels, scores))
    assert result["auroc"] == pytest.approx(roc_auc_score(labels, scores))
    assert result["pauc_fpr_05"] == pytest.approx(
        roc_auc_score(labels, scores, max_fpr=0.05)
    )
    assert result["vus_pr"] == 0.72
    assert result["vus_roc"] == 0.81
    assert result["vus_window"] == 7
    assert result["vus_backend"] == "official_vus"
    assert result["manuscript_ready"] is True


def test_fixed_budget_event_metrics_use_ranked_score_budget(monkeypatch) -> None:
    labels = np.array([0, 1, 1, 0, 0, 1, 0, 0], dtype=int)
    scores = np.array([0.0, 0.9, 0.1, 0.0, 0.0, 0.8, 0.0, 0.0])
    monkeypatch.setattr(
        metrics,
        "range_vus_scores",
        lambda *args, **kwargs: {
            "vus_roc": 0.5,
            "vus_pr": 0.5,
            "vus_window": 1,
            "vus_backend": "buffered_fallback",
        },
    )

    result = compute_metrics(labels, scores, event_budget=0.25, vus_window=1)

    assert result["event_budget"] == 0.25
    assert result["event_predictions"] == 2
    assert result["event_recall"] == 1.0
    assert result["event_precision"] == 1.0
    assert result["manuscript_ready"] is False


def write_result(
    root: Path,
    method: str,
    family: str,
    trajectory: str,
    seed: int,
    value: float,
    status: str = "complete",
    adapter_revision: str | None = None,
    identity_suffix: str | None = None,
) -> None:
    identity_suffix = identity_suffix or (
        adapter_revision[:8] if adapter_revision else "identity"
    )
    run_dir = root / "runs" / family / trajectory / method / f"seed-{seed}" / identity_suffix
    record = {
        "status": status,
        "run_identity": f"{method}-{family}-{trajectory}-{seed}",
        "case_id": f"Dataset/{trajectory}",
        "dataset_id": "Dataset",
        "source_family": family,
        "trajectory_id": trajectory,
        "seed": seed,
        "method_id": method,
        "executed_method_id": method,
        "elapsed_seconds": value + 0.5,
        "run_directory": str(run_dir),
    }
    if adapter_revision is not None:
        record["adapter_revision"] = adapter_revision
    write_run_record(run_dir / "run.json", record)
    if status == "complete":
        private = run_dir / "case" / "private"
        execution = run_dir / "execution"
        private.mkdir(parents=True)
        execution.mkdir(parents=True)
        np.save(private / "labels.npy", np.array([0, 1]))
        np.save(execution / "scores.npy", np.array([value, value + 1.0]))


def metric_stub(labels, scores, event_budget, vus_window):
    value = float(scores[0])
    return {
        "auprc": value,
        "auroc": value,
        "pauc_fpr_05": value,
        "vus_pr": value,
        "vus_roc": value,
        "vus_window": 1,
        "vus_backend": "official_vus",
        "event_budget": event_budget,
        "event_predictions": 1,
        "event_recall": value,
        "event_precision": value,
        "manuscript_ready": True,
    }


def test_family_weighting_precedes_unweighted_family_macro(tmp_path: Path, monkeypatch) -> None:
    result_root = tmp_path / "results"
    write_result(result_root, "method-a", "Family-A", "a1", 37, 1.0)
    write_result(result_root, "method-a", "Family-A", "a2", 37, 3.0)
    write_result(result_root, "method-a", "Family-B", "b1", 37, 9.0)
    monkeypatch.setattr(aggregate, "compute_metrics", metric_stub)

    artifacts = aggregate_runs(
        result_root, tmp_path / "aggregate", bootstrap_replicates=100
    )
    family = pd.read_csv(artifacts.family_csv)
    macro = pd.read_csv(artifacts.family_macro_csv)

    assert family.loc[family.source_family == "Family-A", "auprc"].item() == 2.0
    assert family.loc[family.source_family == "Family-B", "auprc"].item() == 9.0
    assert macro.loc[macro.method_id == "method-a", "auprc"].item() == 5.5


def test_paired_analysis_reports_interval_win_rate_effect_and_missing_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    result_root = tmp_path / "results"
    for trajectory, left, right in (("t1", 4.0, 1.0), ("t2", 2.0, 3.0), ("t3", 7.0, 2.0)):
        write_result(result_root, "method-a", "Family", trajectory, 37, left)
        write_result(result_root, "method-b", "Family", trajectory, 37, right)
    write_result(result_root, "method-a", "Family", "unpaired", 37, 8.0)
    monkeypatch.setattr(aggregate, "compute_metrics", metric_stub)

    artifacts = aggregate_runs(
        result_root,
        tmp_path / "aggregate",
        bootstrap_seed=11,
        bootstrap_replicates=500,
    )
    paired = pd.read_csv(artifacts.paired_csv)
    row = paired[
        (paired.method_a == "method-a")
        & (paired.method_b == "method-b")
        & (paired.metric == "auprc")
    ].iloc[0]

    assert row.paired_count == 3
    assert row.incomplete_pair_count == 1
    assert row.mean_difference == pytest.approx((3.0 - 1.0 + 5.0) / 3.0)
    assert row.win_rate == pytest.approx(2.0 / 3.0)
    assert -1.0 <= row.rank_biserial <= 1.0
    assert row.ci_low <= row.mean_difference <= row.ci_high


def test_dynsphere_factorial_reports_four_named_paired_contrasts(
    tmp_path: Path, monkeypatch
) -> None:
    result_root = tmp_path / "results"
    values = {
        "dynsphere": (1.0, 2.0),
        "dynsphere_euclidean": (3.0, 5.0),
        "dynsphere_one_regime": (0.5, 1.0),
        "dynsphere_euclidean_one_regime": (2.0, 2.0),
    }
    for method, method_values in values.items():
        for index, value in enumerate(method_values, start=1):
            write_result(result_root, method, "Family", f"t{index}", 37, value)
    monkeypatch.setattr(aggregate, "compute_metrics", metric_stub)

    artifacts = aggregate_runs(
        result_root,
        tmp_path / "aggregate",
        bootstrap_seed=13,
        bootstrap_replicates=200,
    )
    factorial = pd.read_csv(artifacts.geometry_factorial_csv)

    assert set(factorial.contrast) == {"E4-S4", "E1-S1", "S4-S1", "E4-E1"}
    assert set(factorial.metric) == {"auprc", "vus_pr"}
    assert (factorial.paired_count == 2).all()
    assert (factorial.incomplete_pair_count == 0).all()
    assert factorial[["mean_difference", "ci_low", "ci_high", "win_rate"]].notna().all().all()
    e4_s4 = factorial[
        (factorial.contrast == "E4-S4") & (factorial.metric == "auprc")
    ].iloc[0]
    assert e4_s4.mean_difference == pytest.approx(2.5)
    assert e4_s4.win_rate == 1.0


def test_readiness_identifies_missing_runs_metric_warnings_and_mixed_revisions(
    tmp_path: Path, monkeypatch
) -> None:
    result_root = tmp_path / "results"
    revision_a = "a" * 40
    revision_b = "b" * 40
    write_result(
        result_root,
        "dynsphere",
        "Family",
        "spherical-a",
        37,
        1.0,
        adapter_revision=revision_a,
    )
    write_result(
        result_root,
        "dynsphere",
        "Family",
        "spherical-b",
        37,
        2.0,
        adapter_revision=revision_b,
    )
    write_result(
        result_root,
        "mtad_gat",
        "Family",
        "invalid-scores",
        37,
        3.0,
        adapter_revision=revision_a,
    )
    invalid_scores = next(result_root.glob("runs/**/invalid-scores/**/scores.npy"))
    np.save(invalid_scores, np.array([1.0]))

    expected = [
        {
            "adapter_revision": revision_a,
            "case_id": "Dataset/spherical-a",
            "dataset_id": "Dataset",
            "method_id": "dynsphere",
            "seed": 37,
            "source_family": "Family",
            "trajectory_id": "spherical-a",
        },
        {
            "adapter_revision": revision_b,
            "case_id": "Dataset/spherical-b",
            "dataset_id": "Dataset",
            "method_id": "dynsphere",
            "seed": 37,
            "source_family": "Family",
            "trajectory_id": "spherical-b",
        },
        {
            "adapter_revision": revision_a,
            "case_id": "Dataset/invalid-scores",
            "dataset_id": "Dataset",
            "method_id": "mtad_gat",
            "seed": 37,
            "source_family": "Family",
            "trajectory_id": "invalid-scores",
        },
        {
            "adapter_revision": revision_a,
            "case_id": "Dataset/missing",
            "dataset_id": "Dataset",
            "method_id": "catch",
            "seed": 37,
            "source_family": "Family",
            "trajectory_id": "missing",
        },
    ]
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "official-baseline-campaign-v1",
                "campaign_id": "strict-readiness",
                "expected_job_count": len(expected),
                "expected_jobs": expected,
            }
        ),
        encoding="utf-8",
    )

    def warning_metric_stub(labels, scores, event_budget, vus_window):
        result = metric_stub(labels, scores, event_budget, vus_window)
        if float(scores[0]) == 2.0:
            result["vus_backend"] = "buffered_fallback"
            result["manuscript_ready"] = False
        return result

    monkeypatch.setattr(aggregate, "compute_metrics", warning_metric_stub)
    artifacts = aggregate_runs(
        result_root,
        tmp_path / "aggregate",
        bootstrap_replicates=100,
        campaign_manifest=manifest,
    )
    report = json.loads(artifacts.readiness_json.read_text(encoding="utf-8"))

    assert artifacts.manuscript_ready is False
    assert "catch" in report["missing_methods"]
    assert report["missing_run_identities"] == [expected[-1]]
    assert report["metric_errors"][0]["run_identity"] == "mtad_gat-Family-invalid-scores-37"
    assert report["metric_warnings"] == [
        {
            "run_identity": "dynsphere-Family-spherical-b-37",
            "warning": "vus_backend=buffered_fallback",
        }
    ]
    assert report["adapter_revisions_by_method"]["dynsphere"] == [
        revision_a,
        revision_b,
    ]
    assert report["methods_with_multiple_adapter_revisions"] == ["dynsphere"]
    assert report["geometry_factorial_csv"] == str(artifacts.geometry_factorial_csv)


def test_completion_counts_and_missing_valid_method_block_manuscript_table(
    tmp_path: Path, monkeypatch
) -> None:
    result_root = tmp_path / "results"
    write_result(result_root, "complete-method", "Family", "t1", 37, 1.0)
    write_result(result_root, "failed-method", "Family", "t1", 37, 0.0, "failed")
    write_result(result_root, "timeout-method", "Family", "t1", 37, 0.0, "timeout")
    write_result(
        result_root, "incompatible-method", "Family", "t1", 37, 0.0, "incompatible"
    )
    monkeypatch.setattr(aggregate, "compute_metrics", metric_stub)

    artifacts = aggregate_runs(
        result_root, tmp_path / "aggregate", bootstrap_replicates=100
    )
    completion = pd.read_csv(artifacts.completion_csv).set_index("status")["count"]
    report = json.loads(artifacts.readiness_json.read_text(encoding="utf-8"))

    assert completion.to_dict() == {
        "complete": 1,
        "failed": 1,
        "incompatible": 1,
        "timeout": 1,
    }
    assert artifacts.manuscript_ready is False
    assert artifacts.manuscript_table_tex is None
    assert set(report["methods_without_valid_runs"]) == {
        "failed-method",
        "incompatible-method",
        "timeout-method",
    }


def test_aggregation_can_exclude_stale_adapter_revisions(
    tmp_path: Path, monkeypatch
) -> None:
    result_root = tmp_path / "results"
    write_result(
        result_root,
        "method-a",
        "Family",
        "trajectory",
        37,
        1.0,
        adapter_revision="a" * 40,
    )
    write_result(
        result_root,
        "method-a",
        "Family",
        "trajectory",
        37,
        9.0,
        adapter_revision="b" * 40,
    )
    monkeypatch.setattr(aggregate, "compute_metrics", metric_stub)

    artifacts = aggregate_runs(
        result_root,
        tmp_path / "aggregate",
        bootstrap_replicates=100,
        expected_adapter_revision="b" * 40,
    )
    per_run = pd.read_csv(artifacts.per_run_csv)
    report = json.loads(artifacts.readiness_json.read_text(encoding="utf-8"))

    assert per_run.auprc.tolist() == [9.0]
    assert report["expected_adapter_revision"] == "b" * 40
    assert report["excluded_stale_record_count"] == 1


def test_campaign_manifest_excludes_unlisted_records(tmp_path: Path, monkeypatch) -> None:
    result_root = tmp_path / "results"
    revision = "c" * 40
    write_result(
        result_root,
        "method-a",
        "Family",
        "trajectory",
        37,
        2.0,
        adapter_revision=revision,
    )
    write_result(
        result_root,
        "method-b",
        "Family",
        "trajectory",
        37,
        8.0,
        adapter_revision=revision,
    )
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "official-baseline-campaign-v1",
                "campaign_id": "campaign-123",
                "expected_job_count": 1,
                "expected_jobs": [
                    {
                        "adapter_revision": revision,
                        "case_id": "Dataset/trajectory",
                        "dataset_id": "Dataset",
                        "method_id": "method-a",
                        "seed": 37,
                        "source_family": "Family",
                        "trajectory_id": "trajectory",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(aggregate, "compute_metrics", metric_stub)

    artifacts = aggregate_runs(
        result_root,
        tmp_path / "aggregate",
        bootstrap_replicates=100,
        campaign_manifest=manifest,
    )

    per_run = pd.read_csv(artifacts.per_run_csv)
    report = json.loads(artifacts.readiness_json.read_text(encoding="utf-8"))
    assert per_run.method_id.tolist() == ["method-a"]
    assert report["campaign_id"] == "campaign-123"
    assert report["excluded_noncampaign_record_count"] == 1


def test_campaign_aggregation_excludes_superseded_records_for_same_job(
    tmp_path: Path, monkeypatch
) -> None:
    result_root = tmp_path / "results"
    revision = "d" * 40
    write_result(
        result_root,
        "method-a",
        "Family",
        "trajectory",
        37,
        0.0,
        status="incompatible",
        adapter_revision=revision,
        identity_suffix="old",
    )
    write_result(
        result_root,
        "method-a",
        "Family",
        "trajectory",
        37,
        7.0,
        adapter_revision=revision,
        identity_suffix="new",
    )
    old = next(result_root.glob("runs/**/old/run.json"))
    new = next(result_root.glob("runs/**/new/run.json"))
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    manifest = tmp_path / "campaign.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "official-baseline-campaign-v1",
                "campaign_id": "campaign-superseded",
                "expected_job_count": 1,
                "expected_jobs": [
                    {
                        "adapter_revision": revision,
                        "case_id": "Dataset/trajectory",
                        "dataset_id": "Dataset",
                        "method_id": "method-a",
                        "seed": 37,
                        "source_family": "Family",
                        "trajectory_id": "trajectory",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(aggregate, "compute_metrics", metric_stub)

    artifacts = aggregate_runs(
        result_root,
        tmp_path / "aggregate",
        bootstrap_replicates=100,
        campaign_manifest=manifest,
    )

    per_run = pd.read_csv(artifacts.per_run_csv)
    completion = pd.read_csv(artifacts.completion_csv).set_index("status")["count"]
    report = json.loads(artifacts.readiness_json.read_text(encoding="utf-8"))
    assert per_run.auprc.tolist() == [7.0]
    assert completion.to_dict() == {"complete": 1}
    assert report["excluded_superseded_record_count"] == 1
