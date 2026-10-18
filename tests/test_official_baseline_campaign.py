from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.official_baselines.campaign import (
    campaign_document,
    merge_campaign_documents,
    partition_jobs,
    resolve_profile,
    scheduler_shard_index,
    write_campaign_manifest,
)
from benchmarks.official_baselines.provenance import ImplementationSpec
from benchmarks.official_baselines.run_expanded_evaluation import (
    CaseDescriptor,
    ExperimentJob,
    RunConfig,
    build_matrix,
)


def _job(case_number: int, method: str = "dagmm") -> ExperimentJob:
    implementation = ImplementationSpec(
        method_id=method,
        display_name=method,
        source_kind="external",
        implementation_category="author-released",
        repository_url="https://example.test/repository.git",
        revision="a" * 40,
        python="3.10",
        adapter_command=("python", "adapter.py"),
        default_hyperparameters={},
        fallback_ids=(),
        adapter_revision="b" * 40,
    )
    case = CaseDescriptor(
        case_id=f"TSB-AD-M/case-{case_number}",
        dataset_id="TSB-AD-M",
        source_family="MSL",
        trajectory_id=f"case-{case_number}",
        seed=37,
        loader=lambda: None,
    )
    return ExperimentJob(
        case=case,
        implementation=implementation,
        hyperparameters={},
        result_root=Path("results"),
        timeout_seconds=60,
        resume=True,
    )


def test_profiles_select_the_approved_execution_stages() -> None:
    macbook = resolve_profile("macbook-safe")
    hpc = resolve_profile("hpc-seed37")

    assert macbook.methods == ("dagmm", "tranad", "omni_anomaly_tsb_ad")
    assert macbook.datasets == ("MSL", "SMAP", "SMD")
    assert macbook.seeds == (37,)
    assert hpc.methods == (
        "dagmm",
        "catch",
        "mtad_gat",
        "omni_anomaly",
        "anomaly_transformer",
        "tranad",
    )
    assert hpc.datasets[-1] == "TSB-AD-M"
    assert hpc.seeds == (37,)


def test_paper_multiseed_profile_has_the_complete_matrix(tmp_path: Path) -> None:
    profile = resolve_profile("paper-multiseed")
    assert profile.methods == (
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
    assert profile.datasets == (
        "DynSphere-Contextual",
        "MSL",
        "SMAP",
        "SMD",
        "BATADAL",
        "HAI-20.07",
        "PowerSystem",
    )
    assert profile.seeds == (37, 38, 39)
    assert profile.timeout_seconds == 14_400

    matrix = build_matrix(
        RunConfig(
            methods=profile.methods,
            datasets=profile.datasets,
            seeds=profile.seeds,
            result_root=tmp_path,
            timeout_seconds=profile.timeout_seconds,
            dry_run=True,
        )
    )
    assert len(matrix) == 2_457
    for method in profile.methods:
        assert sum(job.implementation.method_id == method for job in matrix) == 189
    for seed in profile.seeds:
        assert sum(job.case.seed == seed for job in matrix) == 819


@pytest.mark.parametrize(
    ("index", "count", "message"),
    [(-1, 2, "non-negative"), (0, 0, "positive"), (2, 2, "smaller")],
)
def test_partition_rejects_invalid_shards(index: int, count: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        partition_jobs((_job(0),), shard_index=index, shard_count=count)


def test_shards_are_disjoint_and_cover_the_stable_matrix() -> None:
    jobs = tuple(_job(index) for index in range(11))
    shards = [partition_jobs(jobs, index, 4) for index in range(4)]

    flattened = [job for shard in shards for job in shard]
    assert len({id(job) for job in flattened}) == len(jobs)
    assert {job.case.case_id for job in flattened} == {
        job.case.case_id for job in jobs
    }
    assert [job.case.case_id for job in shards[2]] == [
        "TSB-AD-M/case-2",
        "TSB-AD-M/case-6",
        "TSB-AD-M/case-10",
    ]


def test_scheduler_index_conversion_handles_lsf_and_slurm() -> None:
    assert scheduler_shard_index("1", one_based=True) == 0
    assert scheduler_shard_index("7", one_based=True) == 6
    assert scheduler_shard_index("0", one_based=False) == 0
    with pytest.raises(ValueError, match="one-based"):
        scheduler_shard_index("0", one_based=True)


def test_campaign_manifest_is_deterministic_and_records_exact_jobs(tmp_path: Path) -> None:
    jobs = (_job(0), _job(1, "tranad"))
    first = campaign_document("macbook-safe", jobs, shard_count=3)
    second = campaign_document("macbook-safe", tuple(reversed(jobs)), shard_count=3)

    assert first == second
    assert len(first["campaign_id"]) == 64
    assert first["shard_count"] == 3
    assert first["expected_jobs"][0] == {
        "adapter_revision": "b" * 40,
        "case_id": "TSB-AD-M/case-0",
        "dataset_id": "TSB-AD-M",
        "method_id": "dagmm",
        "seed": 37,
        "source_family": "MSL",
        "trajectory_id": "case-0",
    }

    path = write_campaign_manifest(tmp_path / "campaign.json", first)
    assert json.loads(path.read_text(encoding="utf-8")) == first


def test_existing_different_campaign_manifest_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    path.write_text('{"campaign_id": "different"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="different campaign"):
        write_campaign_manifest(path, campaign_document("test", (_job(0),), 1))


def test_later_campaign_manifest_replaces_same_logical_method_case() -> None:
    original = campaign_document("gpu-original", (_job(0, "catch"), _job(0, "mtad_gat")), 2)
    corrected_job = _job(0, "catch")
    corrected_job = ExperimentJob(
        case=corrected_job.case,
        implementation=ImplementationSpec(
            **{
                **corrected_job.implementation.__dict__,
                "adapter_revision": "c" * 40,
            }
        ),
        hyperparameters=corrected_job.hyperparameters,
        result_root=corrected_job.result_root,
        timeout_seconds=corrected_job.timeout_seconds,
        resume=corrected_job.resume,
    )
    corrected = campaign_document("catch-corrected", (corrected_job,), 2)

    merged = merge_campaign_documents((original, corrected))

    assert merged["expected_job_count"] == 2
    revisions = {
        row["method_id"]: row["adapter_revision"] for row in merged["expected_jobs"]
    }
    assert revisions == {"catch": "c" * 40, "mtad_gat": "b" * 40}
    assert merged["source_campaign_ids"] == [
        original["campaign_id"],
        corrected["campaign_id"],
    ]


def test_campaign_aggregation_cli_exposes_manifest_filter() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.official_baselines.aggregate_campaign",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--campaign-manifest" in completed.stdout
    assert "--result-root" in completed.stdout
