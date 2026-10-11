from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.official_baselines.contract import EvaluationCase
from benchmarks.official_baselines.provenance import ImplementationSpec
from benchmarks.official_baselines.run_expanded_evaluation import (
    CaseDescriptor,
    ExperimentJob,
    RunConfig,
    build_matrix,
    execute_job,
)


def local_spec(revision: str = "a" * 40) -> ImplementationSpec:
    return ImplementationSpec(
        method_id="pca_reconstruction",
        display_name="PCA reconstruction",
        source_kind="local",
        implementation_category="repository-maintained",
        repository_url=".",
        revision=revision,
        python=">=3.9",
        adapter_command=("__local__", "pca_reconstruction"),
        default_hyperparameters={"variance_retained": 0.95},
        fallback_ids=(),
    )


def case(seed: int = 37, offset: float = 0.0, scaling: str = "scale-a") -> EvaluationCase:
    return EvaluationCase(
        case_id="Fixture/trajectory",
        dataset_id="Fixture",
        source_family="Fixture",
        trajectory_id="trajectory",
        train=np.array([[0.0, 0.0], [1.0, 1.0]]) + offset,
        test=np.array([[0.0, 1.0], [2.0, 3.0]]) + offset,
        labels=np.array([0, 1]),
        metadata={"seed": seed, "scaling_rule": scaling},
    )


def descriptor(value: EvaluationCase) -> CaseDescriptor:
    return CaseDescriptor(
        case_id=value.case_id,
        dataset_id=value.dataset_id,
        source_family=value.source_family,
        trajectory_id=value.trajectory_id,
        seed=int(value.metadata["seed"]),
        loader=lambda: value,
    )


def job(tmp_path: Path, value: EvaluationCase | None = None) -> ExperimentJob:
    selected = value or case()
    return ExperimentJob(
        case=descriptor(selected),
        implementation=local_spec(),
        hyperparameters={"variance_retained": 0.95},
        result_root=tmp_path / "results" / "official_baselines_v1",
        timeout_seconds=30,
        resume=True,
    )


def write_manifest_and_files(tmp_path: Path) -> tuple[Path, Path]:
    names = [
        "000_MSL_id_1_Facility_tr_10_1st_5.csv",
        "001_SMAP_id_1_Facility_tr_10_1st_5.csv",
        "002_SMD_id_1_Facility_tr_10_1st_5.csv",
    ]
    names.extend(
        f"{index:03d}_GHL_id_{index}_Facility_tr_10_1st_5.csv"
        for index in range(3, 180)
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file_name\n" + "\n".join(names) + "\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in names:
        data_root.joinpath(name).touch()
    return manifest, data_root


def test_matrix_uses_three_current_seeds_one_tsb_seed_and_deduplicates(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, data_root = write_manifest_and_files(tmp_path)
    monkeypatch.setattr(
        "benchmarks.official_baselines.run_expanded_evaluation.load_registry",
        lambda path: {"pca_reconstruction": local_spec()},
    )
    selected = (
        "DynSphere-Contextual",
        "MSL",
        "SMAP",
        "SMD",
        "BATADAL",
        "HAI-20.07",
        "PowerSystem",
        "TSB-AD-M",
    )
    config = RunConfig(
        methods=("pca_reconstruction",),
        datasets=selected,
        seeds=(37, 38, 39),
        tsb_manifest=manifest,
        tsb_data_root=data_root,
        result_root=tmp_path / "official_baselines_v1",
    )

    matrix = build_matrix(config)

    for family in (
        "DynSphere-Contextual",
        "MSL",
        "SMAP",
        "SMD",
        "BATADAL",
        "HAI-20.07",
        "PowerSystem",
    ):
        assert {entry.case.seed for entry in matrix if entry.case.source_family == family} == {
            37,
            38,
            39,
        }
    ghl = [entry for entry in matrix if entry.case.source_family == "GHL"]
    assert len(ghl) == 177
    assert {entry.case.seed for entry in ghl} == {37}
    keys = [
        (entry.case.case_id, entry.case.seed, entry.implementation.method_id)
        for entry in matrix
    ]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)


@pytest.mark.parametrize("status", ["pending", "failed", "timeout", "incompatible"])
def test_resume_reexecutes_noncomplete_records(
    tmp_path: Path, monkeypatch, status: str
) -> None:
    calls = []

    def scorer(train, test, seed):
        calls.append(seed)
        return np.arange(len(test), dtype=float), {"scorer": "fixture"}

    monkeypatch.setitem(
        __import__(
            "benchmarks.official_baselines.run_expanded_evaluation",
            fromlist=["LOCAL_SCORERS"],
        ).LOCAL_SCORERS,
        "pca_reconstruction",
        scorer,
    )
    selected = job(tmp_path)
    first = execute_job(selected)
    record_path = Path(first["run_directory"]) / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = status
    record_path.write_text(json.dumps(record), encoding="utf-8")

    second = execute_job(selected)

    assert len(calls) == 2
    assert second["status"] == "complete"
    assert second["skipped"] is False


def test_resume_skips_only_valid_matching_complete_record(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def scorer(train, test, seed):
        calls.append(seed)
        return np.arange(len(test), dtype=float), {}

    module = __import__(
        "benchmarks.official_baselines.run_expanded_evaluation",
        fromlist=["LOCAL_SCORERS"],
    )
    monkeypatch.setitem(module.LOCAL_SCORERS, "pca_reconstruction", scorer)
    selected = job(tmp_path)

    first = execute_job(selected)
    second = execute_job(selected)

    assert len(calls) == 1
    assert first["skipped"] is False
    assert second["skipped"] is True

    record_path = Path(first["run_directory"]) / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["run_identity"] = "mismatch"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    third = execute_job(selected)
    assert len(calls) == 2
    assert third["skipped"] is False


def test_identity_inputs_produce_distinct_run_directories(tmp_path: Path, monkeypatch) -> None:
    module = __import__(
        "benchmarks.official_baselines.run_expanded_evaluation",
        fromlist=["LOCAL_SCORERS"],
    )
    monkeypatch.setitem(
        module.LOCAL_SCORERS,
        "pca_reconstruction",
        lambda train, test, seed: (np.arange(len(test), dtype=float), {}),
    )
    base = job(tmp_path)
    variants = [
        base,
        replace(base, implementation=local_spec("b" * 40)),
        replace(base, hyperparameters={"variance_retained": 0.90}),
        job(tmp_path, case(seed=38)),
        job(tmp_path, case(offset=10.0)),
        job(tmp_path, case(scaling="scale-b")),
    ]

    directories = {execute_job(entry)["run_directory"] for entry in variants}

    assert len(directories) == len(variants)


def test_corrupt_complete_record_executes_again(tmp_path: Path, monkeypatch) -> None:
    calls = []
    module = __import__(
        "benchmarks.official_baselines.run_expanded_evaluation",
        fromlist=["LOCAL_SCORERS"],
    )

    def scorer(train, test, seed):
        calls.append(seed)
        return np.arange(len(test), dtype=float), {}

    monkeypatch.setitem(module.LOCAL_SCORERS, "pca_reconstruction", scorer)
    selected = job(tmp_path)
    first = execute_job(selected)
    Path(first["run_directory"]).joinpath("run.json").write_text("{", encoding="utf-8")

    execute_job(selected)

    assert len(calls) == 2


def test_execution_preserves_files_outside_versioned_result_root(
    tmp_path: Path, monkeypatch
) -> None:
    historical = tmp_path / "historical.csv"
    historical.write_bytes(b"original historical result\n")
    module = __import__(
        "benchmarks.official_baselines.run_expanded_evaluation",
        fromlist=["LOCAL_SCORERS"],
    )
    monkeypatch.setitem(
        module.LOCAL_SCORERS,
        "pca_reconstruction",
        lambda train, test, seed: (np.arange(len(test), dtype=float), {}),
    )

    result = execute_job(job(tmp_path))

    assert historical.read_bytes() == b"original historical result\n"
    assert Path(result["run_directory"]).is_relative_to(
        tmp_path / "results" / "official_baselines_v1"
    )
