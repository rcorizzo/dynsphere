from __future__ import annotations

import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.official_baselines.campaign import (
    campaign_document,
    merge_campaign_documents,
    resolve_profile,
    write_campaign_manifest,
)
from benchmarks.official_baselines.run_expanded_evaluation import RunConfig, build_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = REPO_ROOT / "benchmarks" / "official_baselines" / "hpc"


def _read(name: str) -> str:
    return HPC_ROOT.joinpath(name).read_text(encoding="utf-8")


def _run_au_script(name: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    executable = tmp_path / "python"
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'DEVICE=%s\\n' \"${DYNSPHERE_DEVICE:-unset}\"\n"
        "printf 'ARG=%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "DYNSPHERE_REPO": str(REPO_ROOT),
            "DYNSPHERE_PYTHON": str(executable),
            "DYNSPHERE_RESULT_ROOT": str(tmp_path / "results"),
            "LSB_JOBINDEX": "1",
        }
    )
    return subprocess.run(
        ["bash", str(HPC_ROOT / name)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )


def test_au_runner_uses_lsf_one_based_array_and_campaign_cli() -> None:
    script = _read("au_lsf_array.sh")

    assert "#BSUB -J" in script
    assert "#BSUB -q normal" in script
    assert "LSB_JOBINDEX" in script
    assert "LSB_JOBINDEX - 1" in script
    assert "benchmarks.official_baselines.run_campaign" in script
    assert '"$DYNSPHERE_REPO"' in script


def test_au_runner_loads_cluster_git_module_when_git_is_unavailable() -> None:
    script = _read("run_au_array.sh")

    git_check = script.index("command -v git")
    module_load = script.index('module load "${DYNSPHERE_GIT_MODULE:-git/2.40.0}"')
    campaign_execution = script.index("benchmarks.official_baselines.run_campaign")

    assert git_check < module_load < campaign_execution


def test_final_campaign_rejects_an_uncommitted_source_tree() -> None:
    for name in ("run_au_array.sh", "plgrid_slurm_array.sh"):
        script = _read(name)
        assert 'PROFILE" == "paper-multiseed"' in script
        assert "git status --porcelain" in script
        assert "requires a clean committed repository" in script


def test_plgrid_runner_uses_slurm_zero_based_array_and_explicit_resources() -> None:
    script = _read("plgrid_slurm_array.sh")

    assert script.startswith("#!/bin/bash -l")
    assert "#SBATCH --array=0-15" in script
    assert "#SBATCH --time=" in script
    assert "#SBATCH --mem=" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "benchmarks.official_baselines.run_campaign" in script
    assert '"$DYNSPHERE_REPO"' in script


def test_plgrid_runner_forwards_optional_method_filter() -> None:
    script = _read("plgrid_slurm_array.sh")

    assert 'DYNSPHERE_METHODS:-' in script
    assert 'read -r -a METHODS <<< "$DYNSPHERE_METHODS"' in script
    assert '--methods "${METHODS[@]}"' in script
    assert 'SHARD_INDEX="$SLURM_ARRAY_TASK_ID"' in script


def test_multiseed_au_components_select_disjoint_method_sets_and_manifests() -> None:
    expected = {
        "au_lsf_multiseed_h100_array.bsub": (
            "h100",
            "1-16",
            "paper-multiseed-au-h100.json",
            ("catch", "mtad_gat", "anomaly_transformer", "tranad"),
        ),
        "au_lsf_multiseed_cpu_array.bsub": (
            "long",
            "1-16",
            "paper-multiseed-au-external-cpu.json",
            ("dagmm", "omni_anomaly_tsb_ad"),
        ),
        "au_lsf_multiseed_local_array.bsub": (
            "long",
            "1-32",
            "paper-multiseed-au-local.json",
            (
                "pca_reconstruction",
                "kmeans_distance",
                "predictive_euclidean_single",
                "dynsphere",
                "dynsphere_euclidean",
                "dynsphere_one_regime",
                "dynsphere_euclidean_one_regime",
            ),
        ),
    }
    manifests = set()
    selected_methods = set()
    for filename, (queue, indices, manifest, methods) in expected.items():
        script = _read(filename)
        assert f'#BSUB -q {queue}' in script
        assert f'[{indices}]' in script
        assert 'DYNSPHERE_PROFILE=paper-multiseed' in script
        assert manifest in script
        assert 'DYNSPHERE_SHARD_COUNT=' in script
        for method in methods:
            assert method in script
            assert method not in selected_methods
            selected_methods.add(method)
        manifests.add(manifest)
    assert len(manifests) == 3
    assert selected_methods == set(resolve_profile("paper-multiseed").methods)


def test_multiseed_component_manifests_merge_to_complete_unique_matrix(
    tmp_path: Path,
) -> None:
    profile = resolve_profile("paper-multiseed")
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
    groups = (
        {"dagmm", "omni_anomaly_tsb_ad"},
        {"catch", "mtad_gat", "anomaly_transformer", "tranad"},
        {
            "pca_reconstruction",
            "kmeans_distance",
            "predictive_euclidean_single",
            "dynsphere",
            "dynsphere_euclidean",
            "dynsphere_one_regime",
            "dynsphere_euclidean_one_regime",
        },
    )
    documents = tuple(
        campaign_document(
            f"paper-component-{index}",
            tuple(job for job in matrix if job.implementation.method_id in methods),
            shard_count=16 if index < 2 else 32,
        )
        for index, methods in enumerate(groups)
    )
    merged = merge_campaign_documents(documents)
    assert merged["expected_job_count"] == 2_457
    identities = {
        (
            row["case_id"],
            row["seed"],
            row["method_id"],
        )
        for row in merged["expected_jobs"]
    }
    assert len(identities) == 2_457
    aggregate = _read("au_lsf_multiseed_aggregate.bsub")
    cpu = aggregate.index("paper-multiseed-au-external-cpu.json")
    gpu = aggregate.index("paper-multiseed-au-h100.json")
    local = aggregate.index("paper-multiseed-au-local.json")
    assert cpu < gpu < local
    assert "paper-multiseed-au-complete.json" in aggregate


def test_component_manifest_collision_rejects_changed_membership_or_revision(
    tmp_path: Path,
) -> None:
    profile = resolve_profile("paper-multiseed")
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
    component = tuple(
        job for job in matrix if job.implementation.method_id == "dagmm"
    )
    original = campaign_document("paper-component", component, 16)
    path = write_campaign_manifest(tmp_path / "component.json", original)

    changed_methods = campaign_document("paper-component", matrix[:2], 16)
    with pytest.raises(ValueError, match="different campaign"):
        write_campaign_manifest(path, changed_methods)

    changed_revision = deepcopy(original)
    changed_revision["expected_jobs"][0]["adapter_revision"] = "f" * 40
    with pytest.raises(ValueError, match="different campaign"):
        write_campaign_manifest(path, changed_revision)


def test_setup_script_prepares_every_registered_external_method() -> None:
    script = _read("setup_environment.sh")

    for method in (
        "dagmm",
        "catch",
        "mtad_gat",
        "omni_anomaly",
        "omni_anomaly_tsb_ad",
        "anomaly_transformer",
        "tranad",
    ):
        assert method in script
    assert "prepare_source" in script
    assert "pytest" in script


def test_au_cpu_array_executes_only_cpu_methods_with_distinct_manifest(
    tmp_path: Path,
) -> None:
    completed = _run_au_script("au_lsf_cpu_array.bsub", tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "ARG=dagmm" in completed.stdout
    assert "ARG=omni_anomaly_tsb_ad" in completed.stdout
    assert "ARG=catch" not in completed.stdout
    assert "hpc-seed37-au-cpu.json" in completed.stdout
    assert "DEVICE=cpu" in completed.stdout


def test_au_gpu_arrays_execute_gpu_methods_and_require_cuda(tmp_path: Path) -> None:
    expected = ("catch", "mtad_gat", "anomaly_transformer", "tranad")
    for name in ("au_lsf_h100_array.bsub", "au_lsf_l40_array.bsub"):
        completed = _run_au_script(name, tmp_path)
        assert completed.returncode == 0, completed.stderr
        for method in expected:
            assert f"ARG={method}" in completed.stdout
        assert "ARG=dagmm" not in completed.stdout
        assert "hpc-seed37-au-gpu.json" in completed.stdout
        assert "DEVICE=cuda" in completed.stdout


def test_au_gpu_resource_requests_match_cluster_conventions() -> None:
    h100 = _read("au_lsf_h100_array.bsub")
    l40 = _read("au_lsf_l40_array.bsub")

    assert "#BSUB -q h100" in h100
    assert '#BSUB -gpu "num=1:gmodel=NVIDIAH100PCIe"' in h100
    assert "#BSUB -q normal" in l40
    assert '#BSUB -gpu "num=1"' in l40


def test_au_corrected_catch_array_isolated_from_existing_gpu_campaign(
    tmp_path: Path,
) -> None:
    completed = _run_au_script("au_lsf_h100_catch_corrected_array.bsub", tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "ARG=catch" in completed.stdout
    for method in ("mtad_gat", "anomaly_transformer", "tranad"):
        assert f"ARG={method}" not in completed.stdout
    assert "hpc-seed37-au-catch-overlap-v1.json" in completed.stdout
    assert "DEVICE=cuda" in completed.stdout


def test_au_aggregation_merges_component_manifests_with_catch_precedence() -> None:
    script = _read("au_lsf_aggregate.bsub")

    assert "merge_campaign_manifests" in script
    cpu = script.index("hpc-seed37-au-cpu.json")
    gpu = script.index("hpc-seed37-au-gpu.json")
    catch = script.index("hpc-seed37-au-catch-overlap-v1.json")
    assert cpu < gpu < catch


def test_hpc_scripts_have_valid_bash_syntax_and_are_executable() -> None:
    paths = sorted(HPC_ROOT.glob("*.sh")) + sorted(HPC_ROOT.glob("*.bsub"))
    for path in paths:
        completed = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        assert os.access(path, os.X_OK)
