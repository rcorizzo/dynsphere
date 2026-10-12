from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HPC_ROOT = REPO_ROOT / "benchmarks" / "official_baselines" / "hpc"


def _read(name: str) -> str:
    return HPC_ROOT.joinpath(name).read_text(encoding="utf-8")


def test_au_runner_uses_lsf_one_based_array_and_campaign_cli() -> None:
    script = _read("au_lsf_array.sh")

    assert "#BSUB -J" in script
    assert "#BSUB -q normal" in script
    assert "LSB_JOBINDEX" in script
    assert "LSB_JOBINDEX - 1" in script
    assert "benchmarks.official_baselines.run_campaign" in script
    assert '"$DYNSPHERE_REPO"' in script


def test_plgrid_runner_uses_slurm_zero_based_array_and_explicit_resources() -> None:
    script = _read("plgrid_slurm_array.sh")

    assert script.startswith("#!/bin/bash -l")
    assert "#SBATCH --array=0-15" in script
    assert "#SBATCH --time=" in script
    assert "#SBATCH --mem=" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "benchmarks.official_baselines.run_campaign" in script
    assert '"$DYNSPHERE_REPO"' in script


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


def test_hpc_scripts_have_valid_bash_syntax_and_are_executable() -> None:
    for path in sorted(HPC_ROOT.glob("*.sh")):
        completed = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 0, completed.stderr
        assert os.access(path, os.X_OK)
