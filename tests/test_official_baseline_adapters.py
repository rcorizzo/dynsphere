from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.official_baselines import setup_external
from benchmarks.official_baselines.adapters import (
    anomaly_transformer,
    catch,
    mtad_gat,
    omni_anomaly,
    tab_dagmm,
    tranad,
)
from benchmarks.official_baselines.adapters.common import BackendOutput
from benchmarks.official_baselines.provenance import ImplementationSpec
from benchmarks.official_baselines.setup_external import prepare_source, verify_source


ADAPTERS = [tab_dagmm, catch, mtad_gat, omni_anomaly, anomaly_transformer, tranad]


def git(*arguments: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def make_source_repository(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "source"
    source.mkdir()
    git("init", cwd=source)
    git("config", "user.email", "fixture@example.invalid", cwd=source)
    git("config", "user.name", "Fixture", cwd=source)
    (source / "README.md").write_text("fixture\n", encoding="utf-8")
    git("add", "README.md", cwd=source)
    git("commit", "-m", "fixture", cwd=source)
    return source, git("rev-parse", "HEAD", cwd=source)


def source_spec(source: Path, revision: str, python: str | None = None) -> ImplementationSpec:
    return ImplementationSpec(
        method_id="fixture",
        display_name="Fixture",
        source_kind="external",
        implementation_category="test-fixture",
        repository_url=str(source),
        revision=revision,
        python=python or f"{sys.version_info.major}.{sys.version_info.minor}",
        adapter_command=("{python}", "adapter.py"),
        default_hyperparameters={},
        fallback_ids=(),
    )


def test_setup_checks_out_detached_commit_and_reuses_verified_environment(
    tmp_path: Path, monkeypatch
) -> None:
    source, revision = make_source_repository(tmp_path)
    spec = source_spec(source, revision)
    monkeypatch.setattr(setup_external, "load_registry", lambda path: {"fixture": spec})
    root = tmp_path / "official"

    first = prepare_source("fixture", root)
    second = prepare_source("fixture", root)

    assert first.status == "ready"
    assert first.reused is False
    assert git("rev-parse", "HEAD", cwd=first.source_dir) == revision
    assert git("branch", "--show-current", cwd=first.source_dir) == ""
    assert first.environment_dir.joinpath("bin", "python").is_file()
    assert second.status == "ready"
    assert second.reused is True


def test_setup_refuses_modified_source_checkout(tmp_path: Path, monkeypatch) -> None:
    source, revision = make_source_repository(tmp_path)
    spec = source_spec(source, revision)
    monkeypatch.setattr(setup_external, "load_registry", lambda path: {"fixture": spec})
    root = tmp_path / "official"
    prepared = prepare_source("fixture", root)
    prepared.source_dir.joinpath("README.md").write_text("modified\n", encoding="utf-8")

    verified = verify_source("fixture", root)

    assert verified.status == "incompatible"
    assert "modified" in verified.reason.lower()


def test_setup_reports_unavailable_python_as_incompatible(tmp_path: Path, monkeypatch) -> None:
    source, revision = make_source_repository(tmp_path)
    spec = source_spec(source, revision, python="python-version-that-does-not-exist")
    monkeypatch.setattr(setup_external, "load_registry", lambda path: {"fixture": spec})

    record = prepare_source("fixture", tmp_path / "official")

    assert record.status == "incompatible"
    assert record.reason


def test_package_inventory_does_not_require_pip_module(monkeypatch) -> None:
    calls = []

    def run(arguments, cwd=None):
        calls.append(arguments)
        return "numpy==1.0\ntorch==2.0"

    monkeypatch.setattr(setup_external, "_run", run)

    packages = setup_external._package_inventory(Path("environment/bin/python"))

    assert packages == ("numpy==1.0", "torch==2.0")
    assert calls[0][:2] == ["environment/bin/python", "-c"]
    assert "pip" not in calls[0]


def test_environment_command_clears_incomplete_generated_directory() -> None:
    environment = Path("official/.envs/method")

    current = setup_external._environment_command("3.9", environment, "3.9")
    alternate = setup_external._environment_command("3.8", environment, "3.9")

    assert current == [sys.executable, "-m", "venv", "--clear", str(environment)]
    assert alternate == [
        "uv",
        "venv",
        str(environment),
        "--python",
        "3.8",
        "--clear",
    ]


def test_setup_command_error_retains_dependency_diagnostic() -> None:
    with pytest.raises(RuntimeError, match="dependency-conflict-marker"):
        setup_external._run(
            [
                sys.executable,
                "-c",
                "import sys; print('dependency-conflict-marker', file=sys.stderr); sys.exit(2)",
            ]
        )


def test_anomaly_transformer_cpu_compatibility_replaces_cuda_transfer() -> None:
    class Tensor:
        def cuda(self):
            return "cuda"

    class Cuda:
        @staticmethod
        def is_available():
            return False

    Torch = type("Torch", (), {"cuda": Cuda(), "Tensor": Tensor})

    changed = anomaly_transformer._enable_cpu_compatibility(Torch)
    value = Tensor()

    assert changed is True
    assert value.cuda() is value


def test_tranad_upstream_import_arguments_are_scoped(monkeypatch) -> None:
    original = ["adapter", "--input-dir", "private-input"]
    monkeypatch.setattr(sys, "argv", original.copy())

    with tranad._upstream_import_arguments():
        assert sys.argv == ["adapter", "--dataset", "synthetic", "--model", "TranAD"]

    assert sys.argv == original


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda module: module.METHOD_ID)
def test_adapters_fit_train_score_test_and_align_warmup(
    tmp_path: Path, adapter
) -> None:
    input_dir = tmp_path / adapter.METHOD_ID / "input"
    output_dir = tmp_path / adapter.METHOD_ID / "output"
    input_dir.mkdir(parents=True)
    train = np.arange(16, dtype=float).reshape(8, 2)
    test = np.arange(12, dtype=float).reshape(6, 2)
    np.save(input_dir / "train.npy", train)
    np.save(input_dir / "test.npy", test)
    (input_dir / "metadata.json").write_text(
        json.dumps({"schema_version": "official-baseline-v1"}), encoding="utf-8"
    )
    observed: dict[str, object] = {}

    def backend(train_values, test_values, configuration, seed):
        observed["train"] = train_values.copy()
        observed["test"] = test_values.copy()
        observed["configuration"] = dict(configuration)
        observed["seed"] = seed
        return BackendOutput(
            scores=np.arange(len(test_values) - 2, dtype=float),
            warmup_rows=2,
            score_direction="larger-is-more-normal",
            metadata={"backend": "stub"},
        )

    adapter.run_adapter(
        input_dir=input_dir,
        output_dir=output_dir,
        seed=39,
        configuration={"epochs": 2},
        backend=backend,
    )

    np.testing.assert_array_equal(observed["train"], train)
    np.testing.assert_array_equal(observed["test"], test)
    assert observed["configuration"] == {"epochs": 2}
    assert observed["seed"] == 39
    np.testing.assert_array_equal(
        np.load(output_dir / "scores.npy"), np.array([-0.0, -0.0, -0.0, -1.0, -2.0, -3.0])
    )
    record = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["warmup_rows"] == 2
    assert record["warmup_alignment"] == "repeat-first-valid"
    assert record["score_direction"] == "larger-is-more-anomalous"
    assert record["backend"] == "stub"
