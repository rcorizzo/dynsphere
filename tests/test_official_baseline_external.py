from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.official_baselines import external
from benchmarks.official_baselines.contract import EvaluationCase, write_evaluation_case
from benchmarks.official_baselines.external import run_external
from benchmarks.official_baselines.provenance import ImplementationSpec


def make_bundle(tmp_path: Path):
    case = EvaluationCase(
        case_id="fixture/one",
        dataset_id="fixture",
        source_family="fixture",
        trajectory_id="one",
        train=np.arange(12, dtype=float).reshape(6, 2),
        test=np.arange(10, dtype=float).reshape(5, 2),
        labels=np.array([0, 0, 1, 0, 1]),
        metadata={"scaling_rule": "standard-train-only-v1"},
    )
    return write_evaluation_case(case, tmp_path / "case")


def fake_spec(mode: str = "success", method_id: str = "fake") -> ImplementationSpec:
    return ImplementationSpec(
        method_id=method_id,
        display_name="Fake adapter",
        source_kind="external",
        implementation_category="test-fixture",
        repository_url="https://example.invalid/fake.git",
        revision="a" * 40,
        python="3.9",
        adapter_command=(
            sys.executable,
            "-m",
            "benchmarks.official_baselines.adapters.fake_adapter",
            "--mode",
            mode,
        ),
        default_hyperparameters={"constant": 1},
        fallback_ids=(),
    )


def test_fake_adapter_receives_only_public_input_and_returns_scores(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)

    result = run_external(
        fake_spec(), bundle, tmp_path / "output", seed=37, timeout_seconds=5
    )

    assert result.status == "complete"
    np.testing.assert_array_equal(np.load(result.scores_path), np.arange(5) + 37)
    received = json.loads(
        (tmp_path / "output" / "adapter_output" / "received.json").read_text(
            encoding="utf-8"
        )
    )
    assert received["files"] == ["metadata.json", "test.npy", "train.npy"]
    assert received["environment"]["DGLBACKEND"] == "pytorch"
    assert received["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    for name in ("DGLDEFAULTDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        assert Path(received["environment"][name]).is_relative_to(tmp_path / "output")
    serialized = json.dumps(received).lower()
    assert "labels.npy" not in serialized
    assert str(bundle.private_dir).lower() not in serialized
    record = json.loads((tmp_path / "output" / "run.json").read_text(encoding="utf-8"))
    assert record["requested_method_id"] == "fake"
    assert record["executed_method_id"] == "fake"


def test_repository_benchmarks_package_has_precedence() -> None:
    import benchmarks

    expected = Path(__file__).resolve().parents[1] / "benchmarks" / "__init__.py"
    assert Path(benchmarks.__file__).resolve() == expected


@pytest.mark.parametrize(
    "mode, expected_status",
    [
        ("nonzero", "failed"),
        ("missing", "failed"),
        ("truncated", "failed"),
        ("nan", "failed"),
        ("partial", "failed"),
    ],
)
def test_process_and_output_problems_produce_normalized_records(
    tmp_path: Path, mode: str, expected_status: str
) -> None:
    bundle = make_bundle(tmp_path)

    result = run_external(
        fake_spec(mode), bundle, tmp_path / mode, seed=37, timeout_seconds=5
    )

    assert result.status == expected_status
    record = json.loads((tmp_path / mode / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == expected_status
    assert record["error"]


def test_timeout_produces_timeout_record(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)

    result = run_external(
        fake_spec("timeout"), bundle, tmp_path / "timeout", seed=37, timeout_seconds=0.05
    )

    assert result.status == "timeout"
    record = json.loads((tmp_path / "timeout" / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "timeout"


def test_unavailable_preferred_source_is_incompatible_without_fallback(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(tmp_path)
    unavailable = fake_spec(method_id="preferred")
    unavailable = ImplementationSpec(
        **{
            **unavailable.__dict__,
            "adapter_command": (str(tmp_path / "missing-python"), "adapter.py"),
            "fallback_ids": ("fallback",),
        }
    )

    result = run_external(
        unavailable, bundle, tmp_path / "unavailable", seed=37, timeout_seconds=5
    )

    assert result.status == "incompatible"
    assert result.requested_method_id == "preferred"
    assert result.executed_method_id == "preferred"


def test_explicit_fallback_records_requested_and_executed_identities(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = make_bundle(tmp_path)
    unavailable = fake_spec(method_id="preferred")
    unavailable = ImplementationSpec(
        **{
            **unavailable.__dict__,
            "adapter_command": (str(tmp_path / "missing-python"), "adapter.py"),
            "fallback_ids": ("fallback",),
        }
    )
    fallback = fake_spec(method_id="fallback")
    monkeypatch.setattr(external, "lookup_fallback", lambda fallback_id: fallback)

    result = run_external(
        unavailable,
        bundle,
        tmp_path / "fallback-output",
        seed=37,
        timeout_seconds=5,
        fallback_id="fallback",
    )

    assert result.status == "complete"
    assert result.requested_method_id == "preferred"
    assert result.executed_method_id == "fallback"
    record = json.loads(
        (tmp_path / "fallback-output" / "run.json").read_text(encoding="utf-8")
    )
    assert record["requested_method_id"] == "preferred"
    assert record["executed_method_id"] == "fallback"
