from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.official_baselines.contract import (
    EvaluationCase,
    OutputValidationError,
    validate_scores,
    write_evaluation_case,
    write_run_record,
)


def fixture_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="fixture/trajectory-1",
        dataset_id="fixture",
        source_family="fixture-family",
        trajectory_id="trajectory-1",
        train=np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        test=np.array(
            [[3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0]]
        ),
        labels=np.array([0, 0, 1, 1, 0, 0]),
        metadata={"scaling_rule": "standard-train-only-v1", "declared_prefix_length": 3},
    )


def test_public_bundle_excludes_labels_and_private_paths(tmp_path: Path) -> None:
    bundle = write_evaluation_case(fixture_case(), tmp_path / "case")

    assert bundle.labels_path == tmp_path / "case" / "private" / "labels.npy"
    assert bundle.labels_path.exists()
    assert {path.name for path in bundle.input_dir.iterdir()} == {
        "train.npy",
        "test.npy",
        "metadata.json",
    }
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    serialized = json.dumps(metadata).lower()
    assert "label" not in serialized
    assert "private" not in serialized
    assert metadata["schema_version"] == "official-baseline-v1"
    np.testing.assert_array_equal(np.load(bundle.train_path), fixture_case().train)
    np.testing.assert_array_equal(np.load(bundle.test_path), fixture_case().test)


def test_exact_length_finite_scores_are_accepted(tmp_path: Path) -> None:
    path = tmp_path / "scores.npy"
    np.save(path, np.array([0.2, 0.7, 0.1]))

    scores = validate_scores(path, expected_rows=3)

    np.testing.assert_array_equal(scores, np.array([0.2, 0.7, 0.1]))


@pytest.mark.parametrize(
    "scores, expected_rows",
    [
        (np.array([0.1, 0.2]), 3),
        (np.array([0.1, 0.2, 0.3, 0.4]), 3),
        (np.array([[0.1], [0.2], [0.3]]), 3),
        (np.array([0.1, np.nan, 0.3]), 3),
        (np.array([0.1, np.inf, 0.3]), 3),
    ],
)
def test_invalid_score_arrays_are_rejected(
    tmp_path: Path, scores: np.ndarray, expected_rows: int
) -> None:
    path = tmp_path / "scores.npy"
    np.save(path, scores)

    with pytest.raises(OutputValidationError):
        validate_scores(path, expected_rows=expected_rows)


def test_missing_score_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OutputValidationError):
        validate_scores(tmp_path / "scores.npy", expected_rows=3)


def test_run_record_accepts_only_normalized_statuses(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    write_run_record(path, {"status": "complete", "method_id": "fixture"})

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema_version"] == "official-baseline-v1"
    assert record["status"] == "complete"

    with pytest.raises(ValueError):
        write_run_record(path, {"status": "successful"})
