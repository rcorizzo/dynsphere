"""Filesystem contract shared by local and external baseline implementations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_VERSION = "official-baseline-v1"
RUN_STATUSES = {"pending", "running", "complete", "failed", "timeout", "incompatible"}


class OutputValidationError(ValueError):
    """Raised when an adapter output cannot be evaluated safely."""


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    dataset_id: str
    source_family: str
    trajectory_id: str
    train: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ModelInputBundle:
    case_root: Path
    input_dir: Path
    private_dir: Path
    train_path: Path
    test_path: Path
    metadata_path: Path
    labels_path: Path
    expected_rows: int
    data_digest: str


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
            temporary_path = Path(handle.name)
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _data_digest(train: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, values in (("train", train), ("test", test)):
        contiguous = np.ascontiguousarray(values)
        digest.update(name.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validated_case_arrays(case: EvaluationCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(case.train)
    test = np.asarray(case.test)
    labels = np.asarray(case.labels)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("Training and test observations must be two-dimensional")
    if train.shape[1] != test.shape[1]:
        raise ValueError("Training and test feature counts must match")
    if labels.ndim != 1 or len(labels) != len(test):
        raise ValueError("Labels must be one-dimensional and align with test rows")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Training and test observations must be finite")
    return train, test, labels


def write_evaluation_case(case: EvaluationCase, root: Path) -> ModelInputBundle:
    """Write private evaluation data and the label-free public model input."""
    train, test, labels = _validated_case_arrays(case)
    serialized_metadata = json.dumps(dict(case.metadata), sort_keys=True).lower()
    if "label" in serialized_metadata or "private" in serialized_metadata:
        raise ValueError("Public metadata cannot expose labels or private paths")

    input_dir = root / "model_input"
    private_dir = root / "private"
    train_path = input_dir / "train.npy"
    test_path = input_dir / "test.npy"
    metadata_path = input_dir / "metadata.json"
    labels_path = private_dir / "labels.npy"
    digest = _data_digest(train, test)
    public_metadata = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "dataset_id": case.dataset_id,
        "source_family": case.source_family,
        "trajectory_id": case.trajectory_id,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(train.shape[1]),
        "data_digest": digest,
        **dict(case.metadata),
    }
    _atomic_save_array(train_path, train)
    _atomic_save_array(test_path, test)
    _atomic_save_json(metadata_path, public_metadata)
    _atomic_save_array(labels_path, labels)
    return ModelInputBundle(
        case_root=root,
        input_dir=input_dir,
        private_dir=private_dir,
        train_path=train_path,
        test_path=test_path,
        metadata_path=metadata_path,
        labels_path=labels_path,
        expected_rows=len(test),
        data_digest=digest,
    )


def validate_scores(path: Path, expected_rows: int) -> np.ndarray:
    """Load an adapter score vector after strict shape and finiteness checks."""
    if not path.is_file():
        raise OutputValidationError(f"Score file is missing: {path}")
    try:
        scores = np.asarray(np.load(path, allow_pickle=False))
    except Exception as exc:
        raise OutputValidationError(f"Score file cannot be loaded: {path}") from exc
    if scores.ndim != 1:
        raise OutputValidationError("Scores must be a one-dimensional array")
    if len(scores) != expected_rows:
        raise OutputValidationError(
            f"Expected {expected_rows} scores and received {len(scores)}"
        )
    if not np.isfinite(scores).all():
        raise OutputValidationError("Scores contain nonfinite values")
    return scores.astype(float, copy=False)


def write_run_record(path: Path, record: Mapping[str, object]) -> None:
    """Write a normalized run record atomically."""
    status = record.get("status")
    if status not in RUN_STATUSES:
        raise ValueError(f"Unsupported run status: {status!r}")
    payload = {"schema_version": SCHEMA_VERSION, **dict(record)}
    _atomic_save_json(path, payload)
