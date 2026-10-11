"""Leakage-free dataset preparation for the official baseline evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from benchmarks.official_baselines.contract import EvaluationCase
from benchmarks.real_world_datasets.run_real_world_datasets import (
    features_and_labels,
    find_label,
    read_csv,
)
from benchmarks.tsb_ad_m.run_tsb_ad_m import (
    declared_prefix_indices,
    load_tsb_csv,
    source_name,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SMD22 = "078_SMD_id_22_Facility_tr_500_1st_326.csv"
SCALING_RULE = "standard-train-only-v1"
CONTEXTUAL_TRAIN = REPO_ROOT / "data" / "contextual" / "pendulum_contextual_train.csv"
CONTEXTUAL_TEST = REPO_ROOT / "data" / "contextual" / "pendulum_contextual_test.csv"
REAL_WORLD_SPECS = {
    "BATADAL": {
        "train": REPO_ROOT
        / "benchmarks"
        / "real_world_datasets"
        / "data"
        / "batadal"
        / "BATADAL_dataset03.csv",
        "test": REPO_ROOT
        / "benchmarks"
        / "real_world_datasets"
        / "data"
        / "batadal"
        / "BATADAL_dataset04.csv",
        "train_field": "ATT_FLAG",
        "test_field": "ATT_FLAG",
    },
    "HAI-20.07": {
        "train": REPO_ROOT
        / "benchmarks"
        / "real_world_datasets"
        / "data"
        / "hai_20_07"
        / "train1.csv.gz",
        "test": REPO_ROOT
        / "benchmarks"
        / "real_world_datasets"
        / "data"
        / "hai_20_07"
        / "test1.csv.gz",
        "train_field": "attack",
        "test_field": "attack",
    },
    "PowerSystem": {
        "train": REPO_ROOT
        / "benchmarks"
        / "real_world_datasets"
        / "data"
        / "power_system"
        / "binary"
        / "combined_binary.csv",
        "test": REPO_ROOT
        / "benchmarks"
        / "real_world_datasets"
        / "data"
        / "power_system"
        / "binary"
        / "combined_binary.csv",
        "train_field": "marker",
        "test_field": "marker",
    },
}


def read_eval_manifest(path: Path) -> tuple[str, ...]:
    """Read the exact 180-row official multivariate evaluation manifest."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["file_name"]:
            raise ValueError("Evaluation manifest must have one file_name column")
        filenames = tuple((row.get("file_name") or "").strip() for row in reader)
    if len(filenames) != 180:
        raise ValueError(f"Evaluation manifest must contain 180 rows; found {len(filenames)}")
    if any(not name or Path(name).name != name for name in filenames):
        raise ValueError("Evaluation manifest entries must be nonempty filenames")
    if len(set(filenames)) != len(filenames):
        raise ValueError("Evaluation manifest contains duplicate filenames")
    return filenames


def training_indices_for_tsb(
    path: Path, labels: np.ndarray, min_train: int = 16
) -> tuple[np.ndarray, str]:
    """Return verified normal-prefix indices and their declared protocol."""
    labels = np.asarray(labels, dtype=int)
    try:
        return declared_prefix_indices(path, labels, min_train), "declared-prefix"
    except ValueError:
        if path.name != SMD22:
            raise
        positives = np.flatnonzero(labels == 1)
        if len(positives) == 0 or int(positives[0]) < min_train:
            raise ValueError("SMD-22 lacks the documented maximal clean prefix")
        return np.arange(int(positives[0])), "maximal-clean-prefix"


def fit_training_scaler(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit standardization only on training observations."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Training and test arrays must have matching feature dimensions")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    metadata: dict[str, object] = {
        "scaling_rule": SCALING_RULE,
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
    }
    return train_scaled, test_scaled, metadata


def load_contextual_case(seed: int) -> EvaluationCase:
    """Load the released contextual benchmark with normal-only training."""
    train_frame = pd.read_csv(CONTEXTUAL_TRAIN).sort_values(["entity_id", "time"])
    test_frame = pd.read_csv(CONTEXTUAL_TEST).sort_values(["entity_id", "time"])
    features = [column for column in train_frame.columns if column.startswith("x_")]
    train, test, scaling = fit_training_scaler(
        train_frame[features].to_numpy(dtype=float),
        test_frame[features].to_numpy(dtype=float),
    )
    return EvaluationCase(
        case_id="DynSphere-Contextual/complete",
        dataset_id="DynSphere-Contextual",
        source_family="DynSphere-Contextual",
        trajectory_id="complete",
        train=train,
        test=test,
        labels=test_frame["is_anomaly"].to_numpy(dtype=int),
        metadata={
            **scaling,
            "seed": seed,
            "training_protocol": "released-normal-training",
            "row_id_rule": "entity-time-sorted",
            "feature_names": features,
        },
    )


def load_real_world_case(dataset_id: str, seed: int) -> EvaluationCase:
    """Load a real-world dataset using its existing normal training portion."""
    try:
        spec = REAL_WORLD_SPECS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unknown real-world dataset: {dataset_id}") from exc
    train_path = Path(spec["train"])
    test_path = Path(spec["test"])
    train_frame = read_csv(train_path)
    test_frame = read_csv(test_path)
    train_field = find_label(train_frame, str(spec["train_field"]))
    test_field = find_label(test_frame, str(spec["test_field"]))
    train_features, train_states = features_and_labels(
        train_frame, train_field, default_normal=True
    )
    test_features, test_states = features_and_labels(
        test_frame, test_field, default_normal=False
    )
    common = [column for column in train_features.columns if column in test_features.columns]
    if not common:
        raise ValueError(f"No common numeric features for {dataset_id}")
    normal = np.flatnonzero(train_states == 0)
    if len(normal) == 0:
        raise ValueError(f"No normal training rows for {dataset_id}")
    train, test, scaling = fit_training_scaler(
        train_features.iloc[normal][common].to_numpy(dtype=float),
        test_features[common].to_numpy(dtype=float),
    )
    return EvaluationCase(
        case_id=f"{dataset_id}/complete",
        dataset_id=dataset_id,
        source_family=dataset_id,
        trajectory_id="complete",
        train=train,
        test=test,
        labels=np.asarray(test_states, dtype=int),
        metadata={
            **scaling,
            "seed": seed,
            "training_protocol": "normal-training-portion",
            "row_id_rule": "source-row-order",
            "feature_names": common,
            "train_source": str(train_path.relative_to(REPO_ROOT)),
            "test_source": str(test_path.relative_to(REPO_ROOT)),
        },
    )


def load_tsb_cases(
    data_root: Path, manifest_path: Path, seed: int
) -> Iterator[EvaluationCase]:
    """Yield all official TSB-AD-M evaluation trajectories in manifest order."""
    filenames = read_eval_manifest(manifest_path)
    available = {path.name: path for path in data_root.rglob("*.csv") if path.is_file()}
    missing = [name for name in filenames if name not in available]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest trajectories are missing from {data_root}: {missing[:5]}"
        )
    for filename in filenames:
        path = available[filename]
        observations, labels = load_tsb_csv(path)
        indices, protocol = training_indices_for_tsb(path, labels)
        train, test, scaling = fit_training_scaler(observations[indices], observations)
        family = source_name(path)
        yield EvaluationCase(
            case_id=f"TSB-AD-M/{path.stem}",
            dataset_id="TSB-AD-M",
            source_family=family,
            trajectory_id=path.stem,
            train=train,
            test=test,
            labels=labels.astype(int),
            metadata={
                **scaling,
                "seed": seed,
                "training_protocol": protocol,
                "declared_prefix_length": int(len(indices)),
                "row_id_start": 0,
                "row_id_stop": int(len(labels)),
                "source_file": filename,
            },
        )
