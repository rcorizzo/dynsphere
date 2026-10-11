from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.official_baselines.datasets import (
    fit_training_scaler,
    load_tsb_cases,
    read_eval_manifest,
    training_indices_for_tsb,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "benchmarks" / "official_baselines" / "manifests"
MANIFEST = MANIFEST_DIR / "TSB-AD-M-Eva.csv"
PROVENANCE = MANIFEST_DIR / "provenance.json"
SMD22 = "078_SMD_id_22_Facility_tr_500_1st_326.csv"


def write_manifest(path: Path, filenames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name"])
        writer.writerows([[name] for name in filenames])


def test_official_manifest_has_180_unique_rows_and_verified_digest() -> None:
    filenames = read_eval_manifest(MANIFEST)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    assert len(filenames) == 180
    assert len(set(filenames)) == 180
    assert filenames == tuple(dict.fromkeys(filenames))
    assert provenance["row_count"] == 180
    assert provenance["sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert provenance["source_commit"] == "ed74348e7876c9268b7a737c77f2f5f56f7fa458"


def test_manifest_rejects_duplicate_or_extra_rows(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    write_manifest(duplicate, ["same.csv"] * 180)
    with pytest.raises(ValueError, match="duplicate"):
        read_eval_manifest(duplicate)

    extra = tmp_path / "extra.csv"
    write_manifest(extra, [f"{index:03d}_source.csv" for index in range(181)])
    with pytest.raises(ValueError, match="180"):
        read_eval_manifest(extra)


def test_manifest_rejects_an_unrecognized_header(tmp_path: Path) -> None:
    path = tmp_path / "bad-header.csv"
    path.write_text("path\nfile.csv\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file_name"):
        read_eval_manifest(path)


def test_tsb_loader_rejects_manifest_files_missing_from_data_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest trajectories"):
        tuple(load_tsb_cases(tmp_path, MANIFEST, seed=37))


def test_declared_prefix_is_used_when_it_is_clean() -> None:
    path = Path("002_MSL_id_1_Sensor_tr_5_1st_9.csv")
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0])

    indices, protocol = training_indices_for_tsb(path, labels, min_train=2)

    np.testing.assert_array_equal(indices, np.arange(5))
    assert protocol == "declared-prefix"


def test_contaminated_ordinary_declared_prefix_is_rejected() -> None:
    path = Path("002_MSL_id_1_Sensor_tr_5_1st_9.csv")
    labels = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0])

    with pytest.raises(ValueError, match="contains labeled anomalies"):
        training_indices_for_tsb(path, labels, min_train=2)


def test_named_smd22_uses_only_its_maximal_clean_prefix() -> None:
    labels = np.zeros(600, dtype=int)
    labels[326:330] = 1

    indices, protocol = training_indices_for_tsb(Path(SMD22), labels, min_train=16)

    np.testing.assert_array_equal(indices, np.arange(326))
    assert protocol == "maximal-clean-prefix"


def test_training_scaler_never_uses_test_statistics() -> None:
    train = np.array([[0.0], [1.0], [2.0]])
    test = np.array([[100.0], [101.0], [102.0]])

    train_scaled, test_scaled, metadata = fit_training_scaler(train, test)

    assert metadata["mean"] == [1.0]
    assert abs(float(train_scaled.mean())) < 1e-12
    assert float(test_scaled.mean()) > 100.0
