from __future__ import annotations

import json
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.official_baselines.provenance import (
    ImplementationSpec,
    load_registry,
    resolve_local_revision,
    run_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "benchmarks" / "official_baselines" / "registry.json"
REQUIRED_METHODS = {
    "pca_reconstruction",
    "kmeans_distance",
    "dagmm",
    "catch",
    "mtad_gat",
    "omni_anomaly",
    "anomaly_transformer",
    "tranad",
    "predictive_euclidean_single",
    "dynsphere_one_regime",
}


def test_registry_has_all_required_methods_with_resolved_revisions() -> None:
    specs = load_registry(REGISTRY)

    assert REQUIRED_METHODS <= specs.keys()
    assert all(re.fullmatch(r"[0-9a-f]{40}", spec.revision) for spec in specs.values())
    assert all(isinstance(spec.adapter_command, tuple) for spec in specs.values())
    assert all(spec.adapter_command for spec in specs.values())

    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    local_revisions = {spec.revision for spec in specs.values() if spec.source_kind == "local"}
    assert local_revisions == {current}
    assert resolve_local_revision(REPO_ROOT) == current


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("display_name"),
        lambda row: row.update({"revision": "main"}),
        lambda row: row.update({"adapter_command": "python adapter.py"}),
        lambda row: row.update({"source_kind": "unknown"}),
    ],
)
def test_registry_rejects_incomplete_or_mutable_entries(
    tmp_path: Path, mutation
) -> None:
    row = {
        "method_id": "external_method",
        "display_name": "External Method",
        "source_kind": "external",
        "implementation_category": "author-released",
        "repository_url": "https://example.invalid/repository.git",
        "revision": "a" * 40,
        "python": "3.10",
        "adapter_command": ["python", "-m", "adapter"],
        "default_hyperparameters": {},
        "fallback_ids": [],
    }
    mutation(row)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"implementations": [row]}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_registry(path)


def test_fallbacks_are_explicit_distinct_registry_entries() -> None:
    specs = load_registry(REGISTRY)

    for spec in specs.values():
        for fallback_id in spec.fallback_ids:
            assert fallback_id in specs
            assert fallback_id != spec.method_id
            assert specs[fallback_id].revision != ""


def test_catch_uses_fixed_batch_size_valid_for_declared_training_prefixes() -> None:
    specs = load_registry(REGISTRY)

    assert specs["catch"].default_hyperparameters["batch_size"] == 32


def test_adapter_runtime_constraints_are_part_of_provenance() -> None:
    specs = load_registry(REGISTRY)

    assert specs["dagmm"].python == "3.9"
    assert specs["dagmm"].install_upstream_requirements is False
    assert specs["dagmm"].fallback_ids == ()
    assert specs["dagmm"].runtime_requirements == (
        "tensorflow==2.13.1",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        "joblib",
        "pyod",
    )
    assert specs["anomaly_transformer"].runtime_requirements == (
        "torch==2.4.1",
        "numpy==1.24.4",
    )
    assert specs["tranad"].runtime_requirements == (
        "torch==1.13.1",
        "numpy==1.26.4",
        "dgl==1.1.3",
    )


def test_run_identity_changes_for_every_reproducibility_input() -> None:
    base_spec = ImplementationSpec(
        method_id="external_method",
        display_name="External Method",
        source_kind="external",
        implementation_category="author-released",
        repository_url="https://example.invalid/repository.git",
        revision="a" * 40,
        python="3.10",
        adapter_command=("python", "-m", "adapter"),
        default_hyperparameters={"window": 16},
        fallback_ids=(),
    )
    base = run_identity(
        case_identity="dataset/trajectory",
        implementation=base_spec,
        hyperparameters={"window": 16},
        seed=37,
        data_digest="data-a",
        scaling_rule="standard-train-only-v1",
    )
    alternatives = [
        {"implementation": replace(base_spec, revision="b" * 40)},
        {"implementation": replace(base_spec, runtime_requirements=("torch==1.0",))},
        {"implementation": replace(base_spec, install_upstream_requirements=False)},
        {"hyperparameters": {"window": 32}},
        {"seed": 38},
        {"data_digest": "data-b"},
        {"scaling_rule": "robust-train-only-v1"},
    ]

    for change in alternatives:
        arguments = {
            "case_identity": "dataset/trajectory",
            "implementation": base_spec,
            "hyperparameters": {"window": 16},
            "seed": 37,
            "data_digest": "data-a",
            "scaling_rule": "standard-train-only-v1",
        }
        arguments.update(change)
        assert run_identity(**arguments) != base
