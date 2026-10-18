"""Resumable execution matrix for source-traceable baseline comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from benchmarks.official_baselines.contract import (
    EvaluationCase,
    OutputValidationError,
    validate_scores,
    write_evaluation_case,
    write_run_record,
)
from benchmarks.official_baselines.datasets import (
    REAL_WORLD_SPECS,
    SCALING_RULE,
    fit_training_scaler,
    load_contextual_case,
    load_real_world_case,
    load_tsb_csv,
    read_eval_manifest,
    source_name,
    training_indices_for_tsb,
)
from benchmarks.official_baselines.external import run_external
from benchmarks.official_baselines.local_controls import LOCAL_SCORERS
from benchmarks.official_baselines.provenance import (
    ImplementationSpec,
    load_registry,
    run_identity,
)
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_REGISTRY = PACKAGE_ROOT / "registry.json"
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifests" / "TSB-AD-M-Eva.csv"
DEFAULT_TSB_ROOT = REPO_ROOT / "benchmarks" / "tsb_ad_m" / "data" / "TSB-AD-M"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "official_baselines_v1"
CURRENT_DATASETS = (
    "DynSphere-Contextual",
    "MSL",
    "SMAP",
    "SMD",
    "BATADAL",
    "HAI-20.07",
    "PowerSystem",
)
TSB_CURRENT_FAMILIES = {"MSL", "SMAP", "SMD"}
DEFAULT_METHODS = (
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
)


@dataclass(frozen=True)
class RunConfig:
    methods: tuple[str, ...] = DEFAULT_METHODS
    datasets: tuple[str, ...] = CURRENT_DATASETS + ("TSB-AD-M",)
    seeds: tuple[int, ...] = (37, 38, 39)
    tsb_manifest: Path = DEFAULT_MANIFEST
    tsb_data_root: Path = DEFAULT_TSB_ROOT
    result_root: Path = DEFAULT_RESULT_ROOT
    registry_path: Path = DEFAULT_REGISTRY
    timeout_seconds: int = 3600
    resume: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class CaseDescriptor:
    case_id: str
    dataset_id: str
    source_family: str
    trajectory_id: str
    seed: int
    loader: Callable[[], EvaluationCase]


@dataclass(frozen=True)
class ExperimentJob:
    case: CaseDescriptor
    implementation: ImplementationSpec
    hyperparameters: Mapping[str, object]
    result_root: Path
    timeout_seconds: int
    resume: bool


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _data_digest(train: np.ndarray, test: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, values in (("train", train), ("test", test)):
        contiguous = np.ascontiguousarray(values)
        digest.update(name.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_single_tsb(path: Path, seed: int) -> EvaluationCase:
    observations, labels = load_tsb_csv(path)
    indices, protocol = training_indices_for_tsb(path, labels)
    train, test, scaling = fit_training_scaler(observations[indices], observations)
    family = source_name(path)
    return EvaluationCase(
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
            "source_file": path.name,
        },
    )


def _tsb_paths(data_root: Path, manifest_path: Path) -> tuple[Path, ...]:
    names = read_eval_manifest(manifest_path)
    discovered: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in data_root.rglob("*.csv"):
        if path.name in discovered:
            duplicates.add(path.name)
        discovered[path.name] = path
    if duplicates:
        raise ValueError(f"Duplicate TSB filenames under {data_root}: {sorted(duplicates)[:5]}")
    missing = [name for name in names if name not in discovered]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest trajectories are missing from {data_root}: {missing[:5]}"
        )
    return tuple(discovered[name] for name in names)


def _case_descriptors(config: RunConfig) -> tuple[CaseDescriptor, ...]:
    unknown = set(config.datasets).difference(CURRENT_DATASETS + ("TSB-AD-M",))
    if unknown:
        raise ValueError(f"Unknown dataset selections: {sorted(unknown)}")
    if not config.seeds:
        raise ValueError("At least one seed is required")

    descriptors: dict[tuple[str, int], CaseDescriptor] = {}

    def add(entry: CaseDescriptor) -> None:
        descriptors.setdefault((entry.case_id, entry.seed), entry)

    if "DynSphere-Contextual" in config.datasets:
        for seed in config.seeds:
            add(
                CaseDescriptor(
                    case_id="DynSphere-Contextual/complete",
                    dataset_id="DynSphere-Contextual",
                    source_family="DynSphere-Contextual",
                    trajectory_id="complete",
                    seed=seed,
                    loader=lambda seed=seed: load_contextual_case(seed),
                )
            )

    for dataset_id in REAL_WORLD_SPECS:
        if dataset_id not in config.datasets:
            continue
        for seed in config.seeds:
            add(
                CaseDescriptor(
                    case_id=f"{dataset_id}/complete",
                    dataset_id=dataset_id,
                    source_family=dataset_id,
                    trajectory_id="complete",
                    seed=seed,
                    loader=lambda dataset_id=dataset_id, seed=seed: load_real_world_case(
                        dataset_id, seed
                    ),
                )
            )

    needs_tsb = "TSB-AD-M" in config.datasets or bool(
        TSB_CURRENT_FAMILIES.intersection(config.datasets)
    )
    if needs_tsb:
        for path in _tsb_paths(config.tsb_data_root, config.tsb_manifest):
            family = source_name(path)
            seeds: set[int] = set()
            if family in config.datasets and family in TSB_CURRENT_FAMILIES:
                seeds.update(config.seeds)
            if "TSB-AD-M" in config.datasets:
                seeds.add(37)
            for seed in sorted(seeds):
                add(
                    CaseDescriptor(
                        case_id=f"TSB-AD-M/{path.stem}",
                        dataset_id="TSB-AD-M",
                        source_family=family,
                        trajectory_id=path.stem,
                        seed=seed,
                        loader=lambda path=path, seed=seed: _load_single_tsb(path, seed),
                    )
                )

    return tuple(
        sorted(
            descriptors.values(),
            key=lambda entry: (entry.case_id, entry.seed),
        )
    )


def build_matrix(config: RunConfig) -> tuple[ExperimentJob, ...]:
    """Construct a deterministic, deduplicated, lazily loaded experiment matrix."""
    registry = load_registry(config.registry_path)
    missing = [method_id for method_id in config.methods if method_id not in registry]
    if missing:
        raise ValueError(f"Unknown method selections: {missing}")
    cases = _case_descriptors(config)
    jobs = [
        ExperimentJob(
            case=case,
            implementation=registry[method_id],
            hyperparameters=dict(registry[method_id].default_hyperparameters),
            result_root=config.result_root,
            timeout_seconds=config.timeout_seconds,
            resume=config.resume,
        )
        for case in cases
        for method_id in config.methods
    ]
    return tuple(
        sorted(
            jobs,
            key=lambda entry: (
                entry.case.case_id,
                entry.case.seed,
                entry.implementation.method_id,
            ),
        )
    )


def _validated_case(job: ExperimentJob) -> EvaluationCase:
    value = job.case.loader()
    observed = (
        value.case_id,
        value.dataset_id,
        value.source_family,
        value.trajectory_id,
        int(value.metadata.get("seed", -1)),
    )
    expected = (
        job.case.case_id,
        job.case.dataset_id,
        job.case.source_family,
        job.case.trajectory_id,
        job.case.seed,
    )
    if observed != expected:
        raise ValueError(f"Loaded case descriptor mismatch: expected {expected}, received {observed}")
    return value


def _run_directory(job: ExperimentJob, identity: str) -> Path:
    return (
        job.result_root
        / "runs"
        / _slug(job.case.dataset_id)
        / _slug(job.case.source_family)
        / _slug(job.case.trajectory_id)
        / _slug(job.implementation.method_id)
        / f"seed-{job.case.seed}"
        / identity
    )


def _resume_record(run_dir: Path, identity: str, expected_rows: int) -> dict[str, object] | None:
    record_path = run_dir / "run.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or record.get("run_identity") != identity:
            return None
        validate_scores(run_dir / "execution" / "scores.npy", expected_rows)
    except (OSError, ValueError, json.JSONDecodeError, OutputValidationError):
        return None
    record["skipped"] = True
    return record


def execute_job(job: ExperimentJob) -> dict[str, object]:
    """Execute one job and resume only an exact, validated complete identity."""
    started = time.perf_counter()
    try:
        value = _validated_case(job)
    except Exception as exc:
        identity = run_identity(
            job.case.case_id,
            job.implementation,
            job.hyperparameters,
            job.case.seed,
            "unavailable",
            "unavailable",
        )
        run_dir = _run_directory(job, identity)
        terminal: dict[str, object] = {
            "run_identity": identity,
            "case_id": job.case.case_id,
            "dataset_id": job.case.dataset_id,
            "source_family": job.case.source_family,
            "trajectory_id": job.case.trajectory_id,
            "seed": job.case.seed,
            "method_id": job.implementation.method_id,
            "implementation_revision": job.implementation.revision,
            "adapter_revision": job.implementation.adapter_revision,
            "implementation_category": job.implementation.implementation_category,
            "hyperparameters": dict(job.hyperparameters),
            "data_digest": "unavailable",
            "scaling_rule": "unavailable",
            "run_directory": str(run_dir),
            "skipped": False,
            "status": "incompatible",
            "incompatibility_stage": "dataset_preparation",
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_run_record(run_dir / "run.json", terminal)
        return terminal
    scaling_rule = str(value.metadata.get("scaling_rule", ""))
    if not scaling_rule:
        raise ValueError("Evaluation case metadata must identify its scaling rule")
    digest = _data_digest(value.train, value.test)
    identity = run_identity(
        value.case_id,
        job.implementation,
        job.hyperparameters,
        job.case.seed,
        digest,
        scaling_rule,
    )
    run_dir = _run_directory(job, identity)
    if job.resume:
        resumed = _resume_record(run_dir, identity, len(value.test))
        if resumed is not None:
            return resumed

    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = write_evaluation_case(value, run_dir / "case")
    if bundle.data_digest != digest:
        raise RuntimeError("Prepared input digest differs from the run identity")
    execution_dir = run_dir / "execution"
    base_record: dict[str, object] = {
        "run_identity": identity,
        "case_id": value.case_id,
        "dataset_id": value.dataset_id,
        "source_family": value.source_family,
        "trajectory_id": value.trajectory_id,
        "seed": job.case.seed,
        "method_id": job.implementation.method_id,
        "implementation_revision": job.implementation.revision,
        "adapter_revision": job.implementation.adapter_revision,
        "implementation_category": job.implementation.implementation_category,
        "hyperparameters": dict(job.hyperparameters),
        "data_digest": digest,
        "scaling_rule": scaling_rule,
        "run_directory": str(run_dir),
        "skipped": False,
    }
    write_run_record(run_dir / "run.json", {**base_record, "status": "running"})
    try:
        if job.implementation.source_kind == "local":
            scorer = LOCAL_SCORERS[job.implementation.method_id]
            scores, scorer_metadata = scorer(value.train, value.test, job.case.seed)
            execution_dir.mkdir(parents=True, exist_ok=True)
            np.save(execution_dir / "scores.npy", np.asarray(scores), allow_pickle=False)
            scores = validate_scores(execution_dir / "scores.npy", len(value.test))
            write_run_record(
                execution_dir / "run.json",
                {
                    "status": "complete",
                    "method_id": job.implementation.method_id,
                    "seed": job.case.seed,
                    "score_direction": "larger-is-more-anomalous",
                    "scorer_metadata": dict(scorer_metadata),
                },
            )
            executed_method = job.implementation.method_id
            execution_metadata: Mapping[str, object] = dict(scorer_metadata)
        else:
            external = run_external(
                job.implementation,
                bundle,
                execution_dir,
                job.case.seed,
                job.timeout_seconds,
            )
            if external.status != "complete":
                external_record = json.loads(external.run_record_path.read_text(encoding="utf-8"))
                terminal = {
                    **base_record,
                    "status": external.status,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": external_record.get("error", "External execution did not complete"),
                    "executed_method_id": external.executed_method_id,
                }
                write_run_record(run_dir / "run.json", terminal)
                return terminal
            scores = validate_scores(external.scores_path, len(value.test))
            executed_method = external.executed_method_id
            execution_metadata = {}

        terminal = {
            **base_record,
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            "score_rows": int(len(scores)),
            "executed_method_id": executed_method,
            "execution_metadata": dict(execution_metadata),
        }
    except Exception as exc:
        terminal = {
            **base_record,
            "status": "failed",
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_run_record(run_dir / "run.json", terminal)
    return terminal


def matrix_summary(matrix: Sequence[ExperimentJob]) -> dict[str, object]:
    return {
        "jobs": len(matrix),
        "by_method": dict(sorted(Counter(job.implementation.method_id for job in matrix).items())),
        "by_dataset": dict(sorted(Counter(job.case.dataset_id for job in matrix).items())),
        "by_source_family": dict(sorted(Counter(job.case.source_family for job in matrix).items())),
        "by_seed": {str(key): value for key, value in sorted(Counter(job.case.seed for job in matrix).items())},
        "implementations": {
            job.implementation.method_id: job.implementation.revision
            for job in matrix
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--datasets", nargs="+", default=list(CURRENT_DATASETS + ("TSB-AD-M",)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[37, 38, 39])
    parser.add_argument("--tsb-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tsb-data-root", type=Path, default=DEFAULT_TSB_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = RunConfig(
            methods=tuple(args.methods),
            datasets=tuple(args.datasets),
            seeds=tuple(args.seeds),
            tsb_manifest=args.tsb_manifest,
            tsb_data_root=args.tsb_data_root,
            result_root=args.result_root,
            timeout_seconds=args.timeout_seconds,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        matrix = build_matrix(config)
        print(json.dumps(matrix_summary(matrix), indent=2, sort_keys=True))
        if args.dry_run:
            return 0
        outcomes = Counter(str(execute_job(job)["status"]) for job in matrix)
        print(json.dumps({"outcomes": dict(sorted(outcomes.items()))}, indent=2, sort_keys=True))
        return 0 if not {"failed", "timeout", "incompatible"}.intersection(outcomes) else 1
    except Exception as exc:
        print(f"Matrix validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
