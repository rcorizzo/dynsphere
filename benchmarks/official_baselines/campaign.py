"""Portable campaign profiles, deterministic sharding, and manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from benchmarks.official_baselines.run_expanded_evaluation import (
    CURRENT_DATASETS,
    ExperimentJob,
)


CAMPAIGN_SCHEMA = "official-baseline-campaign-v1"
LOGICAL_CAMPAIGN_KEY_FIELDS = (
    "case_id",
    "dataset_id",
    "source_family",
    "trajectory_id",
    "seed",
    "method_id",
)


@dataclass(frozen=True)
class CampaignProfile:
    name: str
    methods: tuple[str, ...]
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    timeout_seconds: int


PROFILES = {
    "macbook-safe": CampaignProfile(
        name="macbook-safe",
        methods=("dagmm", "tranad", "omni_anomaly_tsb_ad"),
        datasets=("MSL", "SMAP", "SMD"),
        seeds=(37,),
        timeout_seconds=3600,
    ),
    "hpc-seed37": CampaignProfile(
        name="hpc-seed37",
        methods=(
            "dagmm",
            "catch",
            "mtad_gat",
            "omni_anomaly",
            "anomaly_transformer",
            "tranad",
        ),
        datasets=CURRENT_DATASETS + ("TSB-AD-M",),
        seeds=(37,),
        timeout_seconds=14400,
    ),
    "hpc-followup": CampaignProfile(
        name="hpc-followup",
        methods=(
            "dagmm",
            "catch",
            "mtad_gat",
            "omni_anomaly",
            "anomaly_transformer",
            "tranad",
        ),
        datasets=CURRENT_DATASETS,
        seeds=(38, 39),
        timeout_seconds=14400,
    ),
    "paper-multiseed": CampaignProfile(
        name="paper-multiseed",
        methods=(
            "pca_reconstruction",
            "kmeans_distance",
            "dagmm",
            "catch",
            "mtad_gat",
            "omni_anomaly_tsb_ad",
            "anomaly_transformer",
            "tranad",
            "predictive_euclidean_single",
            "dynsphere",
            "dynsphere_euclidean",
            "dynsphere_one_regime",
            "dynsphere_euclidean_one_regime",
        ),
        datasets=CURRENT_DATASETS,
        seeds=(37, 38, 39),
        timeout_seconds=14400,
    ),
}


def resolve_profile(name: str) -> CampaignProfile:
    """Return a named campaign profile with a useful validation error."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown campaign profile {name!r}; choose one of {sorted(PROFILES)}"
        ) from exc


def partition_jobs(
    jobs: Sequence[ExperimentJob],
    shard_index: int,
    shard_count: int,
) -> tuple[ExperimentJob, ...]:
    """Select one deterministic modulo partition of an ordered job matrix."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0:
        raise ValueError("shard_index must be non-negative")
    if shard_index >= shard_count:
        raise ValueError("shard_index must be smaller than shard_count")
    return tuple(job for index, job in enumerate(jobs) if index % shard_count == shard_index)


def scheduler_shard_index(raw_index: str, *, one_based: bool) -> int:
    """Normalize a scheduler array index to the zero-based campaign convention."""

    try:
        value = int(raw_index)
    except ValueError as exc:
        raise ValueError(f"Scheduler array index must be an integer: {raw_index!r}") from exc
    if one_based:
        if value < 1:
            raise ValueError("A one-based scheduler array index must be at least 1")
        return value - 1
    if value < 0:
        raise ValueError("A zero-based scheduler array index must be non-negative")
    return value


def _job_document(job: ExperimentJob) -> dict[str, object]:
    return {
        "adapter_revision": job.implementation.adapter_revision,
        "case_id": job.case.case_id,
        "dataset_id": job.case.dataset_id,
        "method_id": job.implementation.method_id,
        "seed": job.case.seed,
        "source_family": job.case.source_family,
        "trajectory_id": job.case.trajectory_id,
    }


def campaign_document(
    profile_name: str,
    jobs: Sequence[ExperimentJob],
    shard_count: int,
) -> dict[str, object]:
    """Build a deterministic manifest for exact campaign membership."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    expected_jobs = sorted(
        (_job_document(job) for job in jobs),
        key=lambda row: (
            str(row["case_id"]),
            int(row["seed"]),
            str(row["method_id"]),
            str(row["adapter_revision"]),
        ),
    )
    payload: dict[str, object] = {
        "schema_version": CAMPAIGN_SCHEMA,
        "profile": profile_name,
        "shard_count": shard_count,
        "expected_job_count": len(expected_jobs),
        "expected_jobs": expected_jobs,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "campaign_id": hashlib.sha256(canonical).hexdigest()}


def merge_campaign_documents(
    documents: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Merge component manifests, with later sources replacing logical jobs."""
    if not documents:
        raise ValueError("At least one campaign manifest is required")
    selected: dict[tuple[object, ...], dict[str, object]] = {}
    source_ids: list[str] = []
    shard_counts: list[int] = []
    for document in documents:
        if document.get("schema_version") != CAMPAIGN_SCHEMA:
            raise ValueError("Campaign manifest has an unsupported schema")
        campaign_id = document.get("campaign_id")
        rows = document.get("expected_jobs")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise ValueError("Every source campaign must identify its campaign_id")
        if not isinstance(rows, list) or document.get("expected_job_count") != len(rows):
            raise ValueError("Source campaign expected jobs are inconsistent")
        source_ids.append(campaign_id)
        shard_counts.append(int(document.get("shard_count", 1)))
        for row in rows:
            if not isinstance(row, dict) or any(
                field not in row
                for field in LOGICAL_CAMPAIGN_KEY_FIELDS + ("adapter_revision",)
            ):
                raise ValueError("Source campaign contains an incomplete job entry")
            key = tuple(row[field] for field in LOGICAL_CAMPAIGN_KEY_FIELDS)
            selected[key] = dict(row)
    expected_jobs = sorted(
        selected.values(),
        key=lambda row: tuple(str(row[field]) for field in LOGICAL_CAMPAIGN_KEY_FIELDS),
    )
    payload: dict[str, object] = {
        "schema_version": CAMPAIGN_SCHEMA,
        "profile": "merged-components",
        "shard_count": max(shard_counts),
        "source_campaign_ids": source_ids,
        "expected_job_count": len(expected_jobs),
        "expected_jobs": expected_jobs,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "campaign_id": hashlib.sha256(canonical).hexdigest()}


def write_campaign_manifest(path: Path, document: dict[str, object]) -> Path:
    """Write a manifest once and reject reuse of its path for another campaign."""

    def validate_existing() -> Path:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Existing campaign manifest is unreadable: {path}") from exc
        if existing != document:
            raise ValueError(f"Existing path belongs to a different campaign: {path}")
        return path

    if path.exists():
        return validate_existing()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return validate_existing()
        return path
    finally:
        temporary.unlink(missing_ok=True)
