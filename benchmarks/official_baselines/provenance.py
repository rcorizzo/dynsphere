"""Implementation provenance and deterministic experiment identities."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_REQUIRED_FIELDS = {
    "method_id",
    "display_name",
    "source_kind",
    "implementation_category",
    "repository_url",
    "revision",
    "python",
    "adapter_command",
    "default_hyperparameters",
    "fallback_ids",
}


@dataclass(frozen=True)
class ImplementationSpec:
    method_id: str
    display_name: str
    source_kind: str
    implementation_category: str
    repository_url: str
    revision: str
    python: str
    adapter_command: tuple[str, ...]
    default_hyperparameters: Mapping[str, object]
    fallback_ids: tuple[str, ...]


def resolve_local_revision(repo: Path) -> str:
    """Return the immutable revision of the local implementation."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if _COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"Local Git revision is invalid: {revision!r}")
    return revision


def _repository_root(start: Path) -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip())


def _parse_entry(raw: object, repo_root: Path | None) -> ImplementationSpec:
    if not isinstance(raw, dict):
        raise ValueError("Every registry entry must be an object")
    missing = _REQUIRED_FIELDS.difference(raw)
    if missing:
        raise ValueError(f"Registry entry is missing fields: {sorted(missing)}")

    source_kind = raw["source_kind"]
    if source_kind not in {"local", "external"}:
        raise ValueError(f"Unsupported source_kind: {source_kind!r}")

    revision = raw["revision"]
    if source_kind == "local":
        if revision != "current-git-commit":
            raise ValueError("Local revisions must use current-git-commit")
        if repo_root is None:
            raise ValueError("A local registry entry requires a Git repository")
        revision = resolve_local_revision(repo_root)
    elif not isinstance(revision, str) or _COMMIT_PATTERN.fullmatch(revision) is None:
        raise ValueError("External revisions must be full 40-character Git commits")

    command = raw["adapter_command"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(argument, str) and argument for argument in command)
    ):
        raise ValueError("adapter_command must be a nonempty argument list")
    hyperparameters = raw["default_hyperparameters"]
    fallbacks = raw["fallback_ids"]
    string_fields = (
        "method_id",
        "display_name",
        "implementation_category",
        "repository_url",
        "python",
    )
    if not all(isinstance(raw[name], str) and raw[name] for name in string_fields):
        raise ValueError("Registry string fields must be nonempty")
    if not isinstance(hyperparameters, dict):
        raise ValueError("default_hyperparameters must be an object")
    if not isinstance(fallbacks, list) or not all(isinstance(value, str) for value in fallbacks):
        raise ValueError("fallback_ids must be a list of method identifiers")

    return ImplementationSpec(
        method_id=raw["method_id"],
        display_name=raw["display_name"],
        source_kind=source_kind,
        implementation_category=raw["implementation_category"],
        repository_url=raw["repository_url"],
        revision=revision,
        python=raw["python"],
        adapter_command=tuple(command),
        default_hyperparameters=dict(hyperparameters),
        fallback_ids=tuple(fallbacks),
    )


def validate_registry(specs: Mapping[str, ImplementationSpec]) -> None:
    """Validate resolved entries and cross-entry fallback references."""
    if not specs:
        raise ValueError("The implementation registry is empty")
    for method_id, spec in specs.items():
        if method_id != spec.method_id:
            raise ValueError(f"Registry key mismatch for {method_id!r}")
        if _COMMIT_PATTERN.fullmatch(spec.revision) is None:
            raise ValueError(f"Implementation {method_id!r} has a mutable revision")
        for fallback_id in spec.fallback_ids:
            if fallback_id == method_id or fallback_id not in specs:
                raise ValueError(f"Invalid fallback {fallback_id!r} for {method_id!r}")


def load_registry(path: Path) -> dict[str, ImplementationSpec]:
    """Load, resolve, and validate the implementation registry."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("implementations") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Registry root must contain an implementations list")
    needs_local_revision = any(
        isinstance(raw, dict) and raw.get("source_kind") == "local" for raw in rows
    )
    repo_root = _repository_root(path.parent) if needs_local_revision else None
    specs: dict[str, ImplementationSpec] = {}
    for raw in rows:
        spec = _parse_entry(raw, repo_root)
        if spec.method_id in specs:
            raise ValueError(f"Duplicate method_id: {spec.method_id!r}")
        specs[spec.method_id] = spec
    validate_registry(specs)
    return specs


def run_identity(
    case_identity: str,
    implementation: ImplementationSpec,
    hyperparameters: Mapping[str, object],
    seed: int,
    data_digest: str,
    scaling_rule: str,
) -> str:
    """Hash every input that can change an experiment's interpretation."""
    payload = {
        "case_identity": case_identity,
        "method_id": implementation.method_id,
        "revision": implementation.revision,
        "hyperparameters": dict(hyperparameters),
        "seed": seed,
        "data_digest": data_digest,
        "scaling_rule": scaling_rule,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
