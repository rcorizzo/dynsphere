"""Prepare pinned, unmodified source checkouts and isolated environments."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.official_baselines.provenance import ImplementationSpec, load_registry


REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"


@dataclass(frozen=True)
class SetupRecord:
    method_id: str
    status: str
    source_dir: Path
    environment_dir: Path
    revision: str
    reused: bool
    reason: str = ""
    python_version: str = ""
    packages: tuple[str, ...] = ()


def _run(arguments: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _package_inventory(python: Path) -> tuple[str, ...]:
    script = (
        "from importlib.metadata import distributions; "
        "rows = sorted(f\"{item.metadata['Name']}=={item.version}\" "
        "for item in distributions() if item.metadata['Name']); "
        "print('\\n'.join(rows))"
    )
    return tuple(_run([str(python), "-c", script]).splitlines())


def _environment_command(
    python_request: str, environment_dir: Path, current_python: str
) -> list[str]:
    if python_request == current_python:
        return [sys.executable, "-m", "venv", "--clear", str(environment_dir)]
    return [
        "uv",
        "venv",
        str(environment_dir),
        "--python",
        python_request,
        "--clear",
    ]


def _paths(method_id: str, root: Path) -> tuple[Path, Path, Path]:
    return (
        root / ".external" / method_id,
        root / ".envs" / method_id,
        root / ".external" / "setup-records" / f"{method_id}.json",
    )


def _write_record(path: Path, record: SetupRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    payload["source_dir"] = str(record.source_dir)
    payload["environment_dir"] = str(record.environment_dir)
    payload["packages"] = list(record.packages)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _spec(method_id: str) -> ImplementationSpec:
    registry = load_registry(REGISTRY_PATH)
    try:
        return registry[method_id]
    except KeyError as exc:
        raise ValueError(f"Unknown implementation: {method_id}") from exc


def verify_source(method_id: str, external_root: Path) -> SetupRecord:
    spec = _spec(method_id)
    source_dir, environment_dir, _ = _paths(method_id, external_root)
    base = {
        "method_id": method_id,
        "source_dir": source_dir,
        "environment_dir": environment_dir,
        "revision": spec.revision,
        "reused": True,
    }
    if not source_dir.joinpath(".git").exists():
        return SetupRecord(status="incompatible", reason="Source checkout is missing", **base)
    try:
        revision = _run(["git", "rev-parse", "HEAD"], cwd=source_dir)
        modified = _run(["git", "status", "--porcelain"], cwd=source_dir)
    except (OSError, subprocess.CalledProcessError) as exc:
        return SetupRecord(status="incompatible", reason=str(exc), **base)
    if revision != spec.revision:
        return SetupRecord(status="incompatible", reason="Pinned revision does not match", **base)
    if modified:
        return SetupRecord(status="incompatible", reason="Source checkout is modified", **base)
    python = environment_dir / "bin" / "python"
    marker = environment_dir / "setup.json"
    if not python.is_file() or not marker.is_file():
        return SetupRecord(status="incompatible", reason="Environment is missing", **base)
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("revision") != spec.revision or saved.get("python_request") != spec.python:
            raise ValueError("Environment marker does not match provenance")
        version = _run([str(python), "--version"])
        packages = _package_inventory(python)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        return SetupRecord(status="incompatible", reason=str(exc), **base)
    return SetupRecord(
        status="ready", python_version=version, packages=packages, **base
    )


def prepare_source(method_id: str, external_root: Path) -> SetupRecord:
    spec = _spec(method_id)
    source_dir, environment_dir, record_path = _paths(method_id, external_root)
    if source_dir.exists():
        existing = verify_source(method_id, external_root)
        if existing.status == "ready" or "modified" in existing.reason.lower():
            _write_record(record_path, existing)
            return existing
    try:
        created = not source_dir.exists()
        if not source_dir.exists():
            source_dir.parent.mkdir(parents=True, exist_ok=True)
            _run(["git", "clone", "--no-checkout", spec.repository_url, str(source_dir)])
        if not created and _run(["git", "status", "--porcelain"], cwd=source_dir):
            raise RuntimeError("Source checkout is modified")
        _run(["git", "checkout", "--detach", spec.revision], cwd=source_dir)
        if _run(["git", "status", "--porcelain"], cwd=source_dir):
            raise RuntimeError("Source checkout is modified")

        if not re.fullmatch(r"\d+\.\d+", spec.python):
            raise RuntimeError(f"Unsupported Python request: {spec.python}")
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        environment_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(_environment_command(spec.python, environment_dir, current))
        python = environment_dir / "bin" / "python"
        requirements = source_dir / "requirements.txt"
        if requirements.is_file():
            _run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "-r",
                    str(requirements),
                ]
            )
        marker = {
            "method_id": method_id,
            "revision": spec.revision,
            "python_request": spec.python,
        }
        environment_dir.joinpath("setup.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        version = _run([str(python), "--version"])
        packages = _package_inventory(python)
        record = SetupRecord(
            method_id=method_id,
            status="ready",
            source_dir=source_dir,
            environment_dir=environment_dir,
            revision=spec.revision,
            reused=False,
            python_version=version,
            packages=packages,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        record = SetupRecord(
            method_id=method_id,
            status="incompatible",
            source_dir=source_dir,
            environment_dir=environment_dir,
            revision=spec.revision,
            reused=False,
            reason=str(exc),
        )
    _write_record(record_path, record)
    return record
