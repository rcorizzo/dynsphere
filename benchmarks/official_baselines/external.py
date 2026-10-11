"""Timeout-safe execution of isolated external baseline adapters."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.official_baselines.contract import (
    ModelInputBundle,
    OutputValidationError,
    validate_scores,
    write_run_record,
)
from benchmarks.official_baselines.provenance import (
    ImplementationSpec,
    load_registry,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
REGISTRY_PATH = PACKAGE_ROOT / "registry.json"
LOG_TAIL_CHARACTERS = 8000


@dataclass(frozen=True)
class ExternalRunResult:
    status: str
    requested_method_id: str
    executed_method_id: str
    output_dir: Path
    scores_path: Path
    run_record_path: Path


def lookup_fallback(fallback_id: str) -> ImplementationSpec:
    """Resolve an explicitly requested fallback from the validated registry."""
    specs = load_registry(REGISTRY_PATH)
    try:
        return specs[fallback_id]
    except KeyError as exc:
        raise ValueError(f"Unknown fallback implementation: {fallback_id}") from exc


def _command_for(spec: ImplementationSpec) -> list[str]:
    environment_python = PACKAGE_ROOT / ".envs" / spec.method_id / "bin" / "python"
    command = [str(environment_python) if part == "{python}" else part for part in spec.adapter_command]
    executable = command[0]
    if Path(executable).is_absolute():
        if not Path(executable).is_file():
            raise FileNotFoundError(f"Adapter executable is unavailable: {executable}")
    elif shutil.which(executable) is None:
        raise FileNotFoundError(f"Adapter executable is unavailable: {executable}")
    return command


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _finish(
    output_dir: Path,
    status: str,
    requested: ImplementationSpec,
    executed: ImplementationSpec,
    started: float,
    *,
    error: str = "",
    stdout: str = "",
    stderr: str = "",
) -> ExternalRunResult:
    record = {
        "status": status,
        "requested_method_id": requested.method_id,
        "requested_revision": requested.revision,
        "executed_method_id": executed.method_id,
        "executed_revision": executed.revision,
        "elapsed_seconds": time.perf_counter() - started,
        "error": error,
        "stdout_tail": stdout[-LOG_TAIL_CHARACTERS:],
        "stderr_tail": stderr[-LOG_TAIL_CHARACTERS:],
    }
    write_run_record(output_dir / "run.json", record)
    return ExternalRunResult(
        status=status,
        requested_method_id=requested.method_id,
        executed_method_id=executed.method_id,
        output_dir=output_dir,
        scores_path=output_dir / "scores.npy",
        run_record_path=output_dir / "run.json",
    )


def run_external(
    spec: ImplementationSpec,
    bundle: ModelInputBundle,
    output_dir: Path,
    seed: int,
    timeout_seconds: int,
    fallback_id: str | None = None,
) -> ExternalRunResult:
    """Execute an external adapter and normalize all terminal outcomes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    executed = spec
    if fallback_id is not None:
        if fallback_id not in spec.fallback_ids:
            raise ValueError(f"{fallback_id!r} is not a declared fallback for {spec.method_id!r}")
        executed = lookup_fallback(fallback_id)

    try:
        command = _command_for(executed)
    except FileNotFoundError as exc:
        return _finish(
            output_dir,
            "incompatible",
            spec,
            executed,
            started,
            error=str(exc),
        )

    adapter_output = output_dir / "adapter_output"
    adapter_output.mkdir(parents=True, exist_ok=True)
    for name in ("scores.npy", "run.json", "received.json"):
        stale = adapter_output / name
        if stale.exists():
            stale.unlink()
    configuration_path = output_dir / "configuration.json"
    configuration_path.write_text(
        json.dumps(dict(executed.default_hyperparameters), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command.extend(
        [
            "--input-dir",
            str(bundle.input_dir),
            "--output-dir",
            str(adapter_output),
            "--seed",
            str(seed),
            "--config-json",
            str(configuration_path),
        ]
    )
    write_run_record(
        output_dir / "run.json",
        {
            "status": "running",
            "requested_method_id": spec.method_id,
            "requested_revision": spec.revision,
            "executed_method_id": executed.method_id,
            "executed_revision": executed.revision,
        },
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "1"),
        }
    )
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
        (output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (output_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        return _finish(
            output_dir,
            "timeout",
            spec,
            executed,
            started,
            error=f"Adapter exceeded {timeout_seconds} seconds",
            stdout=stdout,
            stderr=stderr,
        )
    except OSError as exc:
        return _finish(
            output_dir,
            "incompatible",
            spec,
            executed,
            started,
            error=str(exc),
        )

    (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return _finish(
            output_dir,
            "failed",
            spec,
            executed,
            started,
            error=f"Adapter exited with status {completed.returncode}",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    adapter_record_path = adapter_output / "run.json"
    try:
        adapter_record = json.loads(adapter_record_path.read_text(encoding="utf-8"))
        if not isinstance(adapter_record, dict) or adapter_record.get("status") != "complete":
            raise ValueError("Adapter run record is incomplete")
        scores = validate_scores(adapter_output / "scores.npy", bundle.expected_rows)
    except (OSError, ValueError, json.JSONDecodeError, OutputValidationError) as exc:
        return _finish(
            output_dir,
            "failed",
            spec,
            executed,
            started,
            error=str(exc),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    np.save(output_dir / "scores.npy", scores, allow_pickle=False)
    return _finish(
        output_dir,
        "complete",
        spec,
        executed,
        started,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
