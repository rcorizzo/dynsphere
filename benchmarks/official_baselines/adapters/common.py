"""Shared label-free input, alignment, and output logic for adapters."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class BackendOutput:
    scores: np.ndarray
    warmup_rows: int = 0
    tail_rows: int = 0
    score_direction: str = "larger-is-more-anomalous"
    metadata: Mapping[str, object] = field(default_factory=dict)


Backend = Callable[[np.ndarray, np.ndarray, Mapping[str, object], int], BackendOutput]


def resolve_torch_device(torch, requested: str | None = None):
    """Resolve the requested execution device and reject unavailable CUDA."""

    selection = (requested or os.environ.get("DYNSPHERE_DEVICE", "auto")).lower()
    if selection not in {"auto", "cpu", "cuda"}:
        raise ValueError("DYNSPHERE_DEVICE must be auto, cpu, or cuda")
    if selection == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA accelerator is visible")
    if selection == "auto":
        selection = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(selection)


def source_directory(method_id: str) -> Path:
    return Path(__file__).resolve().parents[1] / ".external" / method_id


def align_scores(output: BackendOutput, expected_rows: int) -> np.ndarray:
    raw = np.asarray(output.scores, dtype=float).reshape(-1)
    if output.warmup_rows < 0 or output.tail_rows < 0:
        raise ValueError("Unscored row counts must be nonnegative")
    if len(raw) + output.warmup_rows + output.tail_rows != expected_rows:
        raise ValueError(
            "Backend scores do not align with warm-up, tail, and test row counts"
        )
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Backend scores must be finite and nonempty")
    if output.score_direction == "larger-is-more-normal":
        raw = -raw
    elif output.score_direction != "larger-is-more-anomalous":
        raise ValueError(f"Unsupported score direction: {output.score_direction}")
    prefix = np.full(output.warmup_rows, raw[0], dtype=float)
    suffix = np.full(output.tail_rows, raw[-1], dtype=float)
    return np.concatenate([prefix, raw, suffix])


def run_backend_adapter(
    method_id: str,
    input_dir: Path,
    output_dir: Path,
    seed: int,
    configuration: Mapping[str, object],
    backend: Backend,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    train = np.asarray(np.load(input_dir / "train.npy", allow_pickle=False), dtype=float)
    test = np.asarray(np.load(input_dir / "test.npy", allow_pickle=False), dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Adapter inputs must have matching two-dimensional shapes")
    backend_output = backend(train, test, configuration, seed)
    scores = align_scores(backend_output, len(test))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "scores.npy", scores, allow_pickle=False)
    record = {
        "status": "complete",
        "method_id": method_id,
        "seed": seed,
        "warmup_rows": backend_output.warmup_rows,
        "tail_rows": backend_output.tail_rows,
        "warmup_alignment": "repeat-first-valid",
        "tail_alignment": "repeat-last-valid",
        "score_direction": "larger-is-more-anomalous",
        "raw_score_direction": backend_output.score_direction,
        **dict(backend_output.metadata),
    }
    (output_dir / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def adapter_main(method_id: str, backend: Backend) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--backend", default="preferred")
    args = parser.parse_args()
    configuration = json.loads(args.config_json.read_text(encoding="utf-8"))
    configuration["implementation_backend"] = args.backend
    run_backend_adapter(
        method_id,
        args.input_dir,
        args.output_dir,
        args.seed,
        configuration,
        backend,
    )
