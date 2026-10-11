"""Adapter boundary for the author-released OmniAnomaly implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from benchmarks.official_baselines.adapters.common import (
    Backend,
    BackendOutput,
    adapter_main,
    run_backend_adapter,
    source_directory,
)


METHOD_ID = "omni_anomaly"


def official_backend(
    train: np.ndarray, test: np.ndarray, configuration: Mapping[str, object], seed: int
) -> BackendOutput:
    del train, test, configuration, seed
    source = source_directory(METHOD_ID)
    if not source.joinpath("omni_anomaly", "model.py").is_file():
        raise RuntimeError("Pinned OmniAnomaly source checkout is unavailable")
    raise RuntimeError(
        "Author OmniAnomaly requires its pinned TensorFlow 1.12 environment; "
        "use the explicitly registered benchmark-maintained fallback when setup reports incompatibility"
    )


def run_adapter(
    input_dir: Path,
    output_dir: Path,
    seed: int,
    configuration: Mapping[str, object],
    backend: Backend | None = None,
) -> None:
    run_backend_adapter(
        METHOD_ID, input_dir, output_dir, seed, configuration, backend or official_backend
    )


if __name__ == "__main__":
    adapter_main(METHOD_ID, official_backend)

