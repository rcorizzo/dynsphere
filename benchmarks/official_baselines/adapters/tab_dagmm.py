"""Adapter for TAB's benchmark-maintained DAGMM implementation."""

from __future__ import annotations

import sys
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


METHOD_ID = "dagmm"


def official_backend(
    train: np.ndarray, test: np.ndarray, configuration: Mapping[str, object], seed: int
) -> BackendOutput:
    if configuration.get("implementation_backend", "preferred") != "preferred":
        raise RuntimeError("The requested DAGMM fallback requires its dedicated environment")
    source = source_directory(METHOD_ID)
    package_root = source / "ts_benchmark" / "baselines" / "tods" / "third_party"
    sys.path.insert(0, str(package_root))
    try:
        from tods.detection_algorithm.core.dagmm.dagmm import DAGMM
    finally:
        sys.path.pop(0)
    model = DAGMM(
        epoch_size=int(configuration.get("epochs", 10)),
        minibatch_size=int(configuration.get("batch_size", 128)),
        random_seed=seed,
        normalize=False,
    )
    model.fit(train)
    scores = np.asarray(model.decision_function(test), dtype=float).reshape(-1)
    return BackendOutput(
        scores=scores,
        metadata={"backend": "TAB DAGMM", "upstream_scaling": "disabled"},
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

