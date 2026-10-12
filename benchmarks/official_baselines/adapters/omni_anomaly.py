"""Adapter boundary for the author-released OmniAnomaly implementation."""

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


METHOD_ID = "omni_anomaly"


def official_backend(
    train: np.ndarray, test: np.ndarray, configuration: Mapping[str, object], seed: int
) -> BackendOutput:
    backend = configuration.get("implementation_backend", "preferred")
    if backend == "tsb-ad":
        import torch

        torch.manual_seed(seed)
        source = source_directory("omni_anomaly_tsb_ad")
        sys.path.insert(0, str(source))
        try:
            from TSB_AD.models.OmniAnomaly import OmniAnomaly
        finally:
            sys.path.pop(0)
        window = min(int(configuration.get("window_size", 100)), len(train), len(test))
        if window < 2:
            raise ValueError("OmniAnomaly requires at least two training and test rows")
        model = OmniAnomaly(
            win_size=window,
            feats=train.shape[1],
            batch_size=int(configuration.get("batch_size", 128)),
            epochs=int(configuration.get("epochs", 20)),
            lr=float(configuration.get("lr", 0.002)),
        )
        model.fit(train)
        scores = np.asarray(model.decision_function(test), dtype=float).reshape(-1)
        return BackendOutput(
            scores=scores,
            metadata={
                "backend": "TSB-AD OmniAnomaly fallback",
                "window_size": window,
            },
        )

    del train, test, seed
    if backend != "preferred":
        raise RuntimeError(f"Unsupported OmniAnomaly backend: {backend}")
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
