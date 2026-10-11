"""Adapter for the author-released CATCH implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from benchmarks.official_baselines.adapters.common import (
    Backend,
    BackendOutput,
    adapter_main,
    run_backend_adapter,
    source_directory,
)


METHOD_ID = "catch"


def official_backend(
    train: np.ndarray, test: np.ndarray, configuration: Mapping[str, object], seed: int
) -> BackendOutput:
    del seed
    source = source_directory(METHOD_ID)
    sys.path.insert(0, str(source))
    try:
        from ts_benchmark.baselines.catch.CATCH import CATCH
    finally:
        sys.path.pop(0)
    model = CATCH(
        num_epochs=int(configuration.get("epochs", 10)),
        seq_len=int(configuration.get("window_size", 100)),
        batch_size=int(configuration.get("batch_size", 128)),
    )
    columns = [f"feature_{index}" for index in range(train.shape[1])]
    train_frame = pd.DataFrame(train, columns=columns)
    test_frame = pd.DataFrame(test, columns=columns)
    model.detect_fit(train_frame, test_frame)
    scores = np.asarray(model.detect_score(test_frame)[0], dtype=float).reshape(-1)
    if len(scores) > len(test):
        scores = scores[: len(test)]
    tail = len(test) - len(scores)
    return BackendOutput(
        scores=scores,
        tail_rows=tail,
        metadata={"backend": "author CATCH", "window_size": model.seq_len},
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

