"""Adapter for the author-released CATCH implementation."""

from __future__ import annotations

import importlib
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
FREQUENCY_OVERLAP_CORRECTION = "coverage-count-v1"


def _overlap_average(torch, patch_losses, patch_stride: int):
    """Average overlapping patch losses using geometric coverage counts."""
    batch, patch_count, patch_size, channels = patch_losses.shape
    output_length = patch_size + (patch_count - 1) * patch_stride
    starts = torch.arange(patch_count, device=patch_losses.device) * patch_stride
    offsets = torch.arange(patch_size, device=patch_losses.device)
    indices = (starts[:, None] + offsets[None, :]).reshape(1, -1, 1)
    indices = indices.expand(batch, -1, channels)
    values = patch_losses.reshape(batch, patch_count * patch_size, channels)
    totals = torch.zeros(
        (batch, output_length, channels),
        dtype=patch_losses.dtype,
        device=patch_losses.device,
    )
    coverage = torch.zeros_like(totals)
    totals.scatter_add_(1, indices, values)
    coverage.scatter_add_(1, indices, torch.ones_like(values))
    return totals / coverage.clamp_min(1)


def _stable_frequency_forward(self, outputs, batch_y):
    """CATCH frequency score with loss-independent patch coverage."""
    import torch

    output_patch = outputs.unfold(
        dimension=1, size=self.patch_size, step=self.patch_stride
    )
    batch, patch_count, channels, patch_size = output_patch.shape
    output_patch = output_patch.permute(0, 1, 3, 2).reshape(
        batch * patch_count, patch_size, channels
    )
    y_patch = batch_y.unfold(
        dimension=1, size=self.patch_size, step=self.patch_stride
    )
    y_patch = y_patch.permute(0, 1, 3, 2).reshape(
        batch * patch_count, patch_size, channels
    )
    patch_losses = self.metric(output_patch, y_patch)
    patch_losses = patch_losses.repeat(1, self.patch_size, 1).reshape(
        batch, patch_count, self.patch_size, channels
    )
    main_loss = _overlap_average(torch, patch_losses, self.patch_stride)
    if self.padding_length > 0:
        padding_loss = self.metric(
            outputs[:, -self.padding_length :, :],
            batch_y[:, -self.padding_length :, :],
        )
        padding_loss = padding_loss.repeat(1, self.padding_length, 1)
        return torch.concat([main_loss, padding_loss], dim=1)
    return main_loss


def _load_corrected_catch():
    source = source_directory(METHOD_ID)
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("ts_benchmark.baselines.catch.CATCH")
    finally:
        sys.path.pop(0)
    module.frequency_criterion.forward = _stable_frequency_forward
    return module.CATCH


def official_backend(
    train: np.ndarray, test: np.ndarray, configuration: Mapping[str, object], seed: int
) -> BackendOutput:
    del seed
    correction = configuration.get("frequency_overlap_correction")
    if correction != FREQUENCY_OVERLAP_CORRECTION:
        raise ValueError(
            "CATCH requires frequency_overlap_correction='coverage-count-v1'"
        )
    CATCH = _load_corrected_catch()
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
        metadata={
            "backend": "author CATCH",
            "window_size": model.seq_len,
            "device": str(getattr(model, "device", "unknown")),
            "frequency_overlap_correction": FREQUENCY_OVERLAP_CORRECTION,
        },
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
