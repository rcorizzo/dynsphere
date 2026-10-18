"""Adapter using the author-released TranAD model and two-phase objective."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

import numpy as np

from benchmarks.official_baselines.adapters.common import (
    Backend,
    BackendOutput,
    adapter_main,
    resolve_torch_device,
    run_backend_adapter,
    source_directory,
)


METHOD_ID = "tranad"


@contextmanager
def _upstream_import_arguments():
    original = sys.argv
    sys.argv = [original[0], "--dataset", "synthetic", "--model", "TranAD"]
    try:
        yield
    finally:
        sys.argv = original


def _prefix_windows(values, window):
    import torch

    windows = []
    for index in range(len(values)):
        if index >= window:
            current = values[index - window : index]
        else:
            current = torch.cat([values[0].repeat(window - index, 1), values[:index]])
        windows.append(current)
    return torch.stack(windows)


def official_backend(
    train: np.ndarray, test: np.ndarray, configuration: Mapping[str, object], seed: int
) -> BackendOutput:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = resolve_torch_device(torch)
    source = source_directory(METHOD_ID)
    sys.path.insert(0, str(source))
    try:
        with _upstream_import_arguments():
            from src.models import TranAD
    finally:
        sys.path.pop(0)
    model = TranAD(train.shape[1]).double().to(device)
    window = model.n_window
    train_windows = _prefix_windows(torch.as_tensor(train, dtype=torch.double), window)
    test_windows = _prefix_windows(torch.as_tensor(test, dtype=torch.double), window)
    batch_size = int(configuration.get("batch_size", model.batch))
    train_loader = DataLoader(TensorDataset(train_windows), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_windows), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(configuration.get("lr", model.lr)), weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    model.train()
    epochs = int(configuration.get("epochs", 5))
    for epoch in range(epochs):
        for (batch,) in train_loader:
            batch = batch.to(device)
            sequence = batch.permute(1, 0, 2)
            target = sequence[-1].reshape(1, len(batch), -1)
            first, second = model(sequence, target)
            loss = (1 / (epoch + 1)) * criterion(first, target)
            loss += (1 - 1 / (epoch + 1)) * criterion(second, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    losses: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(device)
            sequence = batch.permute(1, 0, 2)
            target = sequence[-1].reshape(1, len(batch), -1)
            _, output = model(sequence, target)
            losses.append(torch.mean((output - target) ** 2, dim=(0, 2)).cpu().numpy())
    return BackendOutput(
        scores=np.concatenate(losses),
        metadata={
            "backend": "author TranAD",
            "window_size": window,
            "device": str(device),
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
