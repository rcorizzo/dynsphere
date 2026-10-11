"""Adapter for the ML4ITS MTAD-GAT PyTorch implementation."""

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


METHOD_ID = "mtad_gat"


def official_backend(
    train: np.ndarray, test: np.ndarray, configuration: Mapping[str, object], seed: int
) -> BackendOutput:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    source = source_directory(METHOD_ID)
    sys.path.insert(0, str(source))
    try:
        from mtad_gat import MTAD_GAT
        from utils import SlidingWindowDataset
    finally:
        sys.path.pop(0)
    window = min(int(configuration.get("window_size", 100)), len(train) - 1, len(test) - 1)
    if window < 2:
        raise ValueError("MTAD-GAT requires at least three training and test rows")
    batch_size = int(configuration.get("batch_size", 128))
    train_tensor = torch.as_tensor(train, dtype=torch.float32)
    test_tensor = torch.as_tensor(test, dtype=torch.float32)
    train_loader = DataLoader(
        SlidingWindowDataset(train_tensor, window), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        SlidingWindowDataset(test_tensor, window), batch_size=batch_size, shuffle=False
    )
    model = MTAD_GAT(
        n_features=train.shape[1],
        window_size=window,
        out_dim=train.shape[1],
        use_gatv2=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(configuration.get("lr", 1e-3)))
    criterion = nn.MSELoss()
    model.train()
    for _ in range(int(configuration.get("epochs", 10))):
        for windows, targets in train_loader:
            optimizer.zero_grad()
            forecast, reconstruction = model(windows)
            target = targets.squeeze(1)
            loss = criterion(forecast, target) + criterion(reconstruction, windows)
            loss.backward()
            optimizer.step()
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for windows, targets in test_loader:
            forecast, reconstruction = model(windows)
            target = targets.squeeze(1)
            forecast_error = torch.mean((forecast - target) ** 2, dim=1)
            reconstruction_error = torch.mean((reconstruction - windows) ** 2, dim=(1, 2))
            values.append((forecast_error + reconstruction_error).cpu().numpy())
    return BackendOutput(
        scores=np.concatenate(values),
        warmup_rows=window,
        metadata={"backend": "ML4ITS MTAD-GAT", "window_size": window},
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

