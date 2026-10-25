"""Adapter using the author-released Anomaly Transformer model and objective."""

from __future__ import annotations

import sys
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


METHOD_ID = "anomaly_transformer"


def _enable_cpu_compatibility(torch, selected_device=None) -> bool:
    """Keep the author's unconditional CUDA transfers on the CPU when required."""
    use_cpu = (
        str(selected_device).startswith("cpu")
        if selected_device is not None
        else not torch.cuda.is_available()
    )
    if not use_cpu:
        return False

    def remain_on_cpu(tensor, *args, **kwargs):
        del args, kwargs
        return tensor

    torch.Tensor.cuda = remain_on_cpu
    return True


def _windows(values, window):
    import torch

    return torch.stack([values[index : index + window] for index in range(len(values) - window + 1)])


def _kl(p, q):
    import torch

    return torch.mean(torch.sum(p * (torch.log(p + 0.0001) - torch.log(q + 0.0001)), dim=-1), dim=1)


def _epoch_learning_rate(initial_rate, epoch_index):
    # The release adjusts the rate at the end of each one-based epoch.
    return initial_rate * (0.5 ** max(0, epoch_index - 1))


def official_backend(
    train: np.ndarray, test: np.ndarray, configuration: Mapping[str, object], seed: int
) -> BackendOutput:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = resolve_torch_device(torch)
    cpu_compatibility = _enable_cpu_compatibility(torch, device)
    source = source_directory(METHOD_ID)
    sys.path.insert(0, str(source))
    try:
        from model.AnomalyTransformer import AnomalyTransformer
    finally:
        sys.path.pop(0)
    window = min(int(configuration.get("window_size", 100)), len(train), len(test))
    if window < 2:
        raise ValueError("Anomaly Transformer requires at least two rows")
    train_windows = _windows(torch.as_tensor(train, dtype=torch.float32), window)
    test_windows = _windows(torch.as_tensor(test, dtype=torch.float32), window)
    batch_size = int(configuration.get("batch_size", 128))
    train_loader = DataLoader(TensorDataset(train_windows), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_windows), batch_size=batch_size, shuffle=False)
    model = AnomalyTransformer(
        win_size=window,
        enc_in=train.shape[1],
        c_out=train.shape[1],
        d_model=int(configuration.get("d_model", 128)),
        n_heads=int(configuration.get("n_heads", 4)),
        e_layers=int(configuration.get("e_layers", 3)),
    ).to(device)
    initial_rate = float(configuration.get("lr", 1e-4))
    revision = str(configuration.get('training_revision', 'historical-v1'))
    if revision not in ('historical-v1', 'released-budget-v2'):
        raise ValueError(f'Unknown Anomaly Transformer training revision: {revision}')
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_rate)
    reconstruction = nn.MSELoss()
    k = float(configuration.get("k", 3.0))
    model.train()
    epochs = int(configuration.get('epochs', 3))
    training_loss = []
    for epoch in range(epochs):
        if revision == 'released-budget-v2':
            for group in optimizer.param_groups:
                group['lr'] = _epoch_learning_rate(initial_rate, epoch)
        epoch_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output, series, prior, _ = model(batch)
            series_loss = 0.0
            prior_loss = 0.0
            for series_value, prior_value in zip(series, prior):
                normalized_prior = prior_value / torch.unsqueeze(
                    torch.sum(prior_value, dim=-1), dim=-1
                ).repeat(1, 1, 1, window)
                series_loss += torch.mean(_kl(series_value, normalized_prior.detach()))
                series_loss += torch.mean(_kl(normalized_prior.detach(), series_value))
                prior_loss += torch.mean(_kl(normalized_prior, series_value.detach()))
                prior_loss += torch.mean(_kl(series_value.detach(), normalized_prior))
            series_loss /= len(prior)
            prior_loss /= len(prior)
            rec_loss = reconstruction(output, batch)
            (rec_loss - k * series_loss).backward(retain_graph=True)
            (rec_loss + k * prior_loss).backward()
            optimizer.step()
            epoch_losses.append(float(rec_loss.detach().cpu()))
        training_loss.append(float(np.mean(epoch_losses)))
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    energies: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in test_loader:
            batch = batch.to(device)
            output, series, prior, _ = model(batch)
            loss = torch.mean(criterion(batch, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for series_value, prior_value in zip(series, prior):
                normalized_prior = prior_value / torch.unsqueeze(
                    torch.sum(prior_value, dim=-1), dim=-1
                ).repeat(1, 1, 1, window)
                series_loss += _kl(series_value, normalized_prior.detach()) * 50
                prior_loss += _kl(normalized_prior, series_value.detach()) * 50
            metric = torch.softmax(-(series_loss + prior_loss), dim=-1)
            energies.append((metric * loss)[:, -1].cpu().numpy())
    return BackendOutput(
        scores=np.concatenate(energies),
        warmup_rows=window - 1,
        metadata={
            "backend": "author Anomaly Transformer",
            "window_size": window,
            "d_model": int(configuration.get("d_model", 128)),
            "device": str(device),
            "cpu_compatibility": cpu_compatibility,
            "training_revision": revision,
            "epochs": epochs,
            "training_reconstruction_loss": training_loss,
            "n_heads": int(configuration.get('n_heads', 4)),
            "checkpoint_selection": 'fixed-final-epoch-no-test-access',
            "score_alignment": 'last-position-of-overlapping-window',
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
