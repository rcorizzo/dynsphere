"""Compact causal history representation, trained on normal targets only."""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import history_batches


@contextmanager
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class HistoryEncoder(nn.Module):
    def __init__(self, n_features, window=100, hidden_dim=32, latent_dim=8,
                 reconstruction_weight=1.):
        super().__init__()
        self.window, self.n_features = window, n_features
        self.convolution = nn.Conv1d(n_features, hidden_dim, 5)
        self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, latent_dim)
        self.forecasting = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, n_features))
        self.reconstruction = (nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
                                            nn.Linear(hidden_dim, window * n_features))
                               if reconstruction_weight else None)

    def encode(self, windows):
        convolved = F.silu(self.convolution(F.pad(windows.transpose(1, 2), (4, 0))))
        _, state = self.recurrent(convolved.transpose(1, 2))
        return self.projection(state[0])

    def forecast(self, z):
        return self.forecasting(z)

    def reconstruct(self, z):
        if self.reconstruction is None:
            raise ValueError('The reconstruction objective is disabled')
        return self.reconstruction(z).reshape(-1, self.window, self.n_features)


def train_encoder(x, groups, targets, *, window, hidden_dim, latent_dim,
                  reconstruction_weight, epochs, batch_size, random_state):
    if epochs < 1 or hidden_dim < 1 or latent_dim < 1 or reconstruction_weight < 0:
        raise ValueError('Invalid neural configuration')
    with single_thread(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_state)
        model = HistoryEncoder(x.shape[1], window, hidden_dim, latent_dim, reconstruction_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(random_state)
        losses = []
        for _ in range(epochs):
            total, count = 0., 0
            for rows, windows in history_batches(x, groups, rng.permutation(targets), window, batch_size):
                w = torch.as_tensor(windows, dtype=torch.float32)
                y = torch.as_tensor(x[rows], dtype=torch.float32)
                z = model.encode(w)
                loss = F.mse_loss(model.forecast(z), y)
                if reconstruction_weight:
                    loss = loss + reconstruction_weight * F.mse_loss(model.reconstruct(z), w)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite representation training loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach()) * len(rows)
                count += len(rows)
            losses.append(total / count)
        model.eval()
        return model, losses
