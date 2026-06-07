#!/usr/bin/env python3
"""Runtime smoke test for candidate deep TSAD baselines.

This script is intentionally small: it uses compact local neural sequence
baselines to estimate integration cost on the current data. Do not use these
rows as paper results.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def require_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def set_seed(torch, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def make_windows(
    x: np.ndarray,
    entity_ids: np.ndarray,
    labels: np.ndarray | None,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    centers: list[int] = []
    y: list[int] = []
    for entity_id in np.unique(entity_ids):
        idx = np.flatnonzero(entity_ids == entity_id)
        seq = x[idx]
        if len(seq) < window:
            continue
        for local_start in range(0, len(seq) - window + 1):
            local_end = local_start + window
            center = idx[local_end - 1]
            windows.append(seq[local_start:local_end])
            centers.append(center)
            if labels is not None:
                y.append(int(labels[center]))
    y_arr = np.asarray(y, dtype=int) if labels is not None else np.zeros(len(windows), dtype=int)
    return np.asarray(windows, dtype=np.float32), np.asarray(centers, dtype=int), y_arr


def subsample(windows: np.ndarray, labels: np.ndarray, max_windows: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_windows is None:
        return windows, labels
    if len(windows) <= max_windows:
        return windows, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=max_windows, replace=False)
    return windows[idx], labels[idx]


class OmniAnomalyAdapter:
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 8) -> None:
        torch, nn, _, _ = require_torch()
        self.torch = torch

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
                self.mu = nn.Linear(hidden_dim, latent_dim)
                self.logvar = nn.Linear(hidden_dim, latent_dim)
                self.dec = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self, x):
                h, _ = self.gru(x)
                mu = self.mu(h)
                logvar = self.logvar(h).clamp(-4, 4)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                recon = self.dec(z)
                return recon, mu, logvar

        self.model = Model()

    def loss(self, batch):
        recon, mu, logvar = self.model(batch)
        recon_loss = ((recon - batch) ** 2).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return recon_loss + 0.01 * kl

    def score(self, batch):
        recon, _, _ = self.model(batch)
        return ((recon - batch) ** 2).mean(dim=(1, 2))


class TransformerReconAdapter:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        nhead: int = 4,
        layers: int = 1,
        tranad: bool = False,
    ) -> None:
        torch, nn, _, _ = require_torch()
        self.torch = torch
        self.tranad = tranad

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inp = nn.Linear(input_dim * (2 if tranad else 1), hidden_dim)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.05,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
                self.out = nn.Linear(hidden_dim, input_dim)

            def forward(self, x, context=None):
                if tranad:
                    if context is None:
                        context = x.new_zeros(x.shape)
                    x_in = torch.cat([x, context], dim=2)
                else:
                    x_in = x
                h = self.encoder(self.inp(x_in))
                return self.out(h)

        self.model = Model()

    def loss(self, batch):
        if self.tranad:
            first = self.model(batch)
            second = self.model(batch, self.torch.abs(first - batch).detach())
            return 0.5 * ((first - batch) ** 2).mean() + ((second - batch) ** 2).mean()
        recon = self.model(batch)
        return ((recon - batch) ** 2).mean()

    def score(self, batch):
        if self.tranad:
            first = self.model(batch)
            recon = self.model(batch, self.torch.abs(first - batch))
        else:
            recon = self.model(batch)
        return ((recon - batch) ** 2).mean(dim=(1, 2))


def fit_and_score(adapter, train_windows, test_windows, epochs: int, batch_size: int, lr: float, seed: int):
    torch, _, DataLoader, TensorDataset = require_torch()
    set_seed(torch, seed)
    optimizer = torch.optim.AdamW(adapter.model.parameters(), lr=lr, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.as_tensor(train_windows, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    adapter.model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            loss = adapter.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    adapter.model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(test_windows), batch_size):
            batch = torch.as_tensor(test_windows[start : start + batch_size], dtype=torch.float32)
            scores.append(adapter.score(batch).cpu().numpy())
    return np.concatenate(scores)


@dataclass
class MethodSpec:
    name: str
    builder: object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/contextual/pendulum_contextual_train.csv")
    parser.add_argument("--test", default="data/contextual/pendulum_contextual_test.csv")
    parser.add_argument("--feature-prefix", default="x_")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-windows", type=int, default=1200)
    parser.add_argument("--max-test-windows", type=int, default=1600)
    parser.add_argument("--seeds", nargs="+", type=int, default=[37])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["OmniAnomaly", "AnomalyTransformer", "TranAD"],
        choices=["OmniAnomaly", "AnomalyTransformer", "TranAD"],
    )
    parser.add_argument("--out", type=Path, default=Path("results/deep_tsad_smoke.json"))
    args = parser.parse_args()

    train = pd.read_csv(args.train).sort_values(["entity_id", "time"]).reset_index(drop=True)
    test = pd.read_csv(args.test).sort_values(["entity_id", "time"]).reset_index(drop=True)
    feature_cols = [col for col in train.columns if col.startswith(args.feature_prefix)]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test[feature_cols].to_numpy(dtype=float))
    y_test_rows = test["is_anomaly"].to_numpy(dtype=int)

    train_windows, _, train_y = make_windows(x_train, train["entity_id"].to_numpy(), None, args.window)
    test_windows, _, test_y = make_windows(x_test, test["entity_id"].to_numpy(), y_test_rows, args.window)
    train_windows, train_y = subsample(train_windows, train_y, args.max_train_windows, 11)
    test_windows, test_y = subsample(test_windows, test_y, args.max_test_windows, 13)

    specs = [
        MethodSpec("OmniAnomaly-smoke", lambda d: OmniAnomalyAdapter(d)),
        MethodSpec("AnomalyTransformer-smoke", lambda d: TransformerReconAdapter(d, tranad=False)),
        MethodSpec("TranAD-smoke", lambda d: TransformerReconAdapter(d, tranad=True)),
    ]
    wanted = {f"{name}-smoke" for name in args.methods}
    specs = [spec for spec in specs if spec.name in wanted]

    rows = []
    for spec in specs:
        for seed in args.seeds:
            start = time.perf_counter()
            adapter = spec.builder(train_windows.shape[2])
            try:
                scores = fit_and_score(adapter, train_windows, test_windows, args.epochs, args.batch_size, 1e-3, seed)
                elapsed = time.perf_counter() - start
                row = {
                    "method": spec.name,
                    "seed": seed,
                    "seconds": elapsed,
                    "auroc": float(roc_auc_score(test_y, scores)) if test_y.sum() and test_y.sum() < len(test_y) else None,
                    "auprc": float(average_precision_score(test_y, scores)) if test_y.sum() else None,
                    "status": "ok",
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "method": spec.name,
                    "seed": seed,
                    "seconds": time.perf_counter() - start,
                    "status": "failed",
                    "error": str(exc),
                }
            rows.append(row)
            print(row, flush=True)

    summary = {}
    df = pd.DataFrame(rows)
    for method, group in df[df["status"] == "ok"].groupby("method"):
        summary[method] = {
            "runs": int(len(group)),
            "mean_seconds": float(group["seconds"].mean()),
            "std_seconds": float(group["seconds"].std(ddof=0)),
            "mean_auroc": float(group["auroc"].mean()),
            "std_auroc": float(group["auroc"].std(ddof=0)),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"args": vars(args) | {"out": str(args.out)}, "rows": rows, "summary": summary}, indent=2) + "\n")
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
