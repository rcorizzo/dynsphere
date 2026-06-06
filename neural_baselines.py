#!/usr/bin/env python3
"""Optional PyTorch baselines for the pendulum benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pendulum_detector import MultiConceptPendulumDetector


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for neural baselines. Install requirements or run:\n"
            "  uv run --with torch --with numpy --with pandas --with scikit-learn --with pyod python run_benchmarks.py --include-neural"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _set_seed(torch, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _robust_z(scores: np.ndarray, median: float, scale: float) -> np.ndarray:
    return np.maximum(0.0, (scores - median) / max(scale, 1e-6))


def _fit_score_scale(scores: np.ndarray) -> tuple[float, float]:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    return median, max(1.4826 * mad, 1e-6)


def _mlp(nn, dims: list[int], final_activation: bool = False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _train_autoencoder(
    x_train: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    random_state: int,
):
    torch, nn, DataLoader, TensorDataset = _require_torch()
    _set_seed(torch, random_state)

    class Autoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = _mlp(nn, [input_dim, hidden_dim, latent_dim])
            self.decoder = _mlp(nn, [latent_dim, hidden_dim, input_dim])

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    model = Autoencoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    tensor = torch.as_tensor(x_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            recon, _ = model(batch)
            loss = ((recon - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, torch


def _ae_scores(model, torch, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
            recon, _ = model(batch)
            out.append(((recon - batch) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)


class StaticAutoencoderDetector:
    """Row-wise reconstruction baseline."""

    def __init__(
        self,
        hidden_dim: int = 48,
        latent_dim: int = 6,
        epochs: int = 35,
        batch_size: int = 512,
        lr: float = 1e-3,
        random_state: int = 37,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    def fit(self, x: np.ndarray) -> "StaticAutoencoderDetector":
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        self.model_, self.torch_ = _train_autoencoder(
            xs,
            xs.shape[1],
            self.hidden_dim,
            self.latent_dim,
            self.epochs,
            self.batch_size,
            self.lr,
            self.random_state,
        )
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        xs = self.scaler_.transform(x)
        return _ae_scores(self.model_, self.torch_, xs, self.batch_size)


def _window_matrix(x: np.ndarray, entity_ids: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    centers = []
    width = 2 * radius + 1
    for entity_id in np.unique(entity_ids):
        idx = np.flatnonzero(entity_ids == entity_id)
        seq = x[idx]
        if len(seq) < width:
            continue
        padded = np.pad(seq, ((radius, radius), (0, 0)), mode="edge")
        for i, row_idx in enumerate(idx):
            windows.append(padded[i : i + width].reshape(-1))
            centers.append(row_idx)
    return np.asarray(windows, dtype=float), np.asarray(centers, dtype=int)


class SequenceAutoencoderDetector:
    """Window reconstruction baseline that can notice temporal inconsistency."""

    def __init__(
        self,
        radius: int = 8,
        hidden_dim: int = 96,
        latent_dim: int = 10,
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        random_state: int = 37,
    ) -> None:
        self.radius = radius
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    def fit(self, x: np.ndarray, entity_ids: np.ndarray) -> "SequenceAutoencoderDetector":
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        windows, _ = _window_matrix(xs, entity_ids, self.radius)
        self.model_, self.torch_ = _train_autoencoder(
            windows,
            windows.shape[1],
            self.hidden_dim,
            self.latent_dim,
            self.epochs,
            self.batch_size,
            self.lr,
            self.random_state,
        )
        return self

    def decision_function(self, x: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
        xs = self.scaler_.transform(x)
        windows, centers = _window_matrix(xs, entity_ids, self.radius)
        scores = np.zeros(len(x), dtype=float)
        scores[centers] = _ae_scores(self.model_, self.torch_, windows, self.batch_size)
        return scores


class WindowKNNDetector:
    """kNN distance on temporal windows.

    This is a simple non-neural temporal baseline: it borrows the locality idea
    of KNN/LOF but compares short multivariate windows instead of individual
    rows.
    """

    def __init__(self, radius: int = 8, n_neighbors: int = 15) -> None:
        self.radius = radius
        self.n_neighbors = n_neighbors

    def fit(self, x: np.ndarray, entity_ids: np.ndarray) -> "WindowKNNDetector":
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        windows, _ = _window_matrix(xs, entity_ids, self.radius)
        self.n_neighbors_ = min(self.n_neighbors, len(windows))
        self.model_ = NearestNeighbors(n_neighbors=self.n_neighbors_)
        self.model_.fit(windows)
        return self

    def decision_function(self, x: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
        xs = self.scaler_.transform(x)
        windows, centers = _window_matrix(xs, entity_ids, self.radius)
        scores = np.zeros(len(x), dtype=float)
        if len(windows) == 0:
            return scores
        distances, _ = self.model_.kneighbors(windows)
        scores[centers] = distances[:, -1]
        return scores


class ForecastingMLPDetector:
    """Predict the current row from its recent past and score prediction error."""

    def __init__(
        self,
        lag: int = 8,
        hidden_dim: int = 96,
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        random_state: int = 37,
    ) -> None:
        self.lag = lag
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    def _supervised_windows(self, x: np.ndarray, entity_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = []
        targets = []
        centers = []
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            seq = x[idx]
            if len(seq) <= self.lag:
                continue
            for i in range(self.lag, len(seq)):
                inputs.append(seq[i - self.lag : i].reshape(-1))
                targets.append(seq[i])
                centers.append(idx[i])
        return np.asarray(inputs, dtype=float), np.asarray(targets, dtype=float), np.asarray(centers, dtype=int)

    def fit(self, x: np.ndarray, entity_ids: np.ndarray) -> "ForecastingMLPDetector":
        torch, nn, DataLoader, TensorDataset = _require_torch()
        _set_seed(torch, self.random_state)
        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)
        inputs, targets, _ = self._supervised_windows(xs, entity_ids)

        self.model_ = _mlp(nn, [inputs.shape[1], self.hidden_dim, targets.shape[1]])
        self.torch_ = torch
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(inputs, dtype=torch.float32),
                torch.as_tensor(targets, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                pred = self.model_(batch_x)
                loss = ((pred - batch_y) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def decision_function(self, x: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
        xs = self.scaler_.transform(x)
        inputs, targets, centers = self._supervised_windows(xs, entity_ids)
        scores = np.zeros(len(x), dtype=float)
        if len(inputs) == 0:
            return scores
        self.model_.eval()
        out = []
        with self.torch_.no_grad():
            for start in range(0, len(inputs), self.batch_size):
                batch = self.torch_.as_tensor(inputs[start : start + self.batch_size], dtype=self.torch_.float32)
                pred = self.model_(batch).cpu().numpy()
                out.append(((pred - targets[start : start + self.batch_size]) ** 2).mean(axis=1))
        scores[centers] = np.concatenate(out)
        return scores


@dataclass
class _DeepSVDDState:
    model: object
    torch: object
    centers: object


class MultiSphereDeepSVDDDetector:
    """Deep SVDD-style multi-center baseline.

    If `concept_labels` are supplied during fit, this is the oracle multi-class
    version. Otherwise it uses KMeans pseudo-labels on standardized training
    rows, which is the fairer unsupervised variant.
    """

    def __init__(
        self,
        n_concepts: int = 4,
        hidden_dim: int = 48,
        embedding_dim: int = 6,
        pretrain_epochs: int = 20,
        svdd_epochs: int = 45,
        batch_size: int = 512,
        lr: float = 1e-3,
        random_state: int = 37,
    ) -> None:
        self.n_concepts = n_concepts
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pretrain_epochs = pretrain_epochs
        self.svdd_epochs = svdd_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    def fit(self, x: np.ndarray, concept_labels: np.ndarray | None = None) -> "MultiSphereDeepSVDDDetector":
        torch, nn, DataLoader, TensorDataset = _require_torch()
        _set_seed(torch, self.random_state)

        self.scaler_ = StandardScaler()
        xs = self.scaler_.fit_transform(x)

        ae, _ = _train_autoencoder(
            xs,
            xs.shape[1],
            self.hidden_dim,
            self.embedding_dim,
            self.pretrain_epochs,
            self.batch_size,
            self.lr,
            self.random_state,
        )
        encoder = ae.encoder

        with torch.no_grad():
            embeddings = encoder(torch.as_tensor(xs, dtype=torch.float32)).cpu().numpy()

        if concept_labels is None:
            labels = KMeans(n_clusters=self.n_concepts, n_init=20, random_state=self.random_state).fit_predict(xs)
        else:
            _, labels = np.unique(concept_labels, return_inverse=True)
        self.n_concepts_ = int(labels.max()) + 1

        centers = []
        for concept in range(self.n_concepts_):
            idx = labels == concept
            if idx.sum() == 0:
                centers.append(np.zeros(self.embedding_dim, dtype=np.float32))
            else:
                centers.append(embeddings[idx].mean(axis=0).astype(np.float32))
        centers_t = torch.as_tensor(np.vstack(centers), dtype=torch.float32)

        tensor_x = torch.as_tensor(xs, dtype=torch.float32)
        tensor_y = torch.as_tensor(labels, dtype=torch.long)
        loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=self.lr, weight_decay=1e-5)
        encoder.train()
        for _ in range(self.svdd_epochs):
            for batch_x, batch_y in loader:
                z = encoder(batch_x)
                center = centers_t[batch_y]
                loss = ((z - center) ** 2).sum(dim=1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                full_z = encoder(tensor_x)
                for concept in range(self.n_concepts_):
                    idx = tensor_y == concept
                    if bool(idx.any()):
                        centers_t[concept] = full_z[idx].mean(dim=0)

        self.state_ = _DeepSVDDState(model=encoder, torch=torch, centers=centers_t)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        xs = self.scaler_.transform(x)
        torch = self.state_.torch
        model = self.state_.model
        centers = self.state_.centers
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(xs), self.batch_size):
                batch = torch.as_tensor(xs[start : start + self.batch_size], dtype=torch.float32)
                z = model(batch)
                dist = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)
                scores.append(dist.min(dim=1).values.cpu().numpy())
        return np.concatenate(scores)


class DynSphereAEDetector:
    """DynSphereAE: robust fusion of DynSphere and sequence reconstruction.

    Both components are trained only on normal trajectories. Their training
    scores are robustly normalized before being combined, so the weight is not
    dominated by raw score scale.
    """

    def __init__(
        self,
        ae_weight: float = 1.0,
        pendulum_weight: float = 0.5,
        random_state: int = 37,
    ) -> None:
        self.ae_weight = ae_weight
        self.pendulum_weight = pendulum_weight
        self.random_state = random_state

    def fit(self, x: np.ndarray, entity_ids: np.ndarray, times: np.ndarray | None = None) -> "DynSphereAEDetector":
        self.sequence_ae_ = SequenceAutoencoderDetector(random_state=self.random_state)
        self.sequence_ae_.fit(x, entity_ids)
        ae_train = self.sequence_ae_.decision_function(x, entity_ids)
        self.ae_median_, self.ae_scale_ = _fit_score_scale(ae_train)

        self.pendulum_ = MultiConceptPendulumDetector(random_state=self.random_state)
        self.pendulum_.fit(x, entity_ids, times)
        pendulum_train = self.pendulum_.decision_function(x, entity_ids, times)
        self.pendulum_median_, self.pendulum_scale_ = _fit_score_scale(pendulum_train)
        return self

    def decision_function(self, x: np.ndarray, entity_ids: np.ndarray, times: np.ndarray | None = None) -> np.ndarray:
        ae_scores = self.sequence_ae_.decision_function(x, entity_ids)
        pendulum_scores = self.pendulum_.decision_function(x, entity_ids, times)
        return (
            self.ae_weight * _robust_z(ae_scores, self.ae_median_, self.ae_scale_)
            + self.pendulum_weight * _robust_z(pendulum_scores, self.pendulum_median_, self.pendulum_scale_)
        )


class DynSpherePlusPlusDetector:
    """DynSphere++: score-level ensemble of DynSphere, windows, and detectors.

    The real-world benchmark runs show that different detector families can
    dominate different source families. This variant makes that complementarity
    explicit via robust score fusion.
    """

    def __init__(
        self,
        pendulum_weight: float = 0.35,
        sequence_weight: float = 1.0,
        window_knn_weight: float = 0.2,
        static_weight: float = 0.15,
        random_state: int = 37,
    ) -> None:
        self.pendulum_weight = pendulum_weight
        self.sequence_weight = sequence_weight
        self.window_knn_weight = window_knn_weight
        self.static_weight = static_weight
        self.random_state = random_state

    def fit(self, x: np.ndarray, entity_ids: np.ndarray, times: np.ndarray | None = None) -> "DynSpherePlusPlusDetector":
        del times
        self.components_: list[tuple[float, object, str, float, float]] = []

        if self.pendulum_weight > 0:
            pendulum = MultiConceptPendulumDetector(random_state=self.random_state)
            pendulum.fit(x, entity_ids)
            scores = pendulum.decision_function(x, entity_ids)
            median, scale = _fit_score_scale(scores)
            self.components_.append((self.pendulum_weight, pendulum, "entity", median, scale))

        if self.sequence_weight > 0:
            sequence = SequenceAutoencoderDetector(random_state=self.random_state)
            sequence.fit(x, entity_ids)
            scores = sequence.decision_function(x, entity_ids)
            median, scale = _fit_score_scale(scores)
            self.components_.append((self.sequence_weight, sequence, "entity", median, scale))

        if self.window_knn_weight > 0:
            window_knn = WindowKNNDetector()
            window_knn.fit(x, entity_ids)
            scores = window_knn.decision_function(x, entity_ids)
            median, scale = _fit_score_scale(scores)
            self.components_.append((self.window_knn_weight, window_knn, "entity", median, scale))

        if self.static_weight > 0:
            self.static_scaler_ = StandardScaler()
            xs = self.static_scaler_.fit_transform(x)
            static = IsolationForest(n_estimators=250, random_state=self.random_state)
            static.fit(xs)
            scores = -static.score_samples(xs)
            median, scale = _fit_score_scale(scores)
            self.components_.append((self.static_weight, static, "static", median, scale))
        if not self.components_:
            raise ValueError("At least one enhanced ensemble component must have positive weight.")
        return self

    def decision_function(self, x: np.ndarray, entity_ids: np.ndarray, times: np.ndarray | None = None) -> np.ndarray:
        del times
        total = np.zeros(len(x), dtype=float)
        for weight, model, mode, median, scale in self.components_:
            if mode == "static":
                scores = -model.score_samples(self.static_scaler_.transform(x))
            else:
                scores = model.decision_function(x, entity_ids)
            total += weight * _robust_z(scores, median, scale)
        return total
