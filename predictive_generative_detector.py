#!/usr/bin/env python3
"""Causal switching predictive likelihood for time-series anomaly detection.

Unlike score fusion, this detector evaluates one density.  A causal encoder
summarizes observations through time ``t-1``; every dynamical mode advances
that latent state; and a shared decoder maps the candidate successor to a
Gaussian distribution for ``x_t``.  The anomaly score is

    -log sum_m p(m_t | m_{t-1}) p(x_t | z_hat_t^(m)).

Dynamics therefore changes the observation distribution itself.  There is no
separate reconstruction score, dynamics score, or manually selected weight.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


EPS = 1e-10
LOG_2PI = float(np.log(2.0 * np.pi))


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for PredictiveGenerativeDetector. Install "
            "the repository requirements before using this model."
        ) from exc
    return torch, nn, functional


class PredictiveGenerativeDetector(BaseEstimator):
    """Switching next-observation likelihood with learned causal dynamics."""

    def __init__(
        self,
        geometry: str = "euclidean",
        latent_dim: int = 6,
        hidden_dim: int = 48,
        n_modes: int = 4,
        sequence_length: int = 128,
        epochs: int = 15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip: float = 5.0,
        mode_specific_decoders: bool = False,
        contrastive_weight: float = 0.0,
        contrastive_margin: float = 1.0,
        diversity_weight: float = 0.0,
        diversity_margin: float = 0.5,
        balance_weight: float = 0.0,
        persistence_weight: float = 0.0,
        calibration_fraction: float = 0.2,
        calibrate: bool = True,
        random_state: int = 37,
    ) -> None:
        self.geometry = geometry
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_modes = n_modes
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.mode_specific_decoders = mode_specific_decoders
        self.contrastive_weight = contrastive_weight
        self.contrastive_margin = contrastive_margin
        self.diversity_weight = diversity_weight
        self.diversity_margin = diversity_margin
        self.balance_weight = balance_weight
        self.persistence_weight = persistence_weight
        self.calibration_fraction = calibration_fraction
        self.calibrate = calibrate
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> "PredictiveGenerativeDetector":
        self._validate_configuration()
        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)
        torch, nn, functional = _require_torch()
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.set_num_threads(1)
        self.torch_ = torch
        self.functional_ = functional

        fit_mask = self._temporal_fit_mask(entity_ids)
        self.fit_mask_ = fit_mask.copy()
        self.feature_scaler_ = StandardScaler().fit(x[fit_mask])
        xs = self.feature_scaler_.transform(x).astype(np.float32)
        self.dt_scale_ = self._fit_dt_scale(entity_ids, times, fit_mask)
        self.model_ = self._build_model(nn, x.shape[1])
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        chunks = self._fit_chunks(entity_ids, fit_mask)
        if not chunks:
            raise ValueError("No training sequence contains at least two rows.")
        rng = np.random.default_rng(self.random_state)
        self.training_loss_: list[float] = []
        self.model_.train()
        for _ in range(self.epochs):
            epoch_losses = []
            for position in rng.permutation(len(chunks)):
                idx = chunks[int(position)]
                sequence = torch.as_tensor(xs[idx], dtype=torch.float32)
                dt = torch.as_tensor(self._normalized_dt(times[idx]), dtype=torch.float32)
                factors = self.model_.predictive_factors(sequence, dt)
                loss = factors["predictive_nll"][1:].mean()
                if self.contrastive_weight > 0.0 and len(sequence) > 3:
                    shift = int(rng.integers(1, len(sequence) - 1))
                    negative_targets = torch.roll(sequence[1:], shifts=shift, dims=0)
                    negative = self.model_.predictive_factors(
                        sequence,
                        dt,
                        targets=negative_targets,
                    )
                    ranking = torch.relu(
                        self.contrastive_margin
                        + factors["predictive_nll"][1:]
                        - negative["predictive_nll"][1:]
                    ).mean()
                    loss = loss + self.contrastive_weight * ranking
                if self.diversity_weight > 0.0 and self.n_modes > 1:
                    successors = factors["successors"]
                    pairwise = torch.cdist(successors, successors)
                    upper = torch.triu(
                        torch.ones(
                            self.n_modes,
                            self.n_modes,
                            dtype=torch.bool,
                            device=sequence.device,
                        ),
                        diagonal=1,
                    )
                    diversity = torch.relu(
                        self.diversity_margin - pairwise[:, upper]
                    ).square().mean()
                    loss = loss + self.diversity_weight * diversity
                if self.balance_weight > 0.0 and self.n_modes > 1:
                    usage = factors["mode_probabilities"][1:].mean(dim=0)
                    balance = torch.sum(usage * torch.log(usage * self.n_modes + EPS))
                    loss = loss + self.balance_weight * balance
                if self.persistence_weight > 0.0 and self.n_modes > 1:
                    transition = torch.softmax(self.model_.transition_logits, dim=1)
                    persistence = -torch.log(torch.diagonal(transition) + EPS).mean()
                    loss = loss + self.persistence_weight * persistence
                if not torch.isfinite(loss):
                    raise ValueError("Predictive likelihood training produced a non-finite loss.")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.training_loss_.append(float(np.mean(epoch_losses)))

        components = self._raw_components(xs, entity_ids, times)
        calibration = components["predictive_nll"][~fit_mask]
        calibration = np.sort(calibration[np.isfinite(calibration)])
        if self.calibrate and len(calibration) < 2:
            raise ValueError("Calibration produced fewer than two finite scores.")
        if not self.calibrate:
            calibration = np.sort(components["predictive_nll"][fit_mask])
        self.calibration_scores_ = calibration
        return self

    def decision_function(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        components = self.decision_components(x, entity_ids, times)
        return components["calibrated_score"] if self.calibrate else components[
            "predictive_nll"
        ]

    def decision_components(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)
        xs = self.feature_scaler_.transform(x).astype(np.float32)
        components = self._raw_components(xs, entity_ids, times)
        components["calibrated_score"] = self._calibrate(components["predictive_nll"])
        for entity_id in np.unique(entity_ids):
            first = np.flatnonzero(entity_ids == entity_id)[0]
            components["calibrated_score"][first] = 0.0
        return components

    def _build_model(self, nn: Any, input_dim: int):
        torch = self.torch_
        functional = self.functional_
        latent_dim = self.latent_dim
        hidden_dim = self.hidden_dim
        n_modes = self.n_modes
        geometry = self.geometry
        mode_specific_decoders = self.mode_specific_decoders

        class PredictiveMixtureModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
                self.to_latent = nn.Linear(hidden_dim, latent_dim)
                self.fields = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(latent_dim + 1, hidden_dim),
                            nn.Tanh(),
                            nn.Linear(hidden_dim, latent_dim),
                        )
                        for _ in range(n_modes)
                    ]
                )
                if mode_specific_decoders:
                    self.decoders = nn.ModuleList(
                        [
                            nn.Sequential(
                                nn.Linear(latent_dim, hidden_dim),
                                nn.SiLU(),
                                nn.Linear(hidden_dim, input_dim),
                            )
                            for _ in range(n_modes)
                        ]
                    )
                else:
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, input_dim),
                    )
                self.observation_scale = nn.Parameter(torch.zeros(n_modes, input_dim))
                transition = torch.eye(n_modes) * 1.5
                transition += 0.05 * torch.randn(n_modes, n_modes)
                self.transition_logits = nn.Parameter(transition)
                self.initial_logits = nn.Parameter(torch.zeros(n_modes))

            @staticmethod
            def bounded_log_variance(raw):
                return -6.0 + 9.0 * torch.sigmoid(raw)

            def encode(self, sequence):
                encoded, _ = self.encoder(sequence.unsqueeze(0))
                latent = self.to_latent(encoded.squeeze(0))
                if geometry == "sphere":
                    latent = functional.normalize(latent, dim=1, eps=1e-8)
                return latent

            def successor_states(self, previous, dt):
                field_input = torch.cat([previous, torch.log1p(dt)], dim=1)
                velocity = torch.stack(
                    [field(field_input) for field in self.fields],
                    dim=1,
                )
                if geometry == "euclidean":
                    return previous[:, None, :] + dt[:, None, :] * velocity
                base = previous[:, None, :].expand_as(velocity)
                tangent = velocity - (velocity * base).sum(dim=2, keepdim=True) * base
                step = dt[:, None, :] * tangent
                norm = torch.linalg.vector_norm(step, dim=2, keepdim=True)
                direction = step / torch.clamp(norm, min=1e-8)
                destination = torch.cos(norm) * base + torch.sin(norm) * direction
                return functional.normalize(destination, dim=2, eps=1e-8)

            def decode_successors(self, successors):
                if mode_specific_decoders:
                    return torch.stack(
                        [
                            decoder(successors[:, mode, :])
                            for mode, decoder in enumerate(self.decoders)
                        ],
                        dim=1,
                    )
                return self.decoder(successors.reshape(-1, latent_dim)).reshape(
                    len(successors),
                    n_modes,
                    input_dim,
                )

            def predictive_factors(self, sequence, dt, targets=None):
                latent = self.encode(sequence)
                predictive_nll = [torch.zeros((), device=sequence.device)]
                entropy = []
                selected_mode = [torch.zeros((), device=sequence.device)]
                log_posterior = functional.log_softmax(self.initial_logits, dim=0)
                entropy.append(-(log_posterior.exp() * log_posterior).sum())
                mode_probabilities = [log_posterior.exp()]
                successors = torch.empty(
                    (0, n_modes, latent_dim),
                    device=sequence.device,
                )
                means = torch.empty((0, n_modes, input_dim), device=sequence.device)
                if len(sequence) > 1:
                    successors = self.successor_states(latent[:-1], dt[1:])
                    means = self.decode_successors(successors)
                    observed_targets = sequence[1:] if targets is None else targets
                    if len(observed_targets) != len(sequence) - 1:
                        raise ValueError("targets must contain one row per transition")
                    log_variance = self.bounded_log_variance(self.observation_scale)
                    residual = observed_targets[:, None, :] - means
                    emission = -0.5 * torch.sum(
                        residual**2 / torch.exp(log_variance[None, :, :])
                        + log_variance[None, :, :]
                        + LOG_2PI,
                        dim=2,
                    )
                    log_transition = functional.log_softmax(self.transition_logits, dim=1)
                    for row in range(len(sequence) - 1):
                        log_prior = torch.logsumexp(
                            log_posterior[:, None] + log_transition,
                            dim=0,
                        )
                        log_joint = log_prior + emission[row]
                        predictive_log_prob = torch.logsumexp(log_joint, dim=0)
                        log_posterior = log_joint - predictive_log_prob
                        predictive_nll.append(-predictive_log_prob)
                        entropy.append(-(log_posterior.exp() * log_posterior).sum())
                        selected_mode.append(torch.argmax(log_posterior).to(sequence.dtype))
                        mode_probabilities.append(log_posterior.exp())
                return {
                    "predictive_nll": torch.stack(predictive_nll),
                    "mode_entropy": torch.stack(entropy),
                    "selected_mode": torch.stack(selected_mode),
                    "mode_probabilities": torch.stack(mode_probabilities),
                    "successors": successors,
                    "predicted_means": means,
                    "latent": latent,
                }

        return PredictiveMixtureModel()

    def _raw_components(
        self,
        xs: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray,
    ) -> dict[str, np.ndarray]:
        output = {
            "predictive_nll": np.zeros(len(xs), dtype=float),
            "mode_entropy": np.zeros(len(xs), dtype=float),
            "selected_mode": np.zeros(len(xs), dtype=float),
        }
        self.model_.eval()
        with self.torch_.no_grad():
            for entity_id in np.unique(entity_ids):
                idx = np.flatnonzero(entity_ids == entity_id)
                sequence = self.torch_.as_tensor(xs[idx], dtype=self.torch_.float32)
                dt = self.torch_.as_tensor(
                    self._normalized_dt(times[idx]),
                    dtype=self.torch_.float32,
                )
                factors = self.model_.predictive_factors(sequence, dt)
                for name in output:
                    output[name][idx] = factors[name].cpu().numpy()
        return output

    def _temporal_fit_mask(self, entity_ids: np.ndarray) -> np.ndarray:
        mask = np.ones(len(entity_ids), dtype=bool)
        if not self.calibrate:
            return mask
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            calibration_size = max(2, int(math.ceil(len(idx) * self.calibration_fraction)))
            if len(idx) - calibration_size < 8:
                raise ValueError("Each entity requires at least ten rows for calibration.")
            mask[idx[-calibration_size:]] = False
        return mask

    def _fit_chunks(self, entity_ids: np.ndarray, fit_mask: np.ndarray) -> list[np.ndarray]:
        chunks = []
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero((entity_ids == entity_id) & fit_mask)
            for start in range(0, len(idx), self.sequence_length):
                chunk = idx[start : start + self.sequence_length]
                if len(chunk) >= 2:
                    chunks.append(chunk)
        return chunks

    @staticmethod
    def _fit_dt_scale(
        entity_ids: np.ndarray,
        times: np.ndarray,
        fit_mask: np.ndarray,
    ) -> float:
        differences = []
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero((entity_ids == entity_id) & fit_mask)
            if len(idx) > 1:
                differences.append(np.diff(times[idx]))
        if not differences:
            return 1.0
        return max(float(np.median(np.concatenate(differences))), EPS)

    def _normalized_dt(self, times: np.ndarray) -> np.ndarray:
        dt = np.ones((len(times), 1), dtype=np.float32)
        if len(times) > 1:
            dt[1:, 0] = np.diff(times) / self.dt_scale_
        return dt

    def _calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        reference = self.calibration_scores_
        if len(reference) == 0:
            return raw_scores.copy()
        median = float(np.median(reference))
        mad = float(np.median(np.abs(reference - median)))
        scale = max(1.4826 * mad, float(np.std(reference)), 1e-6)
        return (raw_scores - median) / scale

    def _validate_configuration(self) -> None:
        if self.geometry not in {"euclidean", "sphere"}:
            raise ValueError("geometry must be 'euclidean' or 'sphere'.")
        if self.latent_dim < 2 or self.hidden_dim < 2:
            raise ValueError("latent_dim and hidden_dim must be at least two.")
        if self.n_modes < 1 or self.sequence_length < 2 or self.epochs < 1:
            raise ValueError("n_modes, sequence_length, and epochs must be positive.")
        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("learning_rate and gradient_clip must be positive.")
        regularization_values = [
            self.contrastive_weight,
            self.contrastive_margin,
            self.diversity_weight,
            self.diversity_margin,
            self.balance_weight,
            self.persistence_weight,
        ]
        if any(value < 0.0 for value in regularization_values):
            raise ValueError("identifiability weights and margins must be non-negative.")
        if not 0.0 <= self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in [0, 0.5).")
        if self.calibrate and self.calibration_fraction <= 0.0:
            raise ValueError("calibration_fraction must be positive when calibrating.")

    @staticmethod
    def _validate_inputs(
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        entity_ids = np.asarray(entity_ids)
        if x.ndim != 2 or len(x) != len(entity_ids):
            raise ValueError("x must be two-dimensional and match entity_ids.")
        if len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("x must contain finite observations.")
        if times is None:
            times_array = np.empty(len(x), dtype=float)
            for entity_id in np.unique(entity_ids):
                idx = np.flatnonzero(entity_ids == entity_id)
                times_array[idx] = np.arange(len(idx), dtype=float)
            times = times_array
        else:
            times = np.asarray(times, dtype=float)
            if len(times) != len(x):
                raise ValueError("times must match x.")
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            if len(idx) > 1 and np.any(np.diff(times[idx]) <= 0.0):
                raise ValueError("times must be strictly increasing within each entity.")
        return x, entity_ids, times
