#!/usr/bin/env python3
"""Unified switching latent-state likelihood for contextual anomaly detection.

The detector replaces score-level fusion with one probabilistic construction:

    p(x_t, z_t, m_t | H_{t-1})
      = p(x_t | z_t) p(z_t | z_{t-1}, m_t) p(m_t | m_{t-1}).

A causal GRU infers the latent path from present and past observations.  The
decoder, nonlinear mode-conditioned transition fields, Gaussian variances, and
Markov mode process are optimized jointly on normal sequences.  The deployed
joint anomaly score is the negative one-step predictive log likelihood.  Its
observation and dynamics factors are exposed for controlled ablations, not
combined through manually selected detector weights.

The inferred path makes this an approximate latent-state likelihood rather
than an exact marginal likelihood over all possible latent paths.  Keeping that
distinction explicit avoids overstating what the implementation optimizes.
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
            "PyTorch is required for UnifiedGenerativeDetector. Install the "
            "repository requirements before using this model."
        ) from exc
    return torch, nn, functional


class UnifiedGenerativeDetector(BaseEstimator):
    """Causal observation--dynamics detector with soft switching modes.

    Parameters
    ----------
    geometry:
        ``"euclidean"`` learns an unconstrained latent state. ``"sphere"``
        normalizes the learned state and advances it through a spherical
        exponential map. Geometry is therefore a matched experimental choice.
    score_factor:
        ``"joint"`` is the proposed score. ``"observation"`` and
        ``"dynamics"`` expose exact likelihood-factor ablations from the same
        jointly trained model; they do not train or fuse auxiliary detectors.
    """

    def __init__(
        self,
        geometry: str = "euclidean",
        score_factor: str = "joint",
        latent_dim: int = 6,
        hidden_dim: int = 48,
        n_modes: int = 4,
        sequence_length: int = 128,
        epochs: int = 15,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip: float = 5.0,
        calibration_fraction: float = 0.2,
        calibrate: bool = True,
        random_state: int = 37,
    ) -> None:
        self.geometry = geometry
        self.score_factor = score_factor
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_modes = n_modes
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.calibration_fraction = calibration_fraction
        self.calibrate = calibrate
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> "UnifiedGenerativeDetector":
        self._validate_configuration()
        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)
        torch, nn, functional = _require_torch()
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.set_num_threads(1)
        self.torch_ = torch
        self.functional_ = functional

        fit_mask = self._temporal_fit_mask(entity_ids)
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
        self.training_loss_ = []
        self.model_.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(chunks))
            epoch_losses = []
            for position in order:
                idx = chunks[int(position)]
                batch_x = torch.as_tensor(xs[idx], dtype=torch.float32)
                batch_dt = torch.as_tensor(
                    self._normalized_dt(times[idx]),
                    dtype=torch.float32,
                )
                factors = self.model_.likelihood_factors(batch_x, batch_dt)
                loss = factors["joint_nll"].mean()
                if not torch.isfinite(loss):
                    raise ValueError("Unified likelihood training produced a non-finite loss.")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.training_loss_.append(float(np.mean(epoch_losses)))

        raw_components = self._raw_components(xs, entity_ids, times)
        calibration_mask = ~fit_mask
        self.calibration_scores_: dict[str, np.ndarray] = {}
        for factor in ("joint", "observation", "dynamics"):
            reference = raw_components[f"{factor}_nll"][calibration_mask]
            reference = np.sort(reference[np.isfinite(reference)])
            if self.calibrate and len(reference) < 2:
                raise ValueError("Calibration produced fewer than two finite scores.")
            if not self.calibrate:
                reference = np.sort(raw_components[f"{factor}_nll"][fit_mask])
            self.calibration_scores_[factor] = reference
        return self

    def decision_function(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
        score_factor: str | None = None,
    ) -> np.ndarray:
        factor = self.score_factor if score_factor is None else score_factor
        if factor not in {"joint", "observation", "dynamics"}:
            raise ValueError("score_factor must be joint, observation, or dynamics.")
        components = self.decision_components(x, entity_ids, times)
        raw = components[f"{factor}_nll"]
        return self._calibrate(raw, factor) if self.calibrate else raw

    def decision_components(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)
        xs = self.feature_scaler_.transform(x).astype(np.float32)
        components = self._raw_components(xs, entity_ids, times)
        raw = components[f"{self.score_factor}_nll"]
        components["calibrated_score"] = self._calibrate(raw, self.score_factor)
        return components

    def _build_model(self, nn: Any, input_dim: int):
        torch = self.torch_
        functional = self.functional_
        latent_dim = self.latent_dim
        hidden_dim = self.hidden_dim
        n_modes = self.n_modes
        geometry = self.geometry

        class SwitchingStateSpaceModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
                self.to_latent = nn.Linear(hidden_dim, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, input_dim),
                )
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
                self.observation_scale = nn.Parameter(torch.zeros(input_dim))
                self.dynamics_scale = nn.Parameter(torch.zeros(n_modes, latent_dim))
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

            def transition_means(self, previous, dt):
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

            def likelihood_factors(self, sequence, dt):
                latent = self.encode(sequence)
                reconstruction = self.decoder(latent)
                obs_log_var = self.bounded_log_variance(self.observation_scale)
                obs_log_prob = -0.5 * torch.sum(
                    (sequence - reconstruction) ** 2 / torch.exp(obs_log_var)
                    + obs_log_var
                    + LOG_2PI,
                    dim=1,
                )

                transition_log_prob = torch.empty(
                    (max(len(sequence) - 1, 0), n_modes),
                    device=sequence.device,
                )
                if len(sequence) > 1:
                    means = self.transition_means(latent[:-1], dt[1:])
                    dyn_log_var = self.bounded_log_variance(self.dynamics_scale)
                    residual = latent[1:, None, :] - means
                    transition_log_prob = -0.5 * torch.sum(
                        residual**2 / torch.exp(dyn_log_var[None, :, :])
                        + dyn_log_var[None, :, :]
                        + LOG_2PI,
                        dim=2,
                    )

                log_transition = functional.log_softmax(self.transition_logits, dim=1)
                log_posterior = functional.log_softmax(self.initial_logits, dim=0)
                joint_nll = [-obs_log_prob[0]]
                dynamics_nll = [torch.zeros((), device=sequence.device)]
                entropy = [-(log_posterior.exp() * log_posterior).sum()]
                for row in range(1, len(sequence)):
                    log_prior = torch.logsumexp(
                        log_posterior[:, None] + log_transition,
                        dim=0,
                    )
                    mode_joint = log_prior + transition_log_prob[row - 1]
                    dynamics_log_prob = torch.logsumexp(mode_joint, dim=0)
                    log_posterior = mode_joint - dynamics_log_prob
                    dynamics_nll.append(-dynamics_log_prob)
                    joint_nll.append(-obs_log_prob[row] - dynamics_log_prob)
                    entropy.append(-(log_posterior.exp() * log_posterior).sum())
                return {
                    "joint_nll": torch.stack(joint_nll),
                    "observation_nll": -obs_log_prob,
                    "dynamics_nll": torch.stack(dynamics_nll),
                    "mode_entropy": torch.stack(entropy),
                    "latent": latent,
                    "reconstruction": reconstruction,
                    "transition_log_prob": transition_log_prob,
                }

        return SwitchingStateSpaceModel()

    def _raw_components(
        self,
        xs: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray,
    ) -> dict[str, np.ndarray]:
        output = {
            "joint_nll": np.zeros(len(xs), dtype=float),
            "observation_nll": np.zeros(len(xs), dtype=float),
            "dynamics_nll": np.zeros(len(xs), dtype=float),
            "mode_entropy": np.zeros(len(xs), dtype=float),
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
                factors = self.model_.likelihood_factors(sequence, dt)
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

    def _calibrate(self, raw_scores: np.ndarray, factor: str) -> np.ndarray:
        reference = self.calibration_scores_[factor]
        if len(reference) == 0:
            return raw_scores.copy()
        # A robust affine transform preserves the complete likelihood ordering.
        # An empirical tail p-value creates large ties when a deployment source
        # has a shifted likelihood range (for example, a different sampling
        # interval), even though its raw likelihood still ranks observations.
        median = float(np.median(reference))
        mad = float(np.median(np.abs(reference - median)))
        scale = max(1.4826 * mad, float(np.std(reference)), 1e-6)
        return (raw_scores - median) / scale

    def _validate_configuration(self) -> None:
        if self.geometry not in {"euclidean", "sphere"}:
            raise ValueError("geometry must be 'euclidean' or 'sphere'.")
        if self.score_factor not in {"joint", "observation", "dynamics"}:
            raise ValueError("score_factor must be joint, observation, or dynamics.")
        if self.latent_dim < 2 or self.hidden_dim < 2:
            raise ValueError("latent_dim and hidden_dim must be at least two.")
        if self.n_modes < 1 or self.sequence_length < 2 or self.epochs < 1:
            raise ValueError("n_modes, sequence_length, and epochs must be positive.")
        if self.learning_rate <= 0.0 or self.gradient_clip <= 0.0:
            raise ValueError("learning_rate and gradient_clip must be positive.")
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
