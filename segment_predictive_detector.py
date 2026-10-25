#!/usr/bin/env python3
"""Segment-continuity likelihood ratio built on the causal predictor.

For every candidate segment start, the latent state immediately before the
segment is advanced without teacher forcing. The same observations are also
scored under re-anchored, one-step normal predictions. Their log-likelihood
ratio tests whether a locally plausible segment is compatible with the state
that preceded it. A calibrated maximum over predeclared horizons is an online
generalized scan statistic, not a rolling propagation of a point score.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from predictive_generative_detector import (
    LOG_2PI,
    PredictiveGenerativeDetector,
)


class SegmentPredictiveDetector(BaseEstimator):
    """Causal multi-horizon likelihood-ratio test of segment continuity."""

    def __init__(
        self,
        horizons: tuple[int, ...] = (2, 4, 8, 16, 32),
        geometry: str = "euclidean",
        latent_dim: int = 6,
        hidden_dim: int = 48,
        sequence_length: int = 128,
        epochs: int = 15,
        learning_rate: float = 1e-3,
        rollout_epochs: int = 0,
        rollout_learning_rate: float = 5e-4,
        probabilistic_rollout: bool = False,
        calibration_fraction: float = 0.2,
        random_state: int = 37,
    ) -> None:
        self.horizons = horizons
        self.geometry = geometry
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.rollout_epochs = rollout_epochs
        self.rollout_learning_rate = rollout_learning_rate
        self.probabilistic_rollout = probabilistic_rollout
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> "SegmentPredictiveDetector":
        self._validate_configuration()
        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)
        self.base_detector_ = PredictiveGenerativeDetector(
            geometry=self.geometry,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_modes=1,
            sequence_length=self.sequence_length,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            calibration_fraction=self.calibration_fraction,
            calibrate=True,
            random_state=self.random_state,
        ).fit(x, entity_ids, times)

        xs = self.base_detector_.feature_scaler_.transform(x).astype(np.float32)
        self._initialize_process_uncertainty()
        self._fit_multistep_rollouts(xs, entity_ids, times)
        free_nll, reanchored_nll = self._segment_likelihoods(xs, entity_ids, times)
        horizon_evidence = free_nll - reanchored_nll
        calibration_mask = ~self.base_detector_.fit_mask_
        self.horizon_centers_ = np.zeros(len(self.horizons_), dtype=float)
        self.horizon_scales_ = np.ones(len(self.horizons_), dtype=float)
        for column in range(len(self.horizons_)):
            valid = calibration_mask & np.isfinite(horizon_evidence[:, column])
            reference = horizon_evidence[valid, column]
            if len(reference) < 2:
                raise ValueError(
                    f"Horizon {self.horizons_[column]} has insufficient calibration segments."
                )
            center, scale = self._robust_location_scale(reference)
            self.horizon_centers_[column] = center
            self.horizon_scales_[column] = scale

        scan, _ = self._scan_statistic(horizon_evidence)
        scan_reference = scan[calibration_mask & np.isfinite(scan)]
        if len(scan_reference) < 2:
            raise ValueError("Segment scan has insufficient calibration scores.")
        self.scan_center_, self.scan_scale_ = self._robust_location_scale(scan_reference)
        return self

    def _initialize_process_uncertainty(self) -> None:
        """Register diagonal latent process noise and deterministic sigma points."""

        if not self.probabilistic_rollout:
            return
        torch = self.base_detector_.torch_
        nn = torch.nn
        model = self.base_detector_.model_
        model.process_scale_raw = nn.Parameter(
            torch.full((self.latent_dim,), -3.0, dtype=torch.float32)
        )
        particle_count = 2 * self.latent_dim + 1
        amplitude = float(np.sqrt(particle_count / 2.0))
        sigma_points = torch.zeros(
            particle_count,
            self.latent_dim,
            dtype=torch.float32,
        )
        for dimension in range(self.latent_dim):
            sigma_points[1 + 2 * dimension, dimension] = amplitude
            sigma_points[2 + 2 * dimension, dimension] = -amplitude
        self.sigma_points_ = sigma_points

    def _particle_successors(self, states, dt, step: int):
        """Propagate equally weighted particles through one latent transition."""

        torch = self.base_detector_.torch_
        model = self.base_detector_.model_
        particles = len(self.sigma_points_)
        if states.ndim == 2:
            states = states[:, None, :].expand(-1, particles, -1)
        count = len(states)
        flat_states = states.reshape(-1, self.latent_dim)
        flat_dt = dt[:, None, :].expand(-1, particles, -1).reshape(-1, 1)
        means = model.successor_states(flat_states, flat_dt)[:, 0, :].reshape(
            count,
            particles,
            self.latent_dim,
        )
        process_std = 1e-3 + 0.5 * torch.sigmoid(model.process_scale_raw)
        sigma_points = torch.roll(
            self.sigma_points_.to(states.device),
            shifts=step - 1,
            dims=1,
        )
        noise = (
            torch.sqrt(dt)[:, None, :]
            * process_std[None, None, :]
            * sigma_points[None, :, :]
        )
        successors = means + noise
        if self.geometry == "sphere":
            successors = torch.nn.functional.normalize(successors, dim=2, eps=1e-8)
        return successors

    def _particle_observation_nll(self, particles, target, log_variance):
        """Moment-match particle predictions to a diagonal observation Gaussian."""

        torch = self.base_detector_.torch_
        model = self.base_detector_.model_
        count, particle_count, _ = particles.shape
        decoded = model.decode_successors(
            particles.reshape(-1, self.latent_dim)[:, None, :]
        )[:, 0, :].reshape(count, particle_count, -1)
        mean = decoded.mean(dim=1)
        process_variance = torch.mean((decoded - mean[:, None, :]) ** 2, dim=1)
        variance = torch.exp(log_variance)[None, :] + process_variance + 1e-6
        return 0.5 * torch.sum(
            (target - mean) ** 2 / variance + torch.log(variance) + LOG_2PI,
            dim=1,
        )

    def _fit_multistep_rollouts(
        self,
        xs: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray,
    ) -> None:
        """Fine-tune the null model on the free-running horizons used at test time."""

        self.rollout_training_loss_: list[float] = []
        if self.rollout_epochs == 0:
            return
        torch = self.base_detector_.torch_
        model = self.base_detector_.model_
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.rollout_learning_rate,
            weight_decay=self.base_detector_.weight_decay,
        )
        chunks = self.base_detector_._fit_chunks(
            entity_ids,
            self.base_detector_.fit_mask_,
        )
        rng = np.random.default_rng(self.random_state + 1)
        max_horizon = self.horizons_[-1]
        model.train()
        for _ in range(self.rollout_epochs):
            epoch_losses = []
            for position in rng.permutation(len(chunks)):
                idx = chunks[int(position)]
                if len(idx) < 2:
                    continue
                sequence = torch.as_tensor(xs[idx], dtype=torch.float32)
                latent = model.encode(sequence)
                states = latent[:-1]
                time_differences = np.diff(times[idx]) / self.base_detector_.dt_scale_
                log_variance = model.bounded_log_variance(
                    model.observation_scale[0]
                )
                variance = torch.exp(log_variance)
                step_losses = []
                for step in range(1, min(max_horizon, len(idx) - 1) + 1):
                    count = len(idx) - step
                    states = states[:count]
                    dt = torch.as_tensor(
                        time_differences[step - 1 :, None],
                        dtype=torch.float32,
                    )
                    if self.probabilistic_rollout:
                        states = self._particle_successors(states, dt, step)
                        step_nll = self._particle_observation_nll(
                            states,
                            sequence[step:],
                            log_variance,
                        )
                    else:
                        states = model.successor_states(states, dt)[:, 0, :]
                        mean = model.decode_successors(states[:, None, :])[:, 0, :]
                        residual = sequence[step:] - mean
                        step_nll = 0.5 * torch.sum(
                            residual**2 / variance + log_variance + LOG_2PI,
                            dim=1,
                        )
                    step_losses.append(step_nll.mean())
                loss = torch.stack(step_losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Multi-step rollout training produced a non-finite loss."
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.base_detector_.gradient_clip
                )
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            if not epoch_losses:
                raise ValueError("No sequence supports multi-step rollout training.")
            self.rollout_training_loss_.append(float(np.mean(epoch_losses)))

    def decision_function(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.decision_components(x, entity_ids, times)["calibrated_score"]

    def decision_components(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)
        xs = self.base_detector_.feature_scaler_.transform(x).astype(np.float32)
        free_nll, reanchored_nll = self._segment_likelihoods(xs, entity_ids, times)
        horizon_evidence = free_nll - reanchored_nll
        scan, best_horizon = self._scan_statistic(horizon_evidence)
        calibrated = (scan - self.scan_center_) / self.scan_scale_
        for entity_id in np.unique(entity_ids):
            first = np.flatnonzero(entity_ids == entity_id)[0]
            calibrated[first] = 0.0
            scan[first] = 0.0
            best_horizon[first] = 0
        return {
            "calibrated_score": calibrated,
            "scan_statistic": scan,
            "best_horizon": best_horizon,
            "horizon_evidence": horizon_evidence,
            "free_running_nll": free_nll,
            "reanchored_nll": reanchored_nll,
        }

    def _segment_likelihoods(
        self,
        xs: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.base_detector_.torch_
        model = self.base_detector_.model_
        free_output = np.full((len(xs), len(self.horizons_)), np.nan, dtype=float)
        reanchored_output = np.full(
            (len(xs), len(self.horizons_)), np.nan, dtype=float
        )
        horizon_columns = {
            horizon: column for column, horizon in enumerate(self.horizons_)
        }
        max_horizon = self.horizons_[-1]
        model.eval()
        with torch.no_grad():
            log_variance = model.bounded_log_variance(model.observation_scale[0])
            variance = torch.exp(log_variance)
            for entity_id in np.unique(entity_ids):
                idx = np.flatnonzero(entity_ids == entity_id)
                if len(idx) < 2:
                    continue
                sequence = torch.as_tensor(xs[idx], dtype=torch.float32)
                latent = model.encode(sequence)
                states = latent[:-1]
                free_cumulative = torch.zeros(len(idx) - 1, dtype=torch.float32)
                reanchored_cumulative = torch.zeros(
                    len(idx) - 1, dtype=torch.float32
                )
                time_differences = np.diff(times[idx]) / self.base_detector_.dt_scale_
                all_dt = torch.as_tensor(
                    time_differences[:, None], dtype=torch.float32
                )
                if self.probabilistic_rollout:
                    reanchored_successors = self._particle_successors(
                        latent[:-1], all_dt, 1
                    )
                    reanchored_point_nll = self._particle_observation_nll(
                        reanchored_successors,
                        sequence[1:],
                        log_variance,
                    )
                else:
                    reanchored_successors = model.successor_states(
                        latent[:-1], all_dt
                    )[:, 0, :]
                    reanchored_means = model.decode_successors(
                        reanchored_successors[:, None, :]
                    )[:, 0, :]
                    reanchored_residual = sequence[1:] - reanchored_means
                    reanchored_point_nll = 0.5 * torch.sum(
                        reanchored_residual**2 / variance
                        + log_variance
                        + LOG_2PI,
                        dim=1,
                    )
                for step in range(1, min(max_horizon, len(idx) - 1) + 1):
                    count = len(idx) - step
                    states = states[:count]
                    dt = torch.as_tensor(
                        time_differences[step - 1 :, None],
                        dtype=torch.float32,
                    )
                    target = sequence[step:]
                    if self.probabilistic_rollout:
                        successor = self._particle_successors(states, dt, step)
                        point_nll = self._particle_observation_nll(
                            successor,
                            target,
                            log_variance,
                        )
                    else:
                        successor = model.successor_states(states, dt)[:, 0, :]
                        mean = model.decode_successors(successor[:, None, :])[:, 0, :]
                        residual = target - mean
                        point_nll = 0.5 * torch.sum(
                            residual**2 / variance + log_variance + LOG_2PI,
                            dim=1,
                        )
                    free_cumulative = free_cumulative[:count] + point_nll
                    reanchored_cumulative = (
                        reanchored_cumulative[:count]
                        + reanchored_point_nll[step - 1 :]
                    )
                    if step in horizon_columns:
                        column = horizon_columns[step]
                        free_output[idx[step:], column] = free_cumulative.cpu().numpy()
                        reanchored_output[idx[step:], column] = (
                            reanchored_cumulative.cpu().numpy()
                        )
                    states = successor
        return free_output, reanchored_output

    def _scan_statistic(
        self, horizon_evidence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        standardized = (
            horizon_evidence - self.horizon_centers_[None, :]
        ) / self.horizon_scales_[None, :]
        available = np.isfinite(standardized)
        candidates = np.where(available, standardized, -np.inf)
        best_column = np.argmax(candidates, axis=1)
        scan = candidates[np.arange(len(candidates)), best_column]
        no_candidate = ~available.any(axis=1)
        scan[no_candidate] = 0.0
        best_horizon = np.asarray(self.horizons_, dtype=int)[best_column]
        best_horizon[no_candidate] = 0
        return scan, best_horizon

    @staticmethod
    def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = max(1.4826 * mad, float(np.std(values)), 1e-6)
        return center, scale

    def _validate_configuration(self) -> None:
        horizons = tuple(sorted(set(int(value) for value in self.horizons)))
        if not horizons or horizons[0] < 2:
            raise ValueError("horizons must contain integers of at least two.")
        self.horizons_ = horizons
        if self.geometry not in {"euclidean", "sphere"}:
            raise ValueError("geometry must be 'euclidean' or 'sphere'.")
        if self.latent_dim < 2 or self.hidden_dim < 2 or self.epochs < 1:
            raise ValueError(
                "latent_dim and hidden_dim must be at least two; epochs must be positive."
            )
        if self.rollout_epochs < 0 or self.rollout_learning_rate <= 0.0:
            raise ValueError(
                "rollout_epochs must be non-negative and rollout_learning_rate positive."
            )
        if self.probabilistic_rollout and self.rollout_epochs == 0:
            raise ValueError(
                "probabilistic_rollout requires positive rollout_epochs."
            )
        if not 0.0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in (0, 0.5).")

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
            generated = np.empty(len(x), dtype=float)
            for entity_id in np.unique(entity_ids):
                idx = np.flatnonzero(entity_ids == entity_id)
                generated[idx] = np.arange(len(idx), dtype=float)
            times = generated
        else:
            times = np.asarray(times, dtype=float)
            if len(times) != len(x):
                raise ValueError("times must match x.")
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            if len(idx) > 1 and np.any(np.diff(times[idx]) <= 0.0):
                raise ValueError("times must be strictly increasing within each entity.")
        return x, entity_ids, times
