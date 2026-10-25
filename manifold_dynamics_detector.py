#!/usr/bin/env python3
"""Unified conditional-dynamics detector on Euclidean or spherical latent spaces.

The detector is intentionally separate from ``MultiConceptPendulumDetector``.
It provides a matched experimental comparison in which Euclidean and spherical
models share the same encoder, soft modes, dynamics regressors, transition
model, likelihood score, and calibration.  The geometry is therefore the main
controlled difference rather than one component in a heterogeneous ensemble.

The implementation is causal at scoring time:

* optional smoothing uses only present and past observations;
* a score assigned to time ``t`` uses the transition from ``t-1`` to ``t``;
* timestamps determine the displacement scale;
* no centered rolling propagation is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


EPS = 1e-10
LOG_2PI = float(np.log(2.0 * np.pi))


def _row_norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1, keepdims=True)


def _safe_unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(_row_norm(x), EPS)


def sphere_log_map(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Log_base(target) on the unit sphere in ambient coordinates."""

    base = _safe_unit(np.asarray(base, dtype=float))
    target = _safe_unit(np.asarray(target, dtype=float))
    cosine = np.clip(np.sum(base * target, axis=1, keepdims=True), -1.0, 1.0)
    angle = np.arccos(cosine)
    tangent_direction = target - cosine * base
    tangent_norm = _row_norm(tangent_direction)
    scale = np.divide(angle, tangent_norm, out=np.ones_like(angle), where=tangent_norm > EPS)
    displacement = scale * tangent_direction
    displacement[tangent_norm[:, 0] <= EPS] = 0.0
    return displacement


def sphere_parallel_transport(base: np.ndarray, target: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Parallel-transport tangent vectors along the shortest spherical geodesic."""

    base = _safe_unit(np.asarray(base, dtype=float))
    target = _safe_unit(np.asarray(target, dtype=float))
    vector = np.asarray(vector, dtype=float)
    cosine = np.sum(base * target, axis=1, keepdims=True)
    denom = 1.0 + cosine
    transported = vector.copy()
    stable = np.abs(denom[:, 0]) > 1e-7
    if np.any(stable):
        projection = np.sum(vector[stable] * target[stable], axis=1, keepdims=True)
        transported[stable] = vector[stable] - projection / denom[stable] * (base[stable] + target[stable])
    # Antipodal points do not have a unique shortest geodesic.  The fallback is
    # a tangent projection, which remains finite and exposes the ambiguity as a
    # large displacement likelihood residual.
    transported -= np.sum(transported * target, axis=1, keepdims=True) * target
    return transported


def _causal_ema(x: np.ndarray, entity_ids: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1].")
    if alpha == 1.0:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    for entity_id in np.unique(entity_ids):
        idx = np.flatnonzero(entity_ids == entity_id)
        if len(idx) == 0:
            continue
        out[idx[0]] = x[idx[0]]
        for current, previous in zip(idx[1:], idx[:-1]):
            out[current] = alpha * x[current] + (1.0 - alpha) * out[previous]
    return out


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis) + EPS
    )


@dataclass(frozen=True)
class _TransitionBatch:
    landing_indices: np.ndarray
    entity_ids: np.ndarray
    base_state: np.ndarray
    predictors: np.ndarray
    targets: np.ndarray
    displacement_norm: np.ndarray
    radial_change: np.ndarray


class ManifoldDynamicsDetector(BaseEstimator):
    """Mode-conditioned latent dynamics with a single predictive score.

    Parameters
    ----------
    geometry:
        ``"sphere"`` uses logarithmic-map displacements on a unit hypersphere;
        ``"euclidean"`` uses first-order Euclidean displacement.  All other
        modeling choices are matched.
    include_radius:
        For spherical geometry, jointly model the latent radius and its rate of
        change.  This prevents normalization from silently discarding magnitude
        information.  It is ignored for Euclidean geometry.
    field_model:
        ``"ridge"`` estimates one global linear vector field per soft mode.
        ``"local"`` estimates a nonlinear local conditional Gaussian from
        neighboring normal transitions.  Spherical neighbor velocities are
        parallel-transported to the query tangent space before comparison.
    ema_alpha:
        Causal exponential smoothing coefficient.  ``1.0`` disables smoothing.
    calibrate:
        Convert predictive negative log likelihood to an empirical normal-tail
        score using chronologically held-out normal transitions.  Calibration
        uses no anomaly labels.
    """

    def __init__(
        self,
        geometry: str = "sphere",
        latent_dim: int = 5,
        n_modes: int = 4,
        include_radius: bool = True,
        field_model: str = "ridge",
        ridge_alpha: float = 1.0,
        local_neighbors: int = 64,
        local_bandwidth: float = 0.5,
        local_chunk_size: int = 2048,
        covariance_floor: float = 1e-3,
        transition_smoothing: float = 0.5,
        transition_strength: float = 1.0,
        ema_alpha: float = 1.0,
        calibrate: bool = True,
        calibration_fraction: float = 0.2,
        random_state: int = 37,
    ) -> None:
        self.geometry = geometry
        self.latent_dim = latent_dim
        self.n_modes = n_modes
        self.include_radius = include_radius
        self.field_model = field_model
        self.ridge_alpha = ridge_alpha
        self.local_neighbors = local_neighbors
        self.local_bandwidth = local_bandwidth
        self.local_chunk_size = local_chunk_size
        self.covariance_floor = covariance_floor
        self.transition_smoothing = transition_smoothing
        self.transition_strength = transition_strength
        self.ema_alpha = ema_alpha
        self.calibrate = calibrate
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> "ManifoldDynamicsDetector":
        self._validate_configuration()
        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)

        representation_fit = self._temporal_fit_mask(entity_ids)
        self.feature_scaler_ = StandardScaler().fit(x[representation_fit])
        xs = self.feature_scaler_.transform(x)
        xs = _causal_ema(xs, entity_ids, self.ema_alpha)

        latent_dim = min(self.latent_dim, xs.shape[1])
        self.pca_ = PCA(n_components=latent_dim, random_state=self.random_state)
        self.pca_.fit(xs[representation_fit])
        latent = self.pca_.transform(xs)
        state, radius = self._state_representation(latent)
        transitions = self._build_transitions(state, radius, entity_ids, times)
        if len(transitions.targets) < 8:
            raise ValueError("At least eight within-entity transitions are required.")

        dynamics_fit_mask = representation_fit[transitions.landing_indices]
        fit_transitions = self._subset_transitions(transitions, dynamics_fit_mask)
        calibration_transitions = self._subset_transitions(transitions, ~dynamics_fit_mask)
        if len(fit_transitions.targets) < 8:
            raise ValueError("The dynamics fitting split contains fewer than eight transitions.")
        if self.calibrate and len(calibration_transitions.targets) < 2:
            raise ValueError("The calibration split contains fewer than two transitions.")

        self.predictor_scaler_ = RobustScaler()
        predictor_scaled = self.predictor_scaler_.fit_transform(fit_transitions.predictors)
        max_modes = max(1, len(predictor_scaled) // 8)
        requested_modes = min(self.n_modes, max_modes)
        last_gmm_error: ValueError | None = None
        responsibilities: np.ndarray | None = None
        for modes in range(requested_modes, 0, -1):
            for regularization in (1e-5, 1e-4, 1e-3):
                candidate = GaussianMixture(
                    n_components=modes,
                    covariance_type="diag",
                    reg_covar=regularization,
                    n_init=3,
                    random_state=self.random_state,
                )
                try:
                    responsibilities = candidate.fit_predict(predictor_scaled)
                    self.gmm_ = candidate
                    self.n_modes_ = modes
                    self.gmm_reg_covar_ = regularization
                    break
                except ValueError as exc:
                    last_gmm_error = exc
            if responsibilities is not None:
                break
        if responsibilities is None:
            raise ValueError("Unable to fit even a single regularized dynamics mode.") from last_gmm_error
        soft_responsibilities = self.gmm_.predict_proba(predictor_scaled)
        # Guard against a degenerate predict_proba result while retaining the
        # fitted hard assignment as a deterministic fallback.
        if not np.isfinite(soft_responsibilities).all():
            soft_responsibilities = np.eye(self.n_modes_)[responsibilities]

        self.models_: list[Ridge] = []
        self.residual_variances_: list[np.ndarray] = []
        if self.field_model == "ridge":
            for mode in range(self.n_modes_):
                weights = soft_responsibilities[:, mode]
                model = Ridge(alpha=self.ridge_alpha, solver="lsqr")
                model.fit(
                    fit_transitions.predictors,
                    fit_transitions.targets,
                    sample_weight=weights + 1e-6,
                )
                prediction = self._project_prediction(
                    model.predict(fit_transitions.predictors),
                    fit_transitions.base_state,
                )
                residual = fit_transitions.targets - prediction
                weight_sum = float(weights.sum()) + EPS
                variance = np.sum(weights[:, None] * residual**2, axis=0) / weight_sum
                variance = np.maximum(variance, self.covariance_floor)
                self.models_.append(model)
                self.residual_variances_.append(variance)
        else:
            self.local_neighbors_ = min(self.local_neighbors, len(fit_transitions.targets))
            self.neighbor_model_ = NearestNeighbors(
                n_neighbors=self.local_neighbors_,
                n_jobs=-1,
            ).fit(predictor_scaled)
            self.local_targets_ = fit_transitions.targets.copy()
            self.local_base_states_ = fit_transitions.base_state.copy()
            self.local_responsibilities_ = soft_responsibilities.copy()

        self.mode_weights_ = np.maximum(soft_responsibilities.mean(axis=0), EPS)
        self.mode_weights_ /= self.mode_weights_.sum()
        self.transition_matrix_ = self._fit_soft_transition_matrix(
            soft_responsibilities,
            fit_transitions.entity_ids,
        )

        score_reference = calibration_transitions if self.calibrate else fit_transitions
        raw_scores, _ = self._score_transition_batch(score_reference)
        self.training_raw_scores_ = np.sort(raw_scores[np.isfinite(raw_scores)])
        if len(self.training_raw_scores_) == 0:
            raise ValueError("Training produced no finite dynamics likelihood scores.")
        return self

    def decision_function(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)
        xs = self.feature_scaler_.transform(x)
        xs = _causal_ema(xs, entity_ids, self.ema_alpha)
        latent = self.pca_.transform(xs)
        state, radius = self._state_representation(latent)
        transitions = self._build_transitions(state, radius, entity_ids, times)
        raw_scores, _ = self._score_transition_batch(transitions)
        transition_scores = self._calibrate(raw_scores) if self.calibrate else raw_scores

        scores = np.zeros(len(x), dtype=float)
        scores[transitions.landing_indices] = transition_scores
        return scores

    def decision_components(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Return diagnostics without introducing additional anomaly scores."""

        x, entity_ids, times = self._validate_inputs(x, entity_ids, times)
        xs = _causal_ema(self.feature_scaler_.transform(x), entity_ids, self.ema_alpha)
        state, radius = self._state_representation(self.pca_.transform(xs))
        transitions = self._build_transitions(state, radius, entity_ids, times)
        raw_scores, entropy = self._score_transition_batch(transitions)

        out = {
            "raw_nll": np.zeros(len(x), dtype=float),
            "calibrated_score": np.zeros(len(x), dtype=float),
            "mode_entropy": np.zeros(len(x), dtype=float),
            "displacement_norm": np.zeros(len(x), dtype=float),
            "radial_change": np.zeros(len(x), dtype=float),
        }
        idx = transitions.landing_indices
        out["raw_nll"][idx] = raw_scores
        out["calibrated_score"][idx] = self._calibrate(raw_scores)
        out["mode_entropy"][idx] = entropy
        out["displacement_norm"][idx] = transitions.displacement_norm
        out["radial_change"][idx] = transitions.radial_change
        return out

    def _validate_configuration(self) -> None:
        if self.geometry not in {"sphere", "euclidean"}:
            raise ValueError("geometry must be 'sphere' or 'euclidean'.")
        if self.field_model not in {"ridge", "local"}:
            raise ValueError("field_model must be 'ridge' or 'local'.")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if self.n_modes < 1:
            raise ValueError("n_modes must be positive.")
        if self.covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be positive.")
        if self.transition_smoothing <= 0.0:
            raise ValueError("transition_smoothing must be positive.")
        if self.transition_strength < 0.0:
            raise ValueError("transition_strength must be non-negative.")
        if self.local_neighbors < 2:
            raise ValueError("local_neighbors must be at least two.")
        if self.local_bandwidth <= 0.0:
            raise ValueError("local_bandwidth must be positive.")
        if self.local_chunk_size < 1:
            raise ValueError("local_chunk_size must be positive.")
        if not 0.0 <= self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in [0, 0.5).")
        if self.calibrate and self.calibration_fraction <= 0.0:
            raise ValueError("calibration_fraction must be positive when calibrate=True.")

    def _temporal_fit_mask(self, entity_ids: np.ndarray) -> np.ndarray:
        mask = np.ones(len(entity_ids), dtype=bool)
        if not self.calibrate:
            return mask
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            calibration_size = max(2, int(np.ceil(len(idx) * self.calibration_fraction)))
            if len(idx) - calibration_size < 3:
                raise ValueError("Each entity requires at least five rows for temporal calibration.")
            mask[idx[-calibration_size:]] = False
        return mask

    @staticmethod
    def _validate_inputs(
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        entity_ids = np.asarray(entity_ids)
        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional array.")
        if len(x) != len(entity_ids):
            raise ValueError("x and entity_ids must have equal length.")
        if not np.isfinite(x).all():
            raise ValueError("x contains non-finite values.")
        if times is None:
            times_array = np.empty(len(x), dtype=float)
            for entity_id in np.unique(entity_ids):
                idx = np.flatnonzero(entity_ids == entity_id)
                times_array[idx] = np.arange(len(idx), dtype=float)
            times = times_array
        else:
            times = np.asarray(times, dtype=float)
            if len(times) != len(x):
                raise ValueError("x and times must have equal length.")
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            if len(idx) > 1 and np.any(np.diff(times[idx]) <= 0.0):
                raise ValueError("times must be strictly increasing within each entity.")
        return x, entity_ids, times

    def _state_representation(self, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        radius = np.linalg.norm(latent, axis=1)
        if self.geometry == "sphere":
            return _safe_unit(latent), radius
        return latent, radius

    @staticmethod
    def _subset_transitions(transitions: _TransitionBatch, mask: np.ndarray) -> _TransitionBatch:
        return _TransitionBatch(
            landing_indices=transitions.landing_indices[mask],
            entity_ids=transitions.entity_ids[mask],
            base_state=transitions.base_state[mask],
            predictors=transitions.predictors[mask],
            targets=transitions.targets[mask],
            displacement_norm=transitions.displacement_norm[mask],
            radial_change=transitions.radial_change[mask],
        )

    def _build_transitions(
        self,
        state: np.ndarray,
        radius: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray,
    ) -> _TransitionBatch:
        landing_indices: list[np.ndarray] = []
        transition_entities: list[np.ndarray] = []
        base_states: list[np.ndarray] = []
        predictors: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        displacement_norms: list[np.ndarray] = []
        radial_changes: list[np.ndarray] = []

        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            if len(idx) < 2:
                continue
            base = state[idx[:-1]]
            destination = state[idx[1:]]
            dt = np.diff(times[idx])[:, None]
            if self.geometry == "sphere":
                displacement = sphere_log_map(base, destination) / dt
            else:
                displacement = (destination - base) / dt
            radial_delta = np.diff(radius[idx])[:, None] / dt

            previous_displacement = np.zeros_like(displacement)
            previous_radial_delta = np.zeros_like(radial_delta)
            if len(displacement) > 1:
                if self.geometry == "sphere":
                    previous_displacement[1:] = sphere_parallel_transport(
                        state[idx[:-2]],
                        state[idx[1:-1]],
                        displacement[:-1],
                    )
                else:
                    previous_displacement[1:] = displacement[:-1]
                previous_radial_delta[1:] = radial_delta[:-1]

            predictor_parts = [base, previous_displacement]
            target_parts = [displacement]
            if self.geometry == "sphere" and self.include_radius:
                predictor_parts.extend([radius[idx[:-1], None], previous_radial_delta])
                target_parts.append(radial_delta)

            landing_indices.append(idx[1:])
            transition_entities.append(np.full(len(idx) - 1, entity_id))
            base_states.append(base)
            predictors.append(np.hstack(predictor_parts))
            targets.append(np.hstack(target_parts))
            displacement_norms.append(np.linalg.norm(displacement, axis=1))
            radial_changes.append(np.abs(radial_delta[:, 0]))

        if not targets:
            raise ValueError("No entity contains enough observations to form a transition.")
        return _TransitionBatch(
            landing_indices=np.concatenate(landing_indices),
            entity_ids=np.concatenate(transition_entities),
            base_state=np.vstack(base_states),
            predictors=np.vstack(predictors),
            targets=np.vstack(targets),
            displacement_norm=np.concatenate(displacement_norms),
            radial_change=np.concatenate(radial_changes),
        )

    def _project_prediction(self, prediction: np.ndarray, base_state: np.ndarray) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=float).copy()
        if self.geometry == "sphere":
            tangent = prediction[:, : base_state.shape[1]]
            tangent -= np.sum(tangent * base_state, axis=1, keepdims=True) * base_state
            prediction[:, : base_state.shape[1]] = tangent
        return prediction

    def _fit_soft_transition_matrix(
        self,
        responsibilities: np.ndarray,
        transition_entities: np.ndarray,
    ) -> np.ndarray:
        counts = np.full(
            (self.n_modes_, self.n_modes_),
            self.transition_smoothing,
            dtype=float,
        )
        for entity_id in np.unique(transition_entities):
            idx = np.flatnonzero(transition_entities == entity_id)
            for previous, current in zip(idx[:-1], idx[1:]):
                counts += np.outer(responsibilities[previous], responsibilities[current])
        return counts / counts.sum(axis=1, keepdims=True)

    def _score_transition_batch(self, transitions: _TransitionBatch) -> tuple[np.ndarray, np.ndarray]:
        predictor_scaled = self.predictor_scaler_.transform(transitions.predictors)
        gating = np.maximum(self.gmm_.predict_proba(predictor_scaled), EPS)
        gating /= gating.sum(axis=1, keepdims=True)

        if self.field_model == "local":
            log_density = self._local_log_density(transitions, predictor_scaled)
        else:
            log_density = np.empty((len(transitions.targets), self.n_modes_), dtype=float)
            for mode, (model, variance) in enumerate(zip(self.models_, self.residual_variances_)):
                prediction = self._project_prediction(
                    model.predict(transitions.predictors),
                    transitions.base_state,
                )
                residual = transitions.targets - prediction
                log_density[:, mode] = -0.5 * np.sum(
                    residual**2 / variance + np.log(variance) + LOG_2PI,
                    axis=1,
                )

        log_prior = np.empty_like(log_density)
        for entity_id in np.unique(transitions.entity_ids):
            idx = np.flatnonzero(transitions.entity_ids == entity_id)
            previous = self.mode_weights_
            for row in idx:
                predicted_mode = np.maximum(previous @ self.transition_matrix_, EPS)
                combined = np.log(gating[row]) + self.transition_strength * np.log(predicted_mode)
                combined -= _logsumexp(combined[None, :], axis=1)[0]
                log_prior[row] = combined
                previous = gating[row]

        raw_scores = -_logsumexp(log_prior + log_density, axis=1)
        entropy = -np.sum(np.exp(log_prior) * log_prior, axis=1)
        return raw_scores, entropy

    def _local_log_density(
        self,
        transitions: _TransitionBatch,
        predictor_scaled: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a local conditional Gaussian in the query tangent space."""

        log_density = np.empty((len(transitions.targets), self.n_modes_), dtype=float)
        state_dim = transitions.base_state.shape[1]
        for start in range(0, len(transitions.targets), self.local_chunk_size):
            end = min(len(transitions.targets), start + self.local_chunk_size)
            distances, neighbor_indices = self.neighbor_model_.kneighbors(
                predictor_scaled[start:end],
                return_distance=True,
            )
            local_targets = self.local_targets_[neighbor_indices].copy()
            if self.geometry == "sphere":
                source_base = self.local_base_states_[neighbor_indices].reshape(-1, state_dim)
                query_base = np.repeat(
                    transitions.base_state[start:end],
                    self.local_neighbors_,
                    axis=0,
                )
                direction = local_targets[:, :, :state_dim].reshape(-1, state_dim)
                transported = sphere_parallel_transport(source_base, query_base, direction)
                local_targets[:, :, :state_dim] = transported.reshape(
                    end - start,
                    self.local_neighbors_,
                    state_dim,
                )

            adaptive_scale = np.maximum(distances[:, -1:], EPS) * self.local_bandwidth
            kernel_weights = np.exp(-0.5 * (distances / adaptive_scale) ** 2)
            local_modes = self.local_responsibilities_[neighbor_indices]
            observed = transitions.targets[start:end]
            for mode in range(self.n_modes_):
                weights = kernel_weights * local_modes[:, :, mode]
                weight_sum = np.maximum(weights.sum(axis=1, keepdims=True), EPS)
                mean = np.sum(weights[:, :, None] * local_targets, axis=1) / weight_sum
                centered = local_targets - mean[:, None, :]
                variance = np.sum(weights[:, :, None] * centered**2, axis=1) / weight_sum
                variance = np.maximum(variance, self.covariance_floor)
                residual = observed - mean
                log_density[start:end, mode] = -0.5 * np.sum(
                    residual**2 / variance + np.log(variance) + LOG_2PI,
                    axis=1,
                )
        return log_density

    def _calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        sorted_training = self.training_raw_scores_
        insertion = np.searchsorted(sorted_training, raw_scores, side="left")
        upper_count = len(sorted_training) - insertion
        p_value = (upper_count + 1.0) / (len(sorted_training) + 1.0)
        return -np.log(np.maximum(p_value, EPS))
