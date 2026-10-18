#!/usr/bin/env python3
"""Pendulum-inspired multi-concept anomaly detector.

The detector learns normal trajectories as coexisting spherical latent regimes.
It is intentionally lightweight and sklearn-compatible enough for benchmark use:

1. standardize observed features;
2. project to a low-dimensional latent space;
3. normalize latent states to a hypersphere;
4. estimate per-entity velocity and acceleration on that sphere;
5. cluster normal state/velocity pairs into multiple normal sub-concepts;
6. fit a local linear dynamics model per concept;
7. score points by angular fit, dynamic residual, energy deviation, and transition rarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler


EPS = 1e-12


def _row_norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1, keepdims=True)


def _safe_unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(_row_norm(x), EPS)


def _robust_location_scale(x: np.ndarray) -> tuple[float, float]:
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    return median, max(1.4826 * mad, 1e-6)


def _sequence_derivatives(z: np.ndarray, entity_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.zeros_like(z)
    a = np.zeros_like(z)

    for entity_id in np.unique(entity_ids):
        idx = np.flatnonzero(entity_ids == entity_id)
        if len(idx) < 3:
            continue
        zi = z[idx]
        vi = np.gradient(zi, axis=0)
        # Keep velocity tangent to the sphere.
        vi = vi - np.sum(vi * zi, axis=1, keepdims=True) * zi
        ai = np.gradient(vi, axis=0)
        ai = ai - np.sum(ai * zi, axis=1, keepdims=True) * zi
        v[idx] = vi
        a[idx] = ai

    return v, a


def _euclidean_sequence_derivatives(x: np.ndarray, entity_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.zeros_like(x)
    ddx = np.zeros_like(x)
    for entity_id in np.unique(entity_ids):
        idx = np.flatnonzero(entity_ids == entity_id)
        if len(idx) < 3:
            continue
        dxi = np.gradient(x[idx], axis=0)
        ddxi = np.gradient(dxi, axis=0)
        dx[idx] = dxi
        ddx[idx] = ddxi
    return dx, ddx


def _rolling_entity_max(scores: np.ndarray, entity_ids: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return scores
    out = scores.copy()
    for entity_id in np.unique(entity_ids):
        idx = np.flatnonzero(entity_ids == entity_id)
        seq = scores[idx]
        padded = np.pad(seq, (radius, radius), mode="edge")
        rolled = np.empty_like(seq)
        for i in range(len(seq)):
            rolled[i] = np.max(padded[i : i + 2 * radius + 1])
        out[idx] = rolled
    return out


@dataclass(frozen=True)
class _ConceptStats:
    center_z: np.ndarray
    center_state: np.ndarray
    angular_median: float
    angular_scale: float
    energy_median: float
    energy_scale: float
    residual_median: float
    residual_scale: float
    velocity_median: float
    velocity_scale: float


class _GeometryAwareDynSphereDetector(BaseEstimator):
    """Shared multi-regime detector with geometry-specific latent operations.

    Parameters are deliberately explicit so the method can be used as a research
    baseline and ablated easily.
    """

    position_component_name = "angular"
    latent_dynamics_component_name = "spherical_dynamics"

    def __init__(
        self,
        n_concepts: int = 4,
        latent_dim: int = 5,
        ridge_alpha: float = 1.0,
        transition_smoothing: float = 0.5,
        random_state: int = 37,
        angular_weight: float = 1.0,
        dynamic_weight: float = 1.8,
        energy_weight: float = 1.0,
        velocity_weight: float = 0.8,
        transition_weight: float = 0.8,
        raw_dynamic_weight: float = 1.4,
        cluster_weight: float = 0.15,
        propagation_radius: int = 8,
    ) -> None:
        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.ridge_alpha = ridge_alpha
        self.transition_smoothing = transition_smoothing
        self.random_state = random_state
        self.angular_weight = angular_weight
        self.dynamic_weight = dynamic_weight
        self.energy_weight = energy_weight
        self.velocity_weight = velocity_weight
        self.transition_weight = transition_weight
        self.raw_dynamic_weight = raw_dynamic_weight
        self.cluster_weight = cluster_weight
        self.propagation_radius = propagation_radius

    def _latent_state(self, latent: np.ndarray) -> np.ndarray:
        return _safe_unit(latent)

    def _latent_derivatives(
        self,
        state: np.ndarray,
        entity_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _sequence_derivatives(state, entity_ids)

    def _equilibrium(self, state: np.ndarray) -> np.ndarray:
        return _safe_unit(np.mean(state, axis=0, keepdims=True))[0]

    def _concept_center(self, state: np.ndarray) -> np.ndarray:
        return _safe_unit(np.mean(state, axis=0, keepdims=True))[0]

    def _position_distance(self, state: np.ndarray, center: np.ndarray) -> np.ndarray:
        return np.arccos(np.clip(state @ center, -1.0, 1.0))

    def _potential(self, state: np.ndarray, equilibrium: np.ndarray) -> np.ndarray:
        return 1.0 - np.clip(state @ equilibrium, -1.0, 1.0)

    def fit(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> "_GeometryAwareDynSphereDetector":
        del times
        x = np.asarray(x, dtype=float)
        entity_ids = np.asarray(entity_ids)
        latent_dim = min(self.latent_dim, x.shape[1])

        self.feature_scaler_ = StandardScaler()
        xs = self.feature_scaler_.fit_transform(x)

        self.pca_ = PCA(n_components=latent_dim, random_state=self.random_state)
        latent = self.pca_.fit_transform(xs)
        z = self._latent_state(latent)
        v, a = self._latent_derivatives(z, entity_ids)
        dx, ddx = _euclidean_sequence_derivatives(xs, entity_ids)
        velocity = np.linalg.norm(v, axis=1)
        gravity = self._equilibrium(z)
        potential = self._potential(z, gravity)
        energy = 0.5 * velocity**2 + potential

        state = np.hstack([z, v])
        self.state_scaler_ = RobustScaler()
        state_scaled = self.state_scaler_.fit_transform(state)

        n_concepts = min(self.n_concepts, len(x))
        self.kmeans_ = KMeans(n_clusters=n_concepts, n_init=20, random_state=self.random_state)
        labels = self.kmeans_.fit_predict(state_scaled)
        self.gravity_ = gravity
        self.n_concepts_ = n_concepts

        self.models_: list[Ridge] = []
        self.raw_models_: list[Ridge] = []
        self.stats_: list[_ConceptStats] = []
        self.raw_residual_stats_: list[tuple[float, float]] = []
        raw_state = np.hstack([xs, dx])
        for concept in range(n_concepts):
            idx = labels == concept
            if idx.sum() < latent_dim + 4:
                idx = np.ones(len(labels), dtype=bool)

            model = Ridge(alpha=self.ridge_alpha, solver="lsqr")
            model.fit(state[idx], a[idx])
            pred = model.predict(state[idx])
            residual = np.linalg.norm(a[idx] - pred, axis=1)

            raw_model = Ridge(alpha=self.ridge_alpha, solver="lsqr")
            raw_model.fit(raw_state[idx], ddx[idx])
            raw_pred = raw_model.predict(raw_state[idx])
            raw_residual = np.linalg.norm(ddx[idx] - raw_pred, axis=1)

            concept_center = self._concept_center(z[idx])
            angle = self._position_distance(z[idx], concept_center)

            angular_median, angular_scale = _robust_location_scale(angle)
            energy_median, energy_scale = _robust_location_scale(energy[idx])
            residual_median, residual_scale = _robust_location_scale(residual)
            velocity_median, velocity_scale = _robust_location_scale(velocity[idx])
            raw_residual_median, raw_residual_scale = _robust_location_scale(raw_residual)

            self.models_.append(model)
            self.raw_models_.append(raw_model)
            self.raw_residual_stats_.append((raw_residual_median, raw_residual_scale))
            self.stats_.append(
                _ConceptStats(
                    center_z=concept_center,
                    center_state=self.kmeans_.cluster_centers_[concept],
                    angular_median=angular_median,
                    angular_scale=angular_scale,
                    energy_median=energy_median,
                    energy_scale=energy_scale,
                    residual_median=residual_median,
                    residual_scale=residual_scale,
                    velocity_median=velocity_median,
                    velocity_scale=velocity_scale,
                )
            )

        self.transition_log_penalty_ = self._fit_transition_penalties(labels, entity_ids, n_concepts)
        self.train_scores_ = self.decision_function(x, entity_ids)
        return self

    def decision_components(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Return the exact additive DynSphere score decomposition.

        Local component values are selected from the concept that minimizes the
        complete weighted local score. Consequently, the six local channels and
        transition channel sum exactly to ``pre_propagation_score``. The final
        ``score`` additionally applies the configured entity-wise rolling
        maximum, which is deliberately reported as a separate operation.
        """

        del times
        x = np.asarray(x, dtype=float)
        entity_ids = np.asarray(entity_ids)
        xs = self.feature_scaler_.transform(x)
        z = self._latent_state(self.pca_.transform(xs))
        v, a = self._latent_derivatives(z, entity_ids)
        dx, ddx = _euclidean_sequence_derivatives(xs, entity_ids)
        state = np.hstack([z, v])
        raw_state = np.hstack([xs, dx])
        state_scaled = self.state_scaler_.transform(state)
        assigned = self.kmeans_.predict(state_scaled)

        velocity = np.linalg.norm(v, axis=1)
        potential = self._potential(z, self.gravity_)
        energy = 0.5 * velocity**2 + potential

        component_matrices = {
            self.position_component_name: np.empty((len(x), self.n_concepts_), dtype=float),
            self.latent_dynamics_component_name: np.empty((len(x), self.n_concepts_), dtype=float),
            "raw_dynamics": np.empty((len(x), self.n_concepts_), dtype=float),
            "state_speed": np.empty((len(x), self.n_concepts_), dtype=float),
            "velocity": np.empty((len(x), self.n_concepts_), dtype=float),
            "regime_distance": np.empty((len(x), self.n_concepts_), dtype=float),
        }
        for concept, stats in enumerate(self.stats_):
            pred_a = self.models_[concept].predict(state)
            residual = np.linalg.norm(a - pred_a, axis=1)
            raw_pred = self.raw_models_[concept].predict(raw_state)
            raw_residual = np.linalg.norm(ddx - raw_pred, axis=1)
            raw_residual_median, raw_residual_scale = self.raw_residual_stats_[concept]
            angle = self._position_distance(z, stats.center_z)

            angular_score = np.maximum(0.0, (angle - stats.angular_median) / stats.angular_scale)
            dynamic_score = np.maximum(0.0, (residual - stats.residual_median) / stats.residual_scale)
            raw_dynamic_score = np.maximum(0.0, (raw_residual - raw_residual_median) / raw_residual_scale)
            energy_score = np.abs(energy - stats.energy_median) / stats.energy_scale
            velocity_score = np.abs(velocity - stats.velocity_median) / stats.velocity_scale
            cluster_score = np.linalg.norm(state_scaled - stats.center_state, axis=1)

            component_matrices[self.position_component_name][:, concept] = (
                self.angular_weight * angular_score
            )
            component_matrices[self.latent_dynamics_component_name][:, concept] = (
                self.dynamic_weight * dynamic_score
            )
            component_matrices["raw_dynamics"][:, concept] = (
                self.raw_dynamic_weight * raw_dynamic_score
            )
            component_matrices["state_speed"][:, concept] = self.energy_weight * energy_score
            component_matrices["velocity"][:, concept] = self.velocity_weight * velocity_score
            component_matrices["regime_distance"][:, concept] = (
                self.cluster_weight * cluster_score
            )

        local_matrix = sum(component_matrices.values())
        selected_concept = np.argmin(local_matrix, axis=1)
        rows = np.arange(len(x))
        selected = {
            name: values[rows, selected_concept]
            for name, values in component_matrices.items()
        }
        local_score = local_matrix[rows, selected_concept]
        transition = self.transition_weight * self._transition_penalty(assigned, entity_ids)
        pre_propagation_score = local_score + transition
        score = _rolling_entity_max(
            pre_propagation_score,
            entity_ids,
            self.propagation_radius,
        )
        return {
            **selected,
            "transition": transition,
            "local_score": local_score,
            "pre_propagation_score": pre_propagation_score,
            "score": score,
            "assigned_concept": assigned,
            "selected_concept": selected_concept,
        }

    def decision_function(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.decision_components(x, entity_ids, times)["score"]

    def predict(self, x: np.ndarray, entity_ids: np.ndarray, contamination: float = 0.05) -> np.ndarray:
        scores = self.decision_function(x, entity_ids)
        threshold = np.quantile(scores, 1.0 - contamination)
        return (scores >= threshold).astype(int)

    def _fit_transition_penalties(self, labels: np.ndarray, entity_ids: np.ndarray, n_concepts: int) -> np.ndarray:
        counts = np.full((n_concepts, n_concepts), self.transition_smoothing, dtype=float)
        for entity_id in np.unique(entity_ids):
            seq = labels[entity_ids == entity_id]
            if len(seq) < 2:
                continue
            for a, b in zip(seq[:-1], seq[1:]):
                counts[int(a), int(b)] += 1.0
        probs = counts / counts.sum(axis=1, keepdims=True)
        return -np.log(probs)

    def _transition_penalty(self, labels: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
        penalties = np.zeros(len(labels), dtype=float)
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            if len(idx) < 2:
                continue
            prev_labels = labels[idx[:-1]]
            next_labels = labels[idx[1:]]
            penalties[idx[1:]] = self.transition_log_penalty_[prev_labels, next_labels]
        return penalties


class MultiConceptPendulumDetector(_GeometryAwareDynSphereDetector):
    """Spherical multi-regime dynamic detector."""

    def __init__(
        self,
        n_concepts: int = 4,
        latent_dim: int = 5,
        ridge_alpha: float = 1.0,
        transition_smoothing: float = 0.5,
        random_state: int = 37,
        angular_weight: float = 1.0,
        dynamic_weight: float = 1.8,
        energy_weight: float = 1.0,
        velocity_weight: float = 0.8,
        transition_weight: float = 0.8,
        raw_dynamic_weight: float = 1.4,
        cluster_weight: float = 0.15,
        propagation_radius: int = 8,
    ) -> None:
        super().__init__(
            n_concepts=n_concepts,
            latent_dim=latent_dim,
            ridge_alpha=ridge_alpha,
            transition_smoothing=transition_smoothing,
            random_state=random_state,
            angular_weight=angular_weight,
            dynamic_weight=dynamic_weight,
            energy_weight=energy_weight,
            velocity_weight=velocity_weight,
            transition_weight=transition_weight,
            raw_dynamic_weight=raw_dynamic_weight,
            cluster_weight=cluster_weight,
            propagation_radius=propagation_radius,
        )
