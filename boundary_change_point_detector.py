#!/usr/bin/env python3
"""Causal boundary/change-point likelihood ratio for trajectory anomalies.

The continuation hypothesis free-runs the learned normal dynamics from the
state before a candidate boundary. The reset hypothesis draws the first state
after that boundary from the normal marginal density and then follows ordinary
teacher-forced normal transitions. Their likelihood ratio targets a displaced
but internally plausible segment without combining independently weighted
anomaly scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture

from segment_predictive_detector import SegmentPredictiveDetector


class BoundaryChangePointDetector(BaseEstimator):
    """Generalized likelihood-ratio scan over recent candidate boundaries."""

    def __init__(
        self,
        horizons: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        latent_dim: int = 6,
        hidden_dim: int = 48,
        sequence_length: int = 128,
        epochs: int = 15,
        learning_rate: float = 1e-3,
        rollout_epochs: int = 10,
        rollout_learning_rate: float = 5e-4,
        reset_components: int = 4,
        calibration_fraction: float = 0.2,
        random_state: int = 37,
    ) -> None:
        self.horizons = horizons
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.rollout_epochs = rollout_epochs
        self.rollout_learning_rate = rollout_learning_rate
        self.reset_components = reset_components
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def fit(
        self,
        x: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray | None = None,
    ) -> "BoundaryChangePointDetector":
        self._validate_configuration()
        x, entity_ids, times = SegmentPredictiveDetector._validate_inputs(
            x, entity_ids, times
        )
        rollout_horizons = tuple(
            sorted({max(2, horizon) for horizon in self.horizons_})
        )
        self.segment_detector_ = SegmentPredictiveDetector(
            horizons=rollout_horizons,
            geometry="euclidean",
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            rollout_epochs=self.rollout_epochs,
            rollout_learning_rate=self.rollout_learning_rate,
            calibration_fraction=self.calibration_fraction,
            random_state=self.random_state,
        ).fit(x, entity_ids, times)
        self.base_detector_ = self.segment_detector_.base_detector_
        xs = self.base_detector_.feature_scaler_.transform(x).astype(np.float32)
        fit_mask = self.base_detector_.fit_mask_
        component_count = min(self.reset_components, int(fit_mask.sum()))
        self.reset_model_ = GaussianMixture(
            n_components=component_count,
            covariance_type="diag",
            reg_covar=1e-5,
            random_state=self.random_state,
        ).fit(xs[fit_mask])

        evidence, _, _, _ = self._boundary_likelihoods(xs, entity_ids, times)
        calibration_mask = ~fit_mask
        self.horizon_centers_ = np.zeros(len(self.horizons_), dtype=float)
        self.horizon_scales_ = np.ones(len(self.horizons_), dtype=float)
        for column, horizon in enumerate(self.horizons_):
            valid = calibration_mask & np.isfinite(evidence[:, column])
            reference = evidence[valid, column]
            if len(reference) < 2:
                raise ValueError(
                    f"Horizon {horizon} has insufficient boundary calibration rows."
                )
            center, scale = SegmentPredictiveDetector._robust_location_scale(
                reference
            )
            self.horizon_centers_[column] = center
            self.horizon_scales_[column] = scale
        scan, _ = self._scan_statistic(evidence)
        reference = scan[calibration_mask & np.isfinite(scan)]
        if len(reference) < 2:
            raise ValueError("Boundary scan has insufficient calibration scores.")
        self.scan_center_, self.scan_scale_ = (
            SegmentPredictiveDetector._robust_location_scale(reference)
        )
        return self

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
        x, entity_ids, times = SegmentPredictiveDetector._validate_inputs(
            x, entity_ids, times
        )
        xs = self.base_detector_.feature_scaler_.transform(x).astype(np.float32)
        evidence, continuation, reset_segment, reset_nll = self._boundary_likelihoods(
            xs, entity_ids, times
        )
        scan, best_horizon = self._scan_statistic(evidence)
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
            "boundary_evidence": evidence,
            "continuation_nll": continuation,
            "reset_segment_nll": reset_segment,
            "reset_nll": reset_nll,
        }

    def _boundary_likelihoods(
        self,
        xs: np.ndarray,
        entity_ids: np.ndarray,
        times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        free_nll, reanchored_nll = self.segment_detector_._segment_likelihoods(
            xs, entity_ids, times
        )
        rollout_columns = {
            horizon: column
            for column, horizon in enumerate(self.segment_detector_.horizons_)
        }
        one_step = self.base_detector_._raw_components(
            xs, entity_ids, times
        )["predictive_nll"]
        reset_nll = -self.reset_model_.score_samples(xs)
        continuation = np.full(
            (len(xs), len(self.horizons_)), np.nan, dtype=float
        )
        reset_segment = np.full_like(continuation, np.nan)
        for entity_id in np.unique(entity_ids):
            idx = np.flatnonzero(entity_ids == entity_id)
            length = len(idx)
            for column, horizon in enumerate(self.horizons_):
                if length <= horizon:
                    continue
                endpoints = idx[horizon:]
                boundary_points = idx[1 : length - horizon + 1]
                if horizon == 1:
                    continuation[endpoints, column] = one_step[endpoints]
                    reset_segment[endpoints, column] = reset_nll[boundary_points]
                    continue
                rollout_column = rollout_columns[horizon]
                continuation[endpoints, column] = free_nll[
                    endpoints, rollout_column
                ]
                teacher_forced = reanchored_nll[endpoints, rollout_column]
                first_transition = one_step[boundary_points]
                reset_segment[endpoints, column] = (
                    teacher_forced
                    - first_transition
                    + reset_nll[boundary_points]
                )
        evidence = continuation - reset_segment
        return evidence, continuation, reset_segment, reset_nll

    def _scan_statistic(
        self, evidence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        standardized = (
            evidence - self.horizon_centers_[None, :]
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

    def _validate_configuration(self) -> None:
        horizons = tuple(sorted(set(int(value) for value in self.horizons)))
        if not horizons or horizons[0] < 1:
            raise ValueError("horizons must contain positive integers.")
        if self.reset_components < 1:
            raise ValueError("reset_components must be positive.")
        self.horizons_ = horizons
