#!/usr/bin/env python3
"""Semi-Markov marginalization of causal change-point evidence.

Each candidate event age defines a continuation-versus-reset likelihood ratio.
A geometric survival prior supplies the semi-Markov duration model, and the
ratios are marginalized to infer whether an event is currently active. The
procedure propagates a boundary only through an explicit duration posterior;
it does not smooth or copy anomaly scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from boundary_change_point_detector import BoundaryChangePointDetector
from segment_predictive_detector import SegmentPredictiveDetector


class SemiMarkovChangePointDetector(BaseEstimator):
    """Online posterior odds of an active reset episode."""

    def __init__(
        self,
        max_duration: int = 32,
        mean_duration: float = 16.0,
        boundary_hazard: float = 0.01,
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
        self.max_duration = max_duration
        self.mean_duration = mean_duration
        self.boundary_hazard = boundary_hazard
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
    ) -> "SemiMarkovChangePointDetector":
        self._validate_configuration()
        x, entity_ids, times = SegmentPredictiveDetector._validate_inputs(
            x, entity_ids, times
        )
        self.horizons_ = tuple(range(1, self.max_duration + 1))
        self.boundary_detector_ = BoundaryChangePointDetector(
            horizons=self.horizons_,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            rollout_epochs=self.rollout_epochs,
            rollout_learning_rate=self.rollout_learning_rate,
            reset_components=self.reset_components,
            calibration_fraction=self.calibration_fraction,
            random_state=self.random_state,
        ).fit(x, entity_ids, times)
        self.base_detector_ = self.boundary_detector_.base_detector_
        self._fit_log_age_prior()
        xs = self.base_detector_.feature_scaler_.transform(x).astype(np.float32)
        evidence, _, _, _ = self.boundary_detector_._boundary_likelihoods(
            xs, entity_ids, times
        )
        log_odds, _ = self._marginal_log_odds(evidence)
        calibration = log_odds[~self.base_detector_.fit_mask_]
        calibration = calibration[np.isfinite(calibration)]
        if len(calibration) < 2:
            raise ValueError("Semi-Markov calibration produced fewer than two scores.")
        self.score_center_, self.score_scale_ = (
            SegmentPredictiveDetector._robust_location_scale(calibration)
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
        evidence, continuation, reset_segment, _ = (
            self.boundary_detector_._boundary_likelihoods(xs, entity_ids, times)
        )
        log_odds, age_posterior = self._marginal_log_odds(evidence)
        calibrated = (log_odds - self.score_center_) / self.score_scale_
        posterior_probability = self._sigmoid(log_odds)
        best_age = np.argmax(age_posterior, axis=1) + 1
        unavailable = ~np.isfinite(evidence).any(axis=1)
        best_age[unavailable] = 0
        for entity_id in np.unique(entity_ids):
            first = np.flatnonzero(entity_ids == entity_id)[0]
            calibrated[first] = 0.0
            posterior_probability[first] = 0.0
            best_age[first] = 0
        return {
            "calibrated_score": calibrated,
            "active_event_log_odds": log_odds,
            "active_event_probability": posterior_probability,
            "best_event_age": best_age,
            "age_posterior": age_posterior,
            "boundary_evidence": evidence,
            "continuation_nll": continuation,
            "reset_segment_nll": reset_segment,
        }

    def _fit_log_age_prior(self) -> None:
        survival = 1.0 - 1.0 / self.mean_duration
        ages = np.arange(self.max_duration, dtype=float)
        self.log_age_prior_odds_ = (
            np.log(self.boundary_hazard)
            - np.log1p(-self.boundary_hazard)
            + ages * np.log(survival)
        )

    def _marginal_log_odds(
        self, evidence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        log_terms = evidence + self.log_age_prior_odds_[None, :]
        available = np.isfinite(log_terms)
        candidates = np.where(available, log_terms, -np.inf)
        maximum = np.max(candidates, axis=1)
        no_candidate = ~available.any(axis=1)
        maximum[no_candidate] = 0.0
        shifted = np.where(available, candidates - maximum[:, None], -np.inf)
        normalizer = np.sum(np.exp(shifted), axis=1)
        log_odds = maximum + np.log(np.maximum(normalizer, 1e-300))
        log_odds[no_candidate] = 0.0
        age_posterior = np.zeros_like(evidence)
        valid_rows = ~no_candidate
        age_posterior[valid_rows] = (
            np.exp(shifted[valid_rows]) / normalizer[valid_rows, None]
        )
        return log_odds, age_posterior

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        output = np.empty_like(values, dtype=float)
        positive = values >= 0.0
        output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponent = np.exp(values[~positive])
        output[~positive] = exponent / (1.0 + exponent)
        return output

    def _validate_configuration(self) -> None:
        if self.max_duration < 2:
            raise ValueError("max_duration must be at least two.")
        if self.mean_duration <= 1.0:
            raise ValueError("mean_duration must exceed one.")
        if not 0.0 < self.boundary_hazard < 1.0:
            raise ValueError("boundary_hazard must be in (0, 1).")
