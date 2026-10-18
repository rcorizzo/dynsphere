"""Local baselines and controls under the canonical experiment contract."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from pendulum_detector import MultiConceptEuclideanDetector, MultiConceptPendulumDetector
from predictive_generative_detector import PredictiveGenerativeDetector


ScoreResult = tuple[np.ndarray, dict[str, object]]
LocalScorer = Callable[[np.ndarray, np.ndarray, int], ScoreResult]


def _arrays(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Training and test arrays must have matching feature dimensions")
    if len(train) < 2:
        raise ValueError("At least two training rows are required")
    return train, test


def _result(scores: np.ndarray, expected_rows: int, metadata: dict[str, object]) -> ScoreResult:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (expected_rows,) or not np.isfinite(scores).all():
        raise ValueError("Local scorer returned invalid scores")
    return scores, metadata


def score_pca_reconstruction(train: np.ndarray, test: np.ndarray, seed: int) -> ScoreResult:
    """Score squared residual after retaining 95% of training variance."""
    train, test = _arrays(train, test)
    maximum = min(len(train) - 1, train.shape[1])
    model = PCA(n_components=maximum, random_state=seed, svd_solver="full")
    model.fit(train)
    ratios = np.nan_to_num(model.explained_variance_ratio_, nan=0.0)
    cumulative = np.cumsum(ratios)
    n_components = int(np.searchsorted(cumulative, 0.95, side="left") + 1)
    n_components = min(max(n_components, 1), maximum)
    components = model.components_[:n_components]
    centered = test - model.mean_
    reconstructed = (centered @ components.T) @ components + model.mean_
    scores = np.sum((test - reconstructed) ** 2, axis=1)
    return _result(
        scores,
        len(test),
        {
            "variance_retained": 0.95,
            "n_components": n_components,
            "score": "squared-reconstruction-residual",
        },
    )


def score_kmeans_distance(train: np.ndarray, test: np.ndarray, seed: int) -> ScoreResult:
    """Score Euclidean distance to the nearest training centroid."""
    train, test = _arrays(train, test)
    n_clusters = min(4, len(train))
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    model.fit(train)
    distances = model.transform(test)
    scores = distances.min(axis=1)
    return _result(
        scores,
        len(test),
        {"n_clusters": n_clusters, "n_init": 20, "score": "nearest-centroid-distance"},
    )


def score_euclidean_predictor(train: np.ndarray, test: np.ndarray, seed: int) -> ScoreResult:
    """Use the retained single-mode Euclidean predictor without duplicated logic."""
    train, test = _arrays(train, test)
    model = PredictiveGenerativeDetector(
        geometry="euclidean",
        latent_dim=min(6, train.shape[1]),
        hidden_dim=48,
        n_modes=1,
        epochs=15,
        sequence_length=128,
        calibration_fraction=0.2,
        random_state=seed,
    )
    model.fit(
        train,
        np.zeros(len(train), dtype=int),
        np.arange(len(train), dtype=float),
    )
    scores = model.decision_function(
        test,
        np.zeros(len(test), dtype=int),
        np.arange(len(test), dtype=float),
    )
    return _result(
        scores,
        len(test),
        {"base_method": "predictive_euclidean_single"},
    )


DYNSPHERE_WEIGHTS = {
    "position": 1.0,
    "latent_dynamics": 1.8,
    "raw_dynamics": 1.4,
    "state_speed": 1.0,
    "velocity": 0.8,
    "transition": 0.8,
    "regime_distance": 0.15,
}
DYNSPHERE_CONFIGURATION = {
    "latent_dim": 5,
    "ridge_alpha": 1.0,
    "transition_smoothing": 0.5,
    "dynamic_weight": DYNSPHERE_WEIGHTS["latent_dynamics"],
    "energy_weight": DYNSPHERE_WEIGHTS["state_speed"],
    "velocity_weight": DYNSPHERE_WEIGHTS["velocity"],
    "transition_weight": DYNSPHERE_WEIGHTS["transition"],
    "raw_dynamic_weight": DYNSPHERE_WEIGHTS["raw_dynamics"],
    "cluster_weight": DYNSPHERE_WEIGHTS["regime_distance"],
    "propagation_radius": 8,
}


def _score_dynsphere_family(
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    *,
    detector_type: type[MultiConceptPendulumDetector] | type[MultiConceptEuclideanDetector],
    geometry: str,
    n_concepts: int,
) -> ScoreResult:
    train, test = _arrays(train, test)
    train_entities = np.zeros(len(train), dtype=int)
    test_entities = np.zeros(len(test), dtype=int)
    configuration = {
        **DYNSPHERE_CONFIGURATION,
        "n_concepts": n_concepts,
        "random_state": seed,
    }
    if geometry == "spherical":
        configuration["angular_weight"] = DYNSPHERE_WEIGHTS["position"]
    else:
        configuration["position_weight"] = DYNSPHERE_WEIGHTS["position"]
    model = detector_type(**configuration)
    model.fit(train, train_entities, np.arange(len(train)))
    scores = model.decision_function(test, test_entities, np.arange(len(test)))
    return _result(
        scores,
        len(test),
        {
            "geometry": geometry,
            "n_concepts": n_concepts,
            "latent_dim": DYNSPHERE_CONFIGURATION["latent_dim"],
            "ridge_alpha": DYNSPHERE_CONFIGURATION["ridge_alpha"],
            "transition_smoothing": DYNSPHERE_CONFIGURATION["transition_smoothing"],
            "propagation_radius": DYNSPHERE_CONFIGURATION["propagation_radius"],
            "weights": dict(DYNSPHERE_WEIGHTS),
        },
    )


def score_dynsphere(train: np.ndarray, test: np.ndarray, seed: int) -> ScoreResult:
    """Score with four-concept spherical DynSphere."""
    return _score_dynsphere_family(
        train,
        test,
        seed,
        detector_type=MultiConceptPendulumDetector,
        geometry="spherical",
        n_concepts=4,
    )


def score_euclidean_dynsphere(train: np.ndarray, test: np.ndarray, seed: int) -> ScoreResult:
    """Score with four-concept Euclidean DynSphere."""
    return _score_dynsphere_family(
        train,
        test,
        seed,
        detector_type=MultiConceptEuclideanDetector,
        geometry="euclidean",
        n_concepts=4,
    )


def score_one_regime_dynsphere(train: np.ndarray, test: np.ndarray, seed: int) -> ScoreResult:
    """Score with one-concept spherical DynSphere."""
    return _score_dynsphere_family(
        train,
        test,
        seed,
        detector_type=MultiConceptPendulumDetector,
        geometry="spherical",
        n_concepts=1,
    )


def score_one_regime_euclidean_dynsphere(
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> ScoreResult:
    """Score with one-concept Euclidean DynSphere."""
    return _score_dynsphere_family(
        train,
        test,
        seed,
        detector_type=MultiConceptEuclideanDetector,
        geometry="euclidean",
        n_concepts=1,
    )


LOCAL_SCORERS: dict[str, LocalScorer] = {
    "pca_reconstruction": score_pca_reconstruction,
    "kmeans_distance": score_kmeans_distance,
    "predictive_euclidean_single": score_euclidean_predictor,
    "dynsphere": score_dynsphere,
    "dynsphere_euclidean": score_euclidean_dynsphere,
    "dynsphere_one_regime": score_one_regime_dynsphere,
    "dynsphere_euclidean_one_regime": score_one_regime_euclidean_dynsphere,
}
