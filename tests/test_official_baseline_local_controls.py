from __future__ import annotations

import numpy as np

from benchmarks.official_baselines import local_controls
from benchmarks.official_baselines.local_controls import (
    LOCAL_SCORERS,
    score_dynsphere,
    score_euclidean_dynsphere,
    score_euclidean_predictor,
    score_kmeans_distance,
    score_one_regime_euclidean_dynsphere,
    score_one_regime_dynsphere,
    score_pca_reconstruction,
)


def test_pca_uses_95_percent_variance_and_squared_reconstruction_residual() -> None:
    train = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    test = np.array([[0.0, 2.0], [2.0, 0.0]])

    scores, metadata = score_pca_reconstruction(train, test, seed=37)

    np.testing.assert_allclose(scores, np.array([4.0, 0.0]), atol=1e-12)
    assert metadata == {
        "variance_retained": 0.95,
        "n_components": 1,
        "score": "squared-reconstruction-residual",
    }


def test_kmeans_uses_fixed_capacity_and_nearest_center_distance() -> None:
    train = np.array([[0.0], [10.0]])
    test = np.array([[1.0], [7.0]])

    scores, metadata = score_kmeans_distance(train, test, seed=37)

    np.testing.assert_allclose(scores, np.array([1.0, 3.0]), atol=1e-12)
    assert metadata == {"n_clusters": 2, "n_init": 20, "score": "nearest-centroid-distance"}


def test_euclidean_control_uses_retained_single_mode_scorer(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakePredictor:
        def __init__(self, **configuration):
            observed["configuration"] = configuration

        def fit(self, train, entities, times):
            observed["fit"] = (train.copy(), entities.copy(), times.copy())
            return self

        def decision_function(self, test, entities, times):
            observed["score"] = (test.copy(), entities.copy(), times.copy())
            return np.arange(len(test), dtype=float) + 37

    monkeypatch.setattr(local_controls, "PredictiveGenerativeDetector", FakePredictor)
    train = np.zeros((24, 2))
    test = np.zeros((5, 2))

    scores, metadata = score_euclidean_predictor(train, test, seed=37)

    assert observed["configuration"] == {
        "geometry": "euclidean",
        "latent_dim": 2,
        "hidden_dim": 48,
        "n_modes": 1,
        "epochs": 15,
        "sequence_length": 128,
        "calibration_fraction": 0.2,
        "random_state": 37,
    }
    np.testing.assert_array_equal(scores, np.arange(5, dtype=float) + 37)
    assert metadata == {"base_method": "predictive_euclidean_single"}


def test_dynsphere_family_uses_matched_geometry_and_concept_configuration(monkeypatch) -> None:
    observed: list[tuple[str, dict[str, object]]] = []

    class FakeSpherical:
        def __init__(self, **configuration):
            observed.append(("spherical", configuration))

        def fit(self, train, entities, times):
            return self

        def decision_function(self, test, entities, times):
            return np.arange(len(test), dtype=float)

    class FakeEuclidean(FakeSpherical):
        def __init__(self, **configuration):
            observed.append(("euclidean", configuration))

    monkeypatch.setattr(local_controls, "MultiConceptPendulumDetector", FakeSpherical)
    monkeypatch.setattr(local_controls, "MultiConceptEuclideanDetector", FakeEuclidean)
    train = np.arange(12, dtype=float).reshape(6, 2)
    test = np.arange(8, dtype=float).reshape(4, 2)

    scorers = (
        (score_dynsphere, "spherical", 4),
        (score_euclidean_dynsphere, "euclidean", 4),
        (score_one_regime_dynsphere, "spherical", 1),
        (score_one_regime_euclidean_dynsphere, "euclidean", 1),
    )
    for scorer, geometry, n_concepts in scorers:
        scores, metadata = scorer(train, test, seed=39)
        np.testing.assert_array_equal(scores, np.arange(4, dtype=float))
        assert metadata["geometry"] == geometry
        assert metadata["n_concepts"] == n_concepts
        assert metadata["latent_dim"] == 5
        assert metadata["ridge_alpha"] == 1.0
        assert metadata["transition_smoothing"] == 0.5
        assert metadata["propagation_radius"] == 8
        assert metadata["weights"] == {
            "position": 1.0,
            "latent_dynamics": 1.8,
            "raw_dynamics": 1.4,
            "state_speed": 1.0,
            "velocity": 0.8,
            "transition": 0.8,
            "regime_distance": 0.15,
        }

    common = {
        "latent_dim": 5,
        "ridge_alpha": 1.0,
        "transition_smoothing": 0.5,
        "dynamic_weight": 1.8,
        "energy_weight": 1.0,
        "velocity_weight": 0.8,
        "transition_weight": 0.8,
        "raw_dynamic_weight": 1.4,
        "cluster_weight": 0.15,
        "propagation_radius": 8,
        "random_state": 39,
    }
    assert observed == [
        ("spherical", {**common, "n_concepts": 4, "angular_weight": 1.0}),
        ("euclidean", {**common, "n_concepts": 4, "position_weight": 1.0}),
        ("spherical", {**common, "n_concepts": 1, "angular_weight": 1.0}),
        ("euclidean", {**common, "n_concepts": 1, "position_weight": 1.0}),
    ]


def test_all_local_scorers_return_aligned_finite_scores(monkeypatch) -> None:
    class FakePredictor:
        def __init__(self, **configuration):
            pass

        def fit(self, train, entities, times):
            return self

        def decision_function(self, test, entities, times):
            return np.ones(len(test))

    monkeypatch.setattr(local_controls, "PredictiveGenerativeDetector", FakePredictor)

    class FakeDynSphere:
        def __init__(self, **configuration):
            pass

        def fit(self, train, entities, times):
            return self

        def decision_function(self, test, entities, times):
            return np.ones(len(test))

    monkeypatch.setattr(local_controls, "MultiConceptPendulumDetector", FakeDynSphere)
    monkeypatch.setattr(local_controls, "MultiConceptEuclideanDetector", FakeDynSphere)
    train = np.arange(80, dtype=float).reshape(40, 2)
    test = np.arange(20, dtype=float).reshape(10, 2)

    assert set(LOCAL_SCORERS) == {
        "pca_reconstruction",
        "kmeans_distance",
        "predictive_euclidean_single",
        "dynsphere",
        "dynsphere_euclidean",
        "dynsphere_one_regime",
        "dynsphere_euclidean_one_regime",
    }
    for scorer in LOCAL_SCORERS.values():
        scores, metadata = scorer(train, test, 37)
        assert scores.shape == (len(test),)
        assert np.isfinite(scores).all()
        assert metadata
