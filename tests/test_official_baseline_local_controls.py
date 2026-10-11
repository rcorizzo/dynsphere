from __future__ import annotations

import numpy as np

from benchmarks.official_baselines import local_controls
from benchmarks.official_baselines.local_controls import (
    LOCAL_SCORERS,
    score_euclidean_predictor,
    score_kmeans_distance,
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
    requested: list[str] = []

    def fake_method_scorer(method_id: str):
        requested.append(method_id)
        return lambda train, test, indices, seed: np.arange(len(test), dtype=float) + seed

    monkeypatch.setattr(local_controls, "method_scorer", fake_method_scorer)
    train = np.zeros((24, 2))
    test = np.zeros((5, 2))

    scores, metadata = score_euclidean_predictor(train, test, seed=37)

    assert requested == ["predictive_euclidean_single"]
    np.testing.assert_array_equal(scores, np.arange(5, dtype=float) + 37)
    assert metadata == {"base_method": "predictive_euclidean_single"}


def test_one_regime_control_instantiates_existing_dynsphere(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeDynSphere:
        def __init__(self, **configuration):
            observed["configuration"] = configuration

        def fit(self, train, entities, times):
            observed["fit"] = (train.copy(), entities.copy(), times.copy())
            return self

        def decision_function(self, test, entities, times):
            observed["score"] = (test.copy(), entities.copy(), times.copy())
            return np.arange(len(test), dtype=float)

    monkeypatch.setattr(local_controls, "MultiConceptPendulumDetector", FakeDynSphere)
    train = np.arange(12, dtype=float).reshape(6, 2)
    test = np.arange(8, dtype=float).reshape(4, 2)

    scores, metadata = score_one_regime_dynsphere(train, test, seed=39)

    assert observed["configuration"] == {"n_concepts": 1, "random_state": 39}
    np.testing.assert_array_equal(scores, np.arange(4, dtype=float))
    assert metadata == {"n_concepts": 1}


def test_all_local_scorers_return_aligned_finite_scores(monkeypatch) -> None:
    monkeypatch.setattr(
        local_controls,
        "method_scorer",
        lambda method_id: lambda train, test, indices, seed: np.ones(len(test)),
    )

    class FakeDynSphere:
        def __init__(self, **configuration):
            pass

        def fit(self, train, entities, times):
            return self

        def decision_function(self, test, entities, times):
            return np.ones(len(test))

    monkeypatch.setattr(local_controls, "MultiConceptPendulumDetector", FakeDynSphere)
    train = np.arange(80, dtype=float).reshape(40, 2)
    test = np.arange(20, dtype=float).reshape(10, 2)

    assert set(LOCAL_SCORERS) == {
        "pca_reconstruction",
        "kmeans_distance",
        "predictive_euclidean_single",
        "dynsphere_one_regime",
    }
    for scorer in LOCAL_SCORERS.values():
        scores, metadata = scorer(train, test, 37)
        assert scores.shape == (len(test),)
        assert np.isfinite(scores).all()
        assert metadata
