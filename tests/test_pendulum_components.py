import base64
import io
import unittest

import numpy as np

from pendulum_detector import MultiConceptEuclideanDetector, MultiConceptPendulumDetector


SPHERICAL_SNAPSHOT = (
    "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"
)


class DynSphereComponentTests(unittest.TestCase):
    @staticmethod
    def sequence(rows: int = 240) -> tuple[np.ndarray, np.ndarray]:
        time = np.linspace(0.0, 8.0 * np.pi, rows)
        x = np.column_stack(
            [
                np.sin(time),
                np.cos(time),
                np.sin(0.5 * time + 0.2),
                np.cos(0.25 * time - 0.4),
                0.1 * np.sin(2.0 * time),
                0.1 * np.cos(1.5 * time),
            ]
        )
        return x, np.zeros(rows, dtype=int)

    def test_component_sum_matches_unpropagated_score(self) -> None:
        x, entities = self.sequence()
        model = MultiConceptPendulumDetector(
            n_concepts=3,
            latent_dim=4,
            propagation_radius=0,
            random_state=37,
        ).fit(x[:160], entities[:160])
        components = model.decision_components(x, entities)
        additive_names = (
            "angular",
            "spherical_dynamics",
            "raw_dynamics",
            "state_speed",
            "velocity",
            "regime_distance",
            "transition",
        )
        reconstructed = sum(components[name] for name in additive_names)
        np.testing.assert_allclose(reconstructed, components["pre_propagation_score"])
        np.testing.assert_allclose(components["score"], model.decision_function(x, entities))

    def test_spherical_characterization_snapshot(self) -> None:
        x, entities = self.sequence(rows=18)
        model = MultiConceptPendulumDetector(
            n_concepts=3,
            latent_dim=4,
            propagation_radius=0,
            random_state=37,
        ).fit(x[:14], entities[:14])
        actual = model.decision_components(x, entities)
        with np.load(io.BytesIO(base64.b64decode(SPHERICAL_SNAPSHOT))) as snapshot:
            self.assertEqual(set(snapshot.files), set(actual))
            for name in snapshot.files:
                np.testing.assert_allclose(
                    actual[name],
                    snapshot[name],
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=name,
                )

    def test_zero_latent_norm_produces_finite_components(self) -> None:
        x = np.zeros((16, 4), dtype=float)
        entities = np.zeros(len(x), dtype=int)
        model = MultiConceptPendulumDetector(
            n_concepts=2,
            latent_dim=3,
            propagation_radius=0,
            random_state=37,
        ).fit(x, entities)
        components = model.decision_components(x, entities)
        for name, values in components.items():
            self.assertTrue(np.all(np.isfinite(values)), name)

    def test_euclidean_geometry_definitions(self) -> None:
        detector = MultiConceptEuclideanDetector(propagation_radius=0)
        latent = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]])
        entities = np.zeros(3, dtype=int)
        np.testing.assert_array_equal(detector._latent_state(latent), latent)
        velocity, acceleration = detector._latent_derivatives(latent, entities)
        expected_velocity = np.gradient(latent, axis=0)
        np.testing.assert_allclose(velocity, expected_velocity)
        np.testing.assert_allclose(acceleration, np.gradient(expected_velocity, axis=0))
        center = detector._concept_center(latent)
        np.testing.assert_allclose(center, np.mean(latent, axis=0))
        np.testing.assert_allclose(
            detector._position_distance(latent, center),
            np.linalg.norm(latent - center, axis=1),
        )
        equilibrium = np.array([2.0, 4.0])
        np.testing.assert_allclose(
            detector._potential(latent, equilibrium),
            0.5 * np.sum((latent - equilibrium) ** 2, axis=1),
        )

    def test_common_behavior_for_both_geometries(self) -> None:
        x, _ = self.sequence(rows=60)
        entities = np.repeat([0, 1], 30)
        for detector_type, position_name, dynamics_name in (
            (MultiConceptPendulumDetector, "angular", "spherical_dynamics"),
            (MultiConceptEuclideanDetector, "position", "euclidean_dynamics"),
        ):
            with self.subTest(detector=detector_type.__name__):
                model = detector_type(
                    n_concepts=1,
                    latent_dim=3,
                    propagation_radius=0,
                    random_state=37,
                ).fit(x[:40], entities[:40])
                components = model.decision_components(x, entities)
                additive_names = (
                    position_name,
                    dynamics_name,
                    "raw_dynamics",
                    "state_speed",
                    "velocity",
                    "regime_distance",
                    "transition",
                )
                reconstructed = sum(components[name] for name in additive_names)
                np.testing.assert_allclose(
                    reconstructed,
                    components["pre_propagation_score"],
                )
                self.assertEqual(components["score"].shape, (len(x),))
                self.assertTrue(np.all(np.isfinite(components["score"])))
                np.testing.assert_allclose(components["transition"], 0.0)
                self.assertEqual(components["transition"][30], 0.0)

                state = model._latent_state(
                    model.pca_.transform(model.feature_scaler_.transform(x))
                )
                velocity, acceleration = model._latent_derivatives(state, entities)
                isolated_velocity, isolated_acceleration = model._latent_derivatives(
                    state[30:],
                    np.zeros(30, dtype=int),
                )
                np.testing.assert_allclose(velocity[30:], isolated_velocity)
                np.testing.assert_allclose(acceleration[30:], isolated_acceleration)

    def test_small_concepts_use_all_training_fallback(self) -> None:
        x, entities = self.sequence(rows=8)
        for detector_type in (MultiConceptPendulumDetector, MultiConceptEuclideanDetector):
            with self.subTest(detector=detector_type.__name__):
                model = detector_type(
                    n_concepts=4,
                    latent_dim=2,
                    propagation_radius=0,
                    random_state=37,
                ).fit(x, entities)
                scores = model.decision_function(x, entities)
                self.assertTrue(np.all(np.isfinite(scores)))
                for fitted in model.models_[1:]:
                    np.testing.assert_allclose(fitted.coef_, model.models_[0].coef_)
                for fitted in model.raw_models_[1:]:
                    np.testing.assert_allclose(fitted.coef_, model.raw_models_[0].coef_)

    def test_latent_magnitude_is_visible_only_in_euclidean_geometry(self) -> None:
        spherical = MultiConceptPendulumDetector()
        euclidean = MultiConceptEuclideanDetector()
        latent = np.array([[1.0, 2.0, -1.0]])
        scaled = 3.5 * latent
        np.testing.assert_allclose(
            spherical._latent_state(latent),
            spherical._latent_state(scaled),
        )
        self.assertFalse(
            np.allclose(
                euclidean._latent_state(latent),
                euclidean._latent_state(scaled),
            )
        )
        equilibrium = np.zeros(3)
        self.assertNotEqual(
            float(euclidean._potential(latent, equilibrium)[0]),
            float(euclidean._potential(scaled, equilibrium)[0]),
        )

    def test_propagation_is_reported_separately(self) -> None:
        x, entities = self.sequence()
        model = MultiConceptPendulumDetector(
            n_concepts=2,
            latent_dim=3,
            propagation_radius=3,
            random_state=37,
        ).fit(x[:160], entities[:160])
        components = model.decision_components(x, entities)
        self.assertEqual(components["score"].shape, (len(x),))
        self.assertTrue(np.all(components["score"] >= components["pre_propagation_score"]))
        self.assertTrue(np.issubdtype(components["assigned_concept"].dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
