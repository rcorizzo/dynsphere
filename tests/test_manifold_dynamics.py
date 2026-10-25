from __future__ import annotations

import unittest

import numpy as np

from generate_pendulum_dataset import distribution_preserving_index_map, generate_split
from manifold_dynamics_detector import (
    ManifoldDynamicsDetector,
    sphere_log_map,
    sphere_parallel_transport,
)


class SphereGeometryTests(unittest.TestCase):
    def test_log_map_is_tangent_and_has_geodesic_norm(self) -> None:
        base = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        displacement = sphere_log_map(base, target)

        np.testing.assert_allclose(np.sum(base * displacement, axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(
            np.linalg.norm(displacement, axis=1),
            np.full(2, np.pi / 2.0),
            atol=1e-10,
        )

    def test_parallel_transport_preserves_tangency_and_norm(self) -> None:
        base = np.array([[1.0, 0.0, 0.0]])
        target = np.array([[0.0, 1.0, 0.0]])
        vector = np.array([[0.0, 0.0, 2.0]])
        transported = sphere_parallel_transport(base, target, vector)

        np.testing.assert_allclose(np.sum(target * transported, axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(transported, axis=1), [2.0], atol=1e-10)


class DistributionPreservingGeneratorTests(unittest.TestCase):
    def test_anomalous_rows_are_exact_clean_rows(self) -> None:
        rng = np.random.default_rng(11)
        clean = rng.normal(size=(240, 8))
        schedule = np.repeat(np.arange(4), 60)
        index_map, labels, names = distribution_preserving_index_map(
            clean,
            schedule,
            rng,
            anomaly_fraction=0.15,
        )
        generated = clean[index_map]

        self.assertGreater(labels.sum(), 0)
        self.assertTrue(np.all(names[labels == 0] == "none"))
        np.testing.assert_array_equal(generated[labels == 1], clean[index_map[labels == 1]])
        for row in generated[labels == 1]:
            self.assertTrue(np.any(np.all(clean == row, axis=1)))

    def test_irregular_sampling_produces_strict_physical_timestamps(self) -> None:
        rows = generate_split(
            n_entities=2,
            n_steps=120,
            seed=13,
            anomaly_fraction=0.08,
            include_anomalies=True,
            distribution_preserving=True,
            sampling_interval=1.7,
            sampling_jitter=0.35,
            observation_noise_std=0.03,
        )
        times = np.asarray([row["time"] for row in rows if row["entity_id"] == 0], dtype=float)
        increments = np.diff(times)
        self.assertTrue(np.all(increments > 0.0))
        self.assertGreater(float(np.std(increments)), 0.0)
        self.assertAlmostEqual(float(np.mean(increments)), 1.7, delta=0.2)

    def test_timestamp_jitter_keeps_anomaly_labels_paired_by_seed(self) -> None:
        common = dict(
            n_entities=2,
            n_steps=360,
            seed=29,
            anomaly_fraction=0.05,
            include_anomalies=True,
            distribution_preserving=True,
        )
        regular = generate_split(**common, sampling_jitter=0.0)
        irregular = generate_split(**common, sampling_jitter=0.4)
        np.testing.assert_array_equal(
            [row["is_anomaly"] for row in regular],
            [row["is_anomaly"] for row in irregular],
        )
        np.testing.assert_array_equal(
            [row["anomaly_type"] for row in regular],
            [row["anomaly_type"] for row in irregular],
        )


class ManifoldDynamicsDetectorTests(unittest.TestCase):
    @staticmethod
    def _sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = []
        entities = []
        times = []
        for entity in range(3):
            t = np.cumsum(0.8 + 0.4 * np.sin(np.arange(90) * 0.17 + entity) ** 2)
            phase = 0.05 * t + 0.3 * entity
            x = np.column_stack(
                [
                    np.sin(phase),
                    np.cos(phase),
                    np.sin(2.0 * phase),
                    np.cos(0.5 * phase),
                    0.02 * t,
                    np.sin(phase + 0.4) * np.cos(phase),
                ]
            )
            features.append(x)
            entities.append(np.full(len(t), entity))
            times.append(t)
        return np.vstack(features), np.concatenate(entities), np.concatenate(times)

    def test_matched_geometry_variants_fit_and_score(self) -> None:
        x, entity_ids, times = self._sequences()
        for geometry in ("sphere", "euclidean"):
            for field_model in ("ridge", "local"):
                with self.subTest(geometry=geometry, field_model=field_model):
                    model = ManifoldDynamicsDetector(
                        geometry=geometry,
                        field_model=field_model,
                        local_neighbors=20,
                        latent_dim=4,
                        n_modes=3,
                        random_state=7,
                    )
                    model.fit(x, entity_ids, times)
                    scores = model.decision_function(x, entity_ids, times)

                    self.assertEqual(scores.shape, (len(x),))
                    self.assertTrue(np.isfinite(scores).all())
                    np.testing.assert_array_equal(scores[[0, 90, 180]], 0.0)
                    self.assertGreater(np.max(scores), 0.0)

    def test_invalid_field_model_is_rejected(self) -> None:
        x, entity_ids, times = self._sequences()
        model = ManifoldDynamicsDetector(field_model="unsupported")
        with self.assertRaisesRegex(ValueError, "field_model"):
            model.fit(x, entity_ids, times)

    def test_degenerate_mode_fit_falls_back_without_failure(self) -> None:
        x = np.ones((60, 6), dtype=float)
        entity_ids = np.zeros(len(x), dtype=int)
        times = np.arange(len(x), dtype=float)
        model = ManifoldDynamicsDetector(
            geometry="euclidean",
            latent_dim=3,
            n_modes=4,
            random_state=5,
        )
        model.fit(x, entity_ids, times)
        scores = model.decision_function(x, entity_ids, times)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreaterEqual(model.n_modes_, 1)
        self.assertLessEqual(model.n_modes_, 4)

    def test_non_increasing_entity_time_is_rejected(self) -> None:
        x, entity_ids, times = self._sequences()
        times[10] = times[9]
        model = ManifoldDynamicsDetector()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            model.fit(x, entity_ids, times)


if __name__ == "__main__":
    unittest.main()
