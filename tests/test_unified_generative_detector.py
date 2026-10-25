from __future__ import annotations

import unittest

import numpy as np

from unified_generative_detector import UnifiedGenerativeDetector


class UnifiedGenerativeDetectorTests(unittest.TestCase):
    @staticmethod
    def _sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = []
        entities = []
        times = []
        for entity in range(3):
            t = np.cumsum(0.8 + 0.3 * np.sin(np.arange(64) * 0.13 + entity) ** 2)
            phase = 0.08 * t + 0.4 * entity
            x = np.column_stack(
                [
                    np.sin(phase),
                    np.cos(phase),
                    np.sin(2.0 * phase),
                    np.cos(0.5 * phase),
                    0.02 * t,
                    np.sin(phase + 0.3) * np.cos(phase),
                ]
            )
            observations.append(x)
            entities.append(np.full(len(t), entity))
            times.append(t)
        return np.vstack(observations), np.concatenate(entities), np.concatenate(times)

    def test_joint_likelihood_is_exact_factorization(self) -> None:
        x, entities, times = self._sequences()
        model = UnifiedGenerativeDetector(
            geometry="euclidean",
            latent_dim=4,
            hidden_dim=12,
            n_modes=3,
            sequence_length=32,
            epochs=2,
            random_state=11,
        ).fit(x, entities, times)
        components = model.decision_components(x, entities, times)

        np.testing.assert_allclose(
            components["joint_nll"],
            components["observation_nll"] + components["dynamics_nll"],
            atol=1e-5,
        )
        self.assertEqual(components["calibrated_score"].shape, (len(x),))
        self.assertTrue(np.isfinite(components["calibrated_score"]).all())
        self.assertGreater(np.unique(components["calibrated_score"]).size, 2)
        self.assertEqual(len(model.training_loss_), 2)
        observation_scores = model.decision_function(
            x,
            entities,
            times,
            score_factor="observation",
        )
        self.assertTrue(np.isfinite(observation_scores).all())

    def test_matched_geometries_and_factor_ablations_score(self) -> None:
        x, entities, times = self._sequences()
        for geometry in ("euclidean", "sphere"):
            for factor in ("joint", "observation", "dynamics"):
                with self.subTest(geometry=geometry, factor=factor):
                    model = UnifiedGenerativeDetector(
                        geometry=geometry,
                        score_factor=factor,
                        latent_dim=3,
                        hidden_dim=10,
                        n_modes=2,
                        sequence_length=48,
                        epochs=1,
                        random_state=5,
                    ).fit(x, entities, times)
                    scores = model.decision_function(x, entities, times)
                    self.assertEqual(scores.shape, (len(x),))
                    self.assertTrue(np.isfinite(scores).all())

    def test_non_increasing_time_is_rejected(self) -> None:
        x, entities, times = self._sequences()
        times[8] = times[7]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            UnifiedGenerativeDetector(epochs=1).fit(x, entities, times)


if __name__ == "__main__":
    unittest.main()
