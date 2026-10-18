from __future__ import annotations

import unittest

import numpy as np

from predictive_generative_detector import PredictiveGenerativeDetector


class PredictiveGenerativeDetectorTests(unittest.TestCase):
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

    def test_matched_geometries_and_mode_counts_fit(self) -> None:
        x, entities, times = self._sequences()
        for geometry in ("euclidean", "sphere"):
            for modes in (1, 3):
                with self.subTest(geometry=geometry, modes=modes):
                    model = PredictiveGenerativeDetector(
                        geometry=geometry,
                        latent_dim=3,
                        hidden_dim=10,
                        n_modes=modes,
                        sequence_length=48,
                        epochs=1,
                        random_state=7,
                    ).fit(x, entities, times)
                    components = model.decision_components(x, entities, times)
                    self.assertTrue(np.isfinite(components["predictive_nll"]).all())
                    self.assertTrue(np.isfinite(components["calibrated_score"]).all())
                    np.testing.assert_array_equal(
                        components["calibrated_score"][[0, 64, 128]],
                        0.0,
                    )

    def test_scores_are_causal_with_respect_to_future_observations(self) -> None:
        x, entities, times = self._sequences()
        model = PredictiveGenerativeDetector(
            geometry="euclidean",
            latent_dim=4,
            hidden_dim=12,
            n_modes=2,
            sequence_length=32,
            epochs=2,
            random_state=11,
        ).fit(x, entities, times)
        baseline = model.decision_function(x, entities, times)
        changed = x.copy()
        changed[40:64] += 20.0
        rescored = model.decision_function(changed, entities, times)
        np.testing.assert_allclose(baseline[:40], rescored[:40], atol=1e-6)
        self.assertFalse(np.allclose(baseline[40:64], rescored[40:64]))

    def test_identifiability_training_uses_distinct_mode_decoders(self) -> None:
        x, entities, times = self._sequences()
        model = PredictiveGenerativeDetector(
            geometry="euclidean",
            latent_dim=4,
            hidden_dim=12,
            n_modes=3,
            sequence_length=32,
            epochs=1,
            mode_specific_decoders=True,
            contrastive_weight=0.5,
            diversity_weight=0.05,
            balance_weight=0.01,
            persistence_weight=0.01,
            random_state=13,
        ).fit(x, entities, times)
        scores = model.decision_function(x, entities, times)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(len(model.model_.decoders), 3)

        scaled = model.feature_scaler_.transform(x[:32]).astype(np.float32)
        sequence = model.torch_.as_tensor(scaled, dtype=model.torch_.float32)
        dt = model.torch_.as_tensor(
            model._normalized_dt(times[:32]),
            dtype=model.torch_.float32,
        )
        with model.torch_.no_grad():
            means = model.model_.predictive_factors(sequence, dt)["predicted_means"]
        self.assertEqual(tuple(means.shape), (31, 3, x.shape[1]))
        self.assertGreater(float(model.torch_.std(means[:, :, 0], dim=1).mean()), 0.0)

    def test_non_increasing_time_is_rejected(self) -> None:
        x, entities, times = self._sequences()
        times[8] = times[7]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PredictiveGenerativeDetector(epochs=1).fit(x, entities, times)


if __name__ == "__main__":
    unittest.main()
