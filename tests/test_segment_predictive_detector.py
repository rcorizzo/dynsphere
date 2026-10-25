from __future__ import annotations

import unittest

import numpy as np

from segment_predictive_detector import SegmentPredictiveDetector


class SegmentPredictiveDetectorTests(unittest.TestCase):
    @staticmethod
    def _sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = []
        entities = []
        times = []
        for entity in range(3):
            t = np.cumsum(0.9 + 0.2 * np.sin(np.arange(64) * 0.13 + entity) ** 2)
            phase = 0.09 * t + 0.35 * entity
            observations.append(
                np.column_stack(
                    [
                        np.sin(phase),
                        np.cos(phase),
                        np.sin(2.0 * phase),
                        np.cos(0.5 * phase),
                        0.02 * t,
                        np.sin(phase + 0.2) * np.cos(phase),
                    ]
                )
            )
            entities.append(np.full(len(t), entity))
            times.append(t)
        return np.vstack(observations), np.concatenate(entities), np.concatenate(times)

    def test_multi_horizon_scan_fits_and_scores(self) -> None:
        x, entities, times = self._sequences()
        model = SegmentPredictiveDetector(
            horizons=(2, 4, 8),
            latent_dim=4,
            hidden_dim=12,
            sequence_length=32,
            epochs=1,
            random_state=7,
        ).fit(x, entities, times)
        components = model.decision_components(x, entities, times)

        self.assertEqual(components["horizon_evidence"].shape, (len(x), 3))
        self.assertEqual(components["free_running_nll"].shape, (len(x), 3))
        self.assertEqual(components["reanchored_nll"].shape, (len(x), 3))
        self.assertTrue(np.isfinite(components["calibrated_score"]).all())
        self.assertTrue(
            set(np.unique(components["best_horizon"])).issubset({0, 2, 4, 8})
        )
        np.testing.assert_array_equal(
            components["calibrated_score"][[0, 64, 128]],
            0.0,
        )

    def test_scan_is_causal_with_respect_to_future_observations(self) -> None:
        x, entities, times = self._sequences()
        model = SegmentPredictiveDetector(
            horizons=(2, 4, 8),
            latent_dim=4,
            hidden_dim=12,
            sequence_length=32,
            epochs=1,
            random_state=11,
        ).fit(x, entities, times)
        baseline = model.decision_function(x, entities, times)
        changed = x.copy()
        changed[40:64] += 20.0
        rescored = model.decision_function(changed, entities, times)
        np.testing.assert_allclose(baseline[:40], rescored[:40], atol=1e-6)
        self.assertFalse(np.allclose(baseline[40:64], rescored[40:64]))

    def test_explicit_multistep_training_runs(self) -> None:
        x, entities, times = self._sequences()
        model = SegmentPredictiveDetector(
            horizons=(2, 4),
            latent_dim=4,
            hidden_dim=12,
            sequence_length=32,
            epochs=1,
            rollout_epochs=1,
            random_state=13,
        ).fit(x, entities, times)

        self.assertEqual(len(model.rollout_training_loss_), 1)
        self.assertTrue(np.isfinite(model.rollout_training_loss_[0]))
        self.assertTrue(
            np.isfinite(model.decision_function(x, entities, times)).all()
        )

    def test_probabilistic_rollout_training_runs(self) -> None:
        x, entities, times = self._sequences()
        model = SegmentPredictiveDetector(
            horizons=(2, 4),
            latent_dim=4,
            hidden_dim=12,
            sequence_length=32,
            epochs=1,
            rollout_epochs=1,
            probabilistic_rollout=True,
            random_state=19,
        ).fit(x, entities, times)

        self.assertEqual(model.sigma_points_.shape, (9, 4))
        self.assertTrue(np.isfinite(model.rollout_training_loss_[0]))
        components = model.decision_components(x, entities, times)
        self.assertTrue(np.isfinite(components["calibrated_score"]).all())

    def test_invalid_horizons_are_rejected(self) -> None:
        x, entities, times = self._sequences()
        with self.assertRaisesRegex(ValueError, "horizons"):
            SegmentPredictiveDetector(horizons=(1, 2), epochs=1).fit(
                x,
                entities,
                times,
            )


if __name__ == "__main__":
    unittest.main()
