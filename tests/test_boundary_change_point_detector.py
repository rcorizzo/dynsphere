from __future__ import annotations

import unittest

import numpy as np

from boundary_change_point_detector import BoundaryChangePointDetector


class BoundaryChangePointDetectorTests(unittest.TestCase):
    @staticmethod
    def _sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = []
        entities = []
        times = []
        for entity in range(3):
            t = np.arange(64, dtype=float)
            phase = 0.08 * t + 0.4 * entity
            observations.append(
                np.column_stack(
                    [
                        np.sin(phase),
                        np.cos(phase),
                        np.sin(2.0 * phase),
                        np.cos(0.5 * phase),
                        t / 64.0,
                        np.sin(phase + 0.3),
                    ]
                )
            )
            entities.append(np.full(len(t), entity))
            times.append(t)
        return np.vstack(observations), np.concatenate(entities), np.concatenate(times)

    def test_boundary_model_fits_and_exposes_hypotheses(self) -> None:
        x, entities, times = self._sequences()
        model = BoundaryChangePointDetector(
            horizons=(1, 2, 4),
            latent_dim=4,
            hidden_dim=12,
            sequence_length=32,
            epochs=1,
            rollout_epochs=1,
            reset_components=2,
            random_state=5,
        ).fit(x, entities, times)
        components = model.decision_components(x, entities, times)

        self.assertEqual(components["boundary_evidence"].shape, (len(x), 3))
        self.assertEqual(components["continuation_nll"].shape, (len(x), 3))
        self.assertEqual(components["reset_segment_nll"].shape, (len(x), 3))
        self.assertTrue(np.isfinite(components["calibrated_score"]).all())
        self.assertTrue(
            set(np.unique(components["best_horizon"])).issubset({0, 1, 2, 4})
        )

    def test_boundary_score_is_causal(self) -> None:
        x, entities, times = self._sequences()
        model = BoundaryChangePointDetector(
            horizons=(1, 2, 4),
            latent_dim=4,
            hidden_dim=12,
            sequence_length=32,
            epochs=1,
            rollout_epochs=1,
            reset_components=2,
            random_state=7,
        ).fit(x, entities, times)
        baseline = model.decision_function(x, entities, times)
        changed = x.copy()
        changed[40:64] += 10.0
        rescored = model.decision_function(changed, entities, times)

        np.testing.assert_allclose(baseline[:40], rescored[:40], atol=1e-6)
        self.assertFalse(np.allclose(baseline[40:64], rescored[40:64]))

    def test_invalid_horizon_is_rejected(self) -> None:
        x, entities, times = self._sequences()
        with self.assertRaisesRegex(ValueError, "horizons"):
            BoundaryChangePointDetector(horizons=(0, 2), epochs=1).fit(
                x, entities, times
            )


if __name__ == "__main__":
    unittest.main()
