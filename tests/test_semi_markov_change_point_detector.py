from __future__ import annotations

import unittest

import numpy as np

from semi_markov_change_point_detector import SemiMarkovChangePointDetector


class SemiMarkovChangePointDetectorTests(unittest.TestCase):
    @staticmethod
    def _sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = []
        entities = []
        times = []
        for entity in range(3):
            t = np.arange(64, dtype=float)
            phase = 0.08 * t + 0.3 * entity
            observations.append(
                np.column_stack(
                    [
                        np.sin(phase),
                        np.cos(phase),
                        np.sin(2.0 * phase),
                        np.cos(0.5 * phase),
                        t / 64.0,
                        np.sin(phase + 0.2),
                    ]
                )
            )
            entities.append(np.full(len(t), entity))
            times.append(t)
        return np.vstack(observations), np.concatenate(entities), np.concatenate(times)

    def test_semi_markov_posterior_is_normalized(self) -> None:
        x, entities, times = self._sequences()
        model = SemiMarkovChangePointDetector(
            max_duration=4,
            mean_duration=3.0,
            latent_dim=4,
            hidden_dim=12,
            sequence_length=32,
            epochs=1,
            rollout_epochs=1,
            reset_components=2,
            random_state=23,
        ).fit(x, entities, times)
        components = model.decision_components(x, entities, times)

        posterior = components["age_posterior"]
        available = np.isfinite(components["boundary_evidence"]).any(axis=1)
        np.testing.assert_allclose(posterior[available].sum(axis=1), 1.0)
        self.assertTrue(
            np.logical_and(
                components["active_event_probability"] >= 0.0,
                components["active_event_probability"] <= 1.0,
            ).all()
        )
        self.assertTrue(np.isfinite(components["calibrated_score"]).all())

    def test_semi_markov_score_is_causal(self) -> None:
        x, entities, times = self._sequences()
        model = SemiMarkovChangePointDetector(
            max_duration=4,
            mean_duration=3.0,
            latent_dim=4,
            hidden_dim=12,
            sequence_length=32,
            epochs=1,
            rollout_epochs=1,
            reset_components=2,
            random_state=29,
        ).fit(x, entities, times)
        baseline = model.decision_function(x, entities, times)
        changed = x.copy()
        changed[40:64] += 10.0
        rescored = model.decision_function(changed, entities, times)

        np.testing.assert_allclose(baseline[:40], rescored[:40], atol=1e-6)
        self.assertFalse(np.allclose(baseline[40:64], rescored[40:64]))

    def test_invalid_duration_prior_is_rejected(self) -> None:
        x, entities, times = self._sequences()
        with self.assertRaisesRegex(ValueError, "mean_duration"):
            SemiMarkovChangePointDetector(
                max_duration=4,
                mean_duration=1.0,
                epochs=1,
            ).fit(x, entities, times)


if __name__ == "__main__":
    unittest.main()
