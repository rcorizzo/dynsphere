from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from benchmarks.tsb_ad_m.run_tsb_ad_m import (
    declared_prefix_indices,
    score_boundary_change_point,
    score_semi_markov_change_point,
    score_manifold_dynamics,
    score_predictive_generative,
    score_segment_predictive,
    score_unified_generative,
)


class RealWorldStrategyIntegrationTests(unittest.TestCase):
    def test_declared_tsb_prefix_follows_the_official_split(self) -> None:
        path = Path("002_MSL_id_1_Sensor_tr_30_1st_90.csv")
        labels = np.zeros(100, dtype=int)
        np.testing.assert_array_equal(
            declared_prefix_indices(path, labels, min_train=16),
            np.arange(30),
        )
        np.testing.assert_array_equal(
            declared_prefix_indices(path, labels[::4], min_train=5, source_stride=4),
            np.arange(8),
        )
        labels[12] = 1
        np.testing.assert_array_equal(
            declared_prefix_indices(path, labels, min_train=16),
            np.arange(30),
        )

    def test_discontinuous_normal_training_segments_are_supported(self) -> None:
        time = np.arange(100, dtype=float)
        x_all = np.column_stack(
            [
                np.sin(0.04 * time),
                np.cos(0.04 * time),
                np.sin(0.09 * time),
                np.cos(0.02 * time),
                time / 100.0,
                np.sin(0.04 * time) * np.cos(0.02 * time),
            ]
        )
        train_idx = np.concatenate([np.arange(30), np.arange(50, 85)])
        x_train = x_all[train_idx]

        scores = score_manifold_dynamics(
            x_train,
            x_all,
            train_idx,
            random_state=17,
            geometry="sphere",
            field_model="ridge",
            include_radius=True,
        )

        self.assertEqual(scores.shape, (len(x_all),))
        self.assertTrue(np.isfinite(scores).all())

    def test_short_normal_islands_are_removed_with_their_timestamps(self) -> None:
        time = np.arange(100, dtype=float)
        x_all = np.column_stack(
            [
                np.sin(0.04 * time),
                np.cos(0.04 * time),
                np.sin(0.09 * time),
                np.cos(0.02 * time),
                time / 100.0,
                np.sin(0.04 * time) * np.cos(0.02 * time),
            ]
        )
        train_idx = np.concatenate([np.arange(30), np.arange(45, 48), np.arange(60, 90)])

        scores = score_manifold_dynamics(
            x_all[train_idx],
            x_all,
            train_idx,
            random_state=17,
            geometry="euclidean",
            field_model="ridge",
            include_radius=False,
        )

        self.assertEqual(scores.shape, (len(x_all),))
        self.assertTrue(np.isfinite(scores).all())

    def test_unified_model_supports_interrupted_normal_training(self) -> None:
        time = np.arange(100, dtype=float)
        x_all = np.column_stack(
            [
                np.sin(0.04 * time),
                np.cos(0.04 * time),
                np.sin(0.09 * time),
                np.cos(0.02 * time),
                time / 100.0,
                np.sin(0.04 * time) * np.cos(0.02 * time),
            ]
        )
        train_idx = np.concatenate([np.arange(35), np.arange(55, 90)])
        scores = score_unified_generative(
            x_all[train_idx],
            x_all,
            train_idx,
            random_state=17,
            geometry="euclidean",
            epochs=1,
        )
        self.assertEqual(scores.shape, (len(x_all),))
        self.assertTrue(np.isfinite(scores).all())

    def test_predictive_model_supports_interrupted_normal_training(self) -> None:
        time = np.arange(100, dtype=float)
        x_all = np.column_stack(
            [
                np.sin(0.04 * time),
                np.cos(0.04 * time),
                np.sin(0.09 * time),
                np.cos(0.02 * time),
                time / 100.0,
                np.sin(0.04 * time) * np.cos(0.02 * time),
            ]
        )
        train_idx = np.concatenate([np.arange(35), np.arange(55, 90)])
        scores = score_predictive_generative(
            x_all[train_idx],
            x_all,
            train_idx,
            random_state=17,
            geometry="euclidean",
            epochs=1,
        )
        self.assertEqual(scores.shape, (len(x_all),))
        self.assertTrue(np.isfinite(scores).all())

    def test_segment_model_supports_interrupted_normal_training(self) -> None:
        time = np.arange(100, dtype=float)
        x_all = np.column_stack(
            [
                np.sin(0.04 * time),
                np.cos(0.04 * time),
                np.sin(0.09 * time),
                np.cos(0.02 * time),
                time / 100.0,
                np.sin(0.04 * time) * np.cos(0.02 * time),
            ]
        )
        train_idx = np.concatenate([np.arange(35), np.arange(55, 90)])
        scores = score_segment_predictive(
            x_all[train_idx],
            x_all,
            train_idx,
            random_state=17,
            epochs=1,
        )
        self.assertEqual(scores.shape, (len(x_all),))
        self.assertTrue(np.isfinite(scores).all())

    def test_boundary_model_supports_interrupted_normal_training(self) -> None:
        time = np.arange(100, dtype=float)
        x_all = np.column_stack(
            [
                np.sin(0.04 * time),
                np.cos(0.04 * time),
                np.sin(0.09 * time),
                np.cos(0.02 * time),
                time / 100.0,
                np.sin(0.04 * time) * np.cos(0.02 * time),
            ]
        )
        train_idx = np.concatenate([np.arange(35), np.arange(55, 90)])
        scores = score_boundary_change_point(
            x_all[train_idx],
            x_all,
            train_idx,
            random_state=17,
            epochs=1,
            rollout_epochs=1,
            horizons=(1, 2, 4, 8),
        )
        self.assertEqual(scores.shape, (len(x_all),))
        self.assertTrue(np.isfinite(scores).all())

    def test_semi_markov_model_supports_interrupted_normal_training(self) -> None:
        time = np.arange(100, dtype=float)
        x_all = np.column_stack(
            [
                np.sin(0.04 * time),
                np.cos(0.04 * time),
                np.sin(0.09 * time),
                np.cos(0.02 * time),
                time / 100.0,
                np.sin(0.04 * time) * np.cos(0.02 * time),
            ]
        )
        train_idx = np.concatenate([np.arange(35), np.arange(55, 90)])
        scores = score_semi_markov_change_point(
            x_all[train_idx],
            x_all,
            train_idx,
            random_state=17,
            epochs=1,
            rollout_epochs=1,
            max_duration=8,
            mean_duration=4.0,
        )
        self.assertEqual(scores.shape, (len(x_all),))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
