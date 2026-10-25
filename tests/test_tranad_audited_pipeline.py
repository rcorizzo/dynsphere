"""Audit the scaling and score-time correspondence independently of fitting."""
import numpy as np
import pytest

from benchmarks.official_baselines.adapters import tranad


def test_train_only_scaling_does_not_clip_or_fit_to_test_anomalies():
    train = np.array([[-2., 5.], [2., 5.]])
    test = np.array([[6., 7.]])
    scaled_train, scaled_test = tranad._train_minmax(train, test)
    np.testing.assert_allclose(scaled_train[:, 0], [0., 4/4.0001])
    np.testing.assert_allclose(scaled_test[0], [8/4.0001, 20000.])


def test_audited_windows_target_the_observation_at_the_score_index():
    torch = pytest.importorskip('torch')
    values = torch.tensor([[10.], [20.], [30.], [40.]])
    windows = tranad._target_windows(values, 3)
    np.testing.assert_array_equal(windows[:, -1, 0], [10., 20., 30., 40.])
    np.testing.assert_array_equal(windows[0, :, 0], [10., 10., 10.])
    np.testing.assert_array_equal(windows[2, :, 0], [10., 20., 30.])


def test_legacy_window_convention_remains_available():
    torch = pytest.importorskip('torch')
    values = torch.tensor([[10.], [20.], [30.], [40.]])
    np.testing.assert_array_equal(tranad._prefix_windows(values, 3)[:, -1, 0],
                                  [10., 10., 20., 30.])
