import numpy as np
from scipy.special import logsumexp

from research.window_dynsphere.density import LocalDynamicsDensity


def test_one_regime_density_matches_hand_calculation():
    q = np.zeros((100, 1))
    model = LocalDynamicsDensity().fit(q, q, np.zeros((100, 2)))
    targets = np.tile([[-1., -2.], [1., 2.]], (50, 1))
    model.calibrate(q, q, np.zeros_like(targets), targets)
    result = model.predict(q[:1], q[:1], np.zeros((1, 2)), np.array([[0., 0.]]))
    np.testing.assert_allclose(result['scores'], [np.log(2 * np.pi) + np.log(2.)])
    np.testing.assert_array_equal(result['weights'], [[1.]])


def test_ridge_uses_mean_loss_and_unpenalized_intercept():
    q = np.tile([[-1.], [1.]], (100, 1))
    model = LocalDynamicsDensity(ridge_alpha=1.).fit(q, q, 10. + 2 * q)
    np.testing.assert_allclose(np.atleast_2d(model.models_[0].coef_), [[1.]])
    np.testing.assert_allclose(np.atleast_1d(model.models_[0].intercept_), [10.])


def test_soft_mixture_is_normalized_and_uses_log_sum_exp():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(400, 2))
    z = np.concatenate([rng.normal(-4, .2, (200, 2)), rng.normal(4, .2, (200, 2))])
    y = q[:, :1] + np.where(z[:, :1] > 0, 3., -3.)
    model = LocalDynamicsDensity(n_regimes=4).fit(q, z, y)
    model.calibrate(q, z, np.zeros_like(y), y + rng.normal(0, .1, y.shape))
    r = model.predict(q, z, np.zeros_like(y), y)
    np.testing.assert_allclose(r['weights'].sum(1), 1.)
    np.testing.assert_allclose(r['scores'], -logsumexp(r['log_weights'] + r['log_density'], axis=1))
    assert np.isfinite(r['scores']).all()
    assert np.isfinite(model.variances_).all() and (model.variances_ >= 1e-6).all()


def test_small_clusters_merge_deterministically():
    q = np.arange(100.)[:, None]
    z = np.repeat(np.arange(4.), 25)[:, None]
    a = LocalDynamicsDensity(n_regimes=4).fit(q, z, q)
    b = LocalDynamicsDensity(n_regimes=4).fit(q, z, q)
    assert 1 <= a.effective_regimes_ < 4
    np.testing.assert_array_equal(a.labels_, b.labels_)
    assert min(np.bincount(a.labels_)) >= 32 or a.effective_regimes_ == 1


def test_constant_descriptors_channels_and_extreme_distance_remain_finite():
    q = np.ones((100, 2))
    y = np.ones((100, 2)) * 7
    model = LocalDynamicsDensity(n_regimes=4).fit(q, q, y)
    assert model.effective_regimes_ == 1
    model.calibrate(q, q, np.zeros_like(y), y)
    result = model.predict(q, q * 1e8, np.zeros_like(y), y)
    assert np.isfinite(result['scores']).all()
    np.testing.assert_allclose(model.variances_, 1e-6)


def test_empty_calibration_cluster_uses_pooled_statistics():
    q = np.zeros((200, 1))
    z = np.repeat([[-1.], [1.]], 100, axis=0)
    model = LocalDynamicsDensity(n_regimes=2).fit(q, z, q)
    model.temperature_ = 1e-6
    y = np.tile([[-1.], [1.]], (10, 1))
    model.calibrate(q[:20], z[:20], q[:20], y)
    assert np.isfinite(model.variances_).all()
    assert np.isfinite(model.biases_).all()
    np.testing.assert_allclose(model.variances_, 1.)


def test_extreme_multi_regime_distance_still_normalizes_weights():
    q = np.zeros((200, 1))
    descriptor = np.tile([[-1., 0.], [1., 0.]], (100, 1))
    model = LocalDynamicsDensity(n_regimes=2).fit(q, descriptor, q)
    model.calibrate(q, descriptor, q, np.tile([[-1.], [1.]], (100, 1)))
    model.temperature_ = 1e-6
    result = model.predict(np.zeros((1,1)), np.array([[0.,1e8]]), np.zeros((1,1)), np.zeros((1,1)))
    np.testing.assert_allclose(result['weights'], [[.5,.5]])
