import inspect
import pickle

import numpy as np
import pytest

from research.window_dynsphere.detector import WindowConditionedDynamicsDetector


def observations(n=400):
    t = np.arange(n)
    return np.column_stack([np.sin(t / 7), np.cos(t / 11), np.ones(n)])


def fit_model(**kwargs):
    options = dict(window=8, hidden_dim=6, latent_dim=3, epochs=1, max_targets=64,
                   batch_size=16, random_state=37)
    options.update(kwargs)
    x = observations()
    return WindowConditionedDynamicsDetector(**options).fit(x, np.zeros(len(x)))


def test_scores_and_predictions_are_causal():
    model = fit_model(n_regimes=4)
    x = observations(100)
    ids = np.zeros(100)
    before = model.decision_components(x, ids, diagnostics=True)
    changed = x.copy()
    changed[50:] += 20
    after = model.decision_components(changed, ids, diagnostics=True)
    np.testing.assert_allclose(before['scores'][:50], after['scores'][:50], equal_nan=True)
    np.testing.assert_allclose(before['predictive_means'][50], after['predictive_means'][50])
    np.testing.assert_allclose(before['regime_weights'][50], after['regime_weights'][50])
    assert after['scores'][50] != before['scores'][50]
    assert np.isnan(before['scores'][:8]).all()
    assert np.isfinite(before['scores'][8:]).all()
    assert before['validity_mask'].sum() == 92


def test_entities_and_batches_do_not_change_scores():
    model = fit_model()
    x = observations(100)
    joined = np.vstack([x, x + 1])
    ids = np.repeat([0, 1], 100)
    baseline = model.decision_function(joined, ids)
    changed = joined.copy()
    changed[:100] += 99
    np.testing.assert_allclose(baseline[100:], model.decision_function(changed, ids)[100:], equal_nan=True)
    model.batch_size = 7
    np.testing.assert_allclose(baseline, model.decision_function(joined, ids), atol=1e-4, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(baseline[:100], model.decision_function(x, np.zeros(100)),
                               atol=1e-4, rtol=1e-6, equal_nan=True)


def test_calibration_suffix_cannot_change_representation_or_scaler():
    import torch
    x = observations()
    a = fit_model()
    x[320:] += 50
    b = WindowConditionedDynamicsDetector(window=8, hidden_dim=6, latent_dim=3, epochs=1,
                                          max_targets=64, batch_size=16, random_state=37).fit(x, np.zeros(400))
    np.testing.assert_allclose(a.scaler_.mean_, b.scaler_.mean_)
    for av, bv in zip(a.model_.parameters(), b.model_.parameters()):
        assert torch.equal(av, bv)


def test_serialization_replay_and_sparse_contexts():
    model = fit_model(reconstruction_weight=0)
    x = observations(100)
    ids = np.zeros(100)
    scores = model.decision_function(x, ids)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(scores, restored.decision_function(x, ids), equal_nan=True)
    np.testing.assert_allclose(scores, fit_model(reconstruction_weight=0).decision_function(x, ids), equal_nan=True)
    xs = model.scaler_.transform(x)
    groups = [np.arange(100)]
    sparse = model._contexts(xs, groups, np.array([10, 60]))
    dense = model._contexts(xs, groups, np.arange(8, 100))
    np.testing.assert_allclose(sparse[0], dense[0][[2, 52]], atol=1e-6)
    head = model.forecast_head_scores(x, ids)
    assert np.isfinite(head[8:]).all() and np.isnan(head[:8]).all()
    assert 'labels' not in inspect.signature(model.fit).parameters


def test_channel_independent_path_has_no_cross_channel_dependencies():
    model = fit_model(channel_independent=True)
    x = observations(100)
    a = model.decision_components(x, np.zeros(100), diagnostics=True)
    changed = x.copy()
    changed[:, 1] += 50
    b = model.decision_components(changed, np.zeros(100), diagnostics=True)
    np.testing.assert_allclose(a['predictive_means'][8:, :, 0], b['predictive_means'][8:, :, 0], atol=1e-6)
    assert np.isfinite(a['scores'][8:]).all()


def test_regime_control_reuses_only_identical_fitting_representation():
    a = fit_model()
    options = a.get_params()
    options['n_regimes'] = 4
    x = observations()
    b = WindowConditionedDynamicsDetector(**options).fit(x, np.zeros(400), representation_from=a)
    assert b.training_loss_ == a.training_loss_
    changed = x.copy()
    changed[:100] += 1
    with pytest.raises(ValueError, match='representation'):
        WindowConditionedDynamicsDetector(**options).fit(changed, np.zeros(400), representation_from=a)


def test_long_sequence_preprocessing_is_reused_across_scoring_batches(monkeypatch):
    model = fit_model(channel_independent=True)
    original = model._flatten
    calls = []
    def counted(xs, groups):
        calls.append(len(xs))
        return original(xs,groups)
    monkeypatch.setattr(model,'_flatten',counted)
    x = observations(200)
    model.decision_function(x,np.zeros(200))
    assert calls == [200]


@pytest.mark.parametrize('kind', ['short', 'nan', 'irregular', 'repeated'])
def test_invalid_inputs_are_rejected(kind):
    x = observations()
    times = np.arange(len(x), dtype=float)
    if kind == 'short':
        x, times = x[:50], times[:50]
    elif kind == 'nan':
        x[2, 0] = np.nan
    elif kind == 'irregular':
        times[5:] += .5
    else:
        times[5] = times[4]
    with pytest.raises(ValueError):
        WindowConditionedDynamicsDetector(window=8, epochs=1).fit(x, np.zeros(len(x)), times)
