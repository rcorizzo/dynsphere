import numpy as np
import pytest

from research.window_dynsphere.data import history_batches, prepare_training, validate_sequences
from research.window_dynsphere.network import HistoryEncoder, train_encoder


def series(n=400, d=3):
    t = np.arange(n)
    return np.column_stack([np.sin(t / (7 + k)) for k in range(d)])


def test_history_stops_before_target_and_respects_interleaved_entities():
    x = np.arange(40.).reshape(20, 2)
    ids = np.arange(20) % 2
    a, groups, times = validate_sequences(x, ids)
    target, windows = next(history_batches(a, groups, np.array([8, 9]), 3, 2))
    np.testing.assert_array_equal(target, [8, 9])
    np.testing.assert_array_equal(windows[0], x[[2, 4, 6]])
    np.testing.assert_array_equal(windows[1], x[[3, 5, 7]])


def test_split_scaling_and_targets_are_invariant_to_calibration_suffix():
    x = series()
    p = prepare_training(x, np.zeros(len(x)), None, 8, 20, 37)
    changed = x.copy()
    changed[320:] += 1000
    q = prepare_training(changed, np.zeros(len(x)), None, 8, 20, 37)
    np.testing.assert_allclose(p['scaler'].mean_, x[:320].mean(0))
    np.testing.assert_allclose(p['scaler'].mean_, q['scaler'].mean_)
    np.testing.assert_array_equal(p['selected_targets'], q['selected_targets'])
    assert len(p['selected_targets']) == 20
    assert p['fit_targets'].max() == 319
    assert p['calibration_targets'].min() == 320


def test_selected_targets_are_balanced_and_deterministic():
    x = np.vstack([series(), series()])
    ids = np.repeat([1, 2], 400)
    p = prepare_training(x, ids, None, 8, 20, 37)
    q = prepare_training(x, ids, None, 8, 20, 37)
    assert np.sum(ids[p['selected_targets']] == 1) == 10
    np.testing.assert_array_equal(p['selected_targets'], q['selected_targets'])


@pytest.mark.parametrize('change', ['nan', 'time', 'irregular', 'short'])
def test_invalid_sequences_are_explicit(change):
    x = series()
    t = np.arange(len(x), dtype=float)
    if change == 'nan':
        x[3, 0] = np.nan
    elif change == 'time':
        t[3] = t[2]
    elif change == 'irregular':
        t[3:] += 0.5
    else:
        x, t = x[:20], t[:20]
    with pytest.raises(ValueError):
        prepare_training(x, np.zeros(len(x)), t, 8, 20, 37)


def test_representation_shapes_objective_and_replay():
    import torch
    x = series(400, 2)
    p = prepare_training(x, np.zeros(len(x)), None, 8, 32, 37)
    options = dict(window=8, hidden_dim=6, latent_dim=3, reconstruction_weight=1., epochs=1,
                   batch_size=16, random_state=37)
    a, loss = train_encoder(p['scaled'], p['groups'], p['selected_targets'], **options)
    b, other_loss = train_encoder(p['scaled'], p['groups'], p['selected_targets'], **options)
    assert np.isfinite(loss).all()
    assert loss == other_loss
    for av, bv in zip(a.parameters(), b.parameters()):
        assert torch.equal(av, bv)
    _, w = next(history_batches(p['scaled'], p['groups'], p['selected_targets'], 8, 16))
    with torch.no_grad():
        z = a.encode(torch.tensor(w, dtype=torch.float32))
        assert tuple(z.shape) == (16, 3)
        assert tuple(a.forecast(z).shape) == (16, 2)
        assert tuple(a.reconstruct(z).shape) == (16, 8, 2)
    no_reconstruction = HistoryEncoder(2, window=8, hidden_dim=6, latent_dim=3,
                                      reconstruction_weight=0)
    assert no_reconstruction.reconstruction is None
