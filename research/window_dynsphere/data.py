"""Entity-local causal windows with training-only preprocessing."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def validate_sequences(x, entity_ids, times=None):
    x = np.asarray(x, dtype=float)
    ids = np.asarray(entity_ids)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite observation matrix')
    if ids.shape != (len(x),) or any(value != value for value in ids):
        raise ValueError('Invalid entity identifiers')
    groups = [np.flatnonzero(ids == entity) for entity in np.unique(ids)]
    if times is None:
        times = np.empty(len(x), dtype=float)
        for idx in groups:
            times[idx] = np.arange(len(idx))
    times = np.asarray(times, dtype=float)
    if times.shape != (len(x),) or not np.isfinite(times).all():
        raise ValueError('Invalid timestamps')
    for idx in groups:
        dt = np.diff(times[idx])
        if len(dt) and (np.any(dt <= 0) or not np.allclose(dt, dt[0], atol=1e-8, rtol=1e-6)):
            raise ValueError('Only strictly increasing regularly sampled entities are supported')
    return x, groups, times


def prepare_training(x, entity_ids, times, window, max_targets, seed):
    x, groups, times = validate_sequences(x, entity_ids, times)
    if window < 2 or max_targets < 1:
        raise ValueError('Invalid window or fitting-target budget')
    fit_rows, fit_targets, calibration_targets = [], [], []
    for idx in groups:
        stop = int(np.floor(.8 * len(idx)))
        if stop - window < window + 32 or len(idx) - stop < 32:
            raise ValueError('Insufficient fitting/calibration history for the fixed window')
        fit_rows.extend(idx[:stop])
        fit_targets.append(idx[window:stop])
        calibration_targets.extend(idx[stop:])
    budget = min(int(max_targets), sum(map(len, fit_targets)))
    counts = np.zeros(len(groups), dtype=int)
    for _ in range(budget):
        eligible = [i for i, target in enumerate(fit_targets) if counts[i] < len(target)]
        chosen = min(eligible, key=lambda i: (counts[i], i))
        counts[chosen] += 1
    selected = []
    for targets, count in zip(fit_targets, counts):
        if count:
            selected.extend(targets[np.linspace(0, len(targets) - 1, count, dtype=int)])
    scaler = StandardScaler().fit(x[np.asarray(fit_rows)])
    return {'scaled': scaler.transform(x), 'scaler': scaler, 'groups': groups, 'times': times,
            'fit_targets': np.concatenate(fit_targets),
            'calibration_targets': np.asarray(calibration_targets, dtype=int),
            'selected_targets': np.sort(np.asarray(selected, dtype=int)),
            'fit_rows': np.asarray(fit_rows, dtype=int), 'seed': int(seed)}


def history_batches(x_scaled, entity_indices, target_indices, window, batch_size, lookup=None):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if lookup is None:
        group_for = np.full(len(x_scaled), -1, dtype=int)
        position = np.zeros(len(x_scaled), dtype=int)
        for group, idx in enumerate(entity_indices):
            group_for[idx] = group
            position[idx] = np.arange(len(idx))
    else:
        group_for, position = lookup
    targets = np.asarray(target_indices, dtype=int)
    if (targets < 0).any() or (targets >= len(x_scaled)).any() or (position[targets] < window).any():
        raise ValueError('A target lacks entity-local history')
    for start in range(0, len(targets), batch_size):
        rows = targets[start:start + batch_size]
        histories = [x_scaled[entity_indices[group_for[t]][position[t] - window:position[t]]]
                     for t in rows]
        yield rows, np.stack(histories)
