"""Internal held-out protocol evaluation; historical experiment files stay unchanged."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
METHODS = ('dynsphere', 'dynsphere_euclidean', 'dynsphere_one_regime',
           'dynsphere_euclidean_one_regime', 'pca_reconstruction',
           'kmeans_distance', 'linear_ridge_ar1')
SEEDS = (37, 38, 39)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gap_entities(positions, groups):
    positions, groups = np.asarray(positions), np.asarray(groups)
    if len(positions) == 0:
        return np.array([], dtype=int)
    starts = np.r_[True, (np.diff(positions) != 1) | (groups[1:] != groups[:-1])]
    return np.cumsum(starts).astype(int) - 1


def boundary_mask(entities, left=100, right=10):
    entities = np.asarray(entities)
    result = np.zeros(len(entities), dtype=bool)
    for entity in np.unique(entities):
        indices = np.flatnonzero(entities == entity)
        if len(indices) > left + right:
            result[indices[left:len(indices)-right if right else None]] = True
    return result


def preprocess(train, calibration, test):
    arrays = [np.array(a, dtype=float, copy=True) for a in (train, calibration, test)]
    if any(a.ndim != 2 or a.shape[1] != arrays[0].shape[1] for a in arrays):
        raise ValueError('Feature dimensions must agree')
    for a in arrays:
        a[~np.isfinite(a)] = np.nan
    keep = np.flatnonzero(np.isfinite(arrays[0]).any(axis=0))
    if len(keep) == 0:
        raise ValueError('No finite fitting features')
    median = np.nanmedian(arrays[0][:, keep], axis=0)
    arrays = [np.where(np.isfinite(a[:, keep]), a[:, keep], median) for a in arrays]
    scaler = StandardScaler().fit(arrays[0])
    return (*[scaler.transform(a) for a in arrays], {
        'retained_features': keep.tolist(), 'imputation_medians': median.tolist(),
        'scaling_mean': scaler.mean_.tolist(), 'scaling_scale': scaler.scale_.tolist()})


def remove_overlap(parts):
    """Retain later partitions; remove exact raw feature-vector matches earlier."""
    seen, result, removed = set(), [], []
    columns = list(parts[0].columns)
    for part in reversed(parts):
        if list(part.columns) != columns:
            raise ValueError('Feature columns must agree')
        # Canonical float representation preserves complete rows, including NaNs.
        values = part.to_numpy(dtype='<f8', copy=True)
        values[np.isnan(values)] = np.nan
        values[values == 0] = 0.  # Canonicalize negative zero.
        hashes = [hashlib.sha256(row.tobytes()).digest() for row in values]
        keep = np.array([h not in seen for h in hashes], dtype=bool)
        seen.update(hashes)
        result.append(part.iloc[np.flatnonzero(keep)].copy())
        removed.append(int((~keep).sum()))
    return list(reversed(result)), {'removed_rows': list(reversed(removed)),
                                    'rule': 'exact-raw-feature-rows; later-partition-priority'}


def _case(train, calibration, test, labels, metadata, entities=None):
    if len(train) < 16 or len(calibration) < 21:
        raise ValueError('Insufficient fitting or calibration observations')
    if entities is None:
        entities = [np.zeros(len(a), dtype=int) for a in (train, calibration, test)]
    train, calibration, test, prep = preprocess(train, calibration, test)
    cal_mask = boundary_mask(entities[1], 10, 10)
    test_mask = boundary_mask(entities[2], 100, 10)
    labels = np.asarray(labels, dtype=int)
    if not cal_mask.any() or set(np.unique(labels[test_mask])) != {0, 1}:
        raise ValueError('Insufficient common support with both classes')
    return {'train': train, 'calibration': calibration, 'test': test,
            'labels': labels, 'entities': entities, 'calibration_mask': cal_mask,
            'test_mask': test_mask, 'metadata': {**metadata, 'preprocessing': prep,
                'fit_rows': len(train), 'calibration_rows': len(calibration),
                'test_rows': len(test), 'calibration_evaluated_rows': int(cal_mask.sum()),
                'test_evaluated_rows': int(test_mask.sum()),
                'score_timing': 'retrospective; DynSphere uses up to 10 future samples'}}


def prepare_nasa(path):
    path = Path(path)
    frame = pd.read_csv(path)
    if 'Label' not in frame:
        raise ValueError('Missing Label column')
    y = frame.Label.to_numpy()
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid binary labels')
    match = re.search(r'_tr_(\d+)_', path.stem)
    if match is None:
        raise ValueError('Missing declared prefix')
    stop = int(match.group(1))
    if not 32 <= stop < len(frame):
        raise ValueError('Invalid declared prefix')
    if y[:stop].any():
        raise ValueError('Declared training prefix is contaminated')
    values = frame.drop(columns='Label').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('NASA inputs must be finite without row deletion')
    split = int(.8 * stop)
    family = path.stem.split('_id_')[0].split('_', 1)[1]
    return _case(values[:split], values[split:stop], values[stop:], y[stop:], {
        'case_id': path.stem, 'source_family': family,
        'protocol': 'clean-prefix-80-fit-20-calibration-disjoint-suffix-v1',
        'test_source_row_start': stop, 'source_sha256': digest(path),
        'feature_names': list(frame.drop(columns='Label').columns)})


def _raw_frame(path, label):
    from benchmarks.real_world_datasets.run_real_world_datasets import read_csv, normalize_labels
    frame = read_csv(Path(path))
    y = normalize_labels(frame[label], False)
    drop = [c for c in frame if c == label or c.lower() in ('time', 'timestamp', 'datetime', 'date')
            or c.lower().startswith('attack_')]
    # No imputation here: all fitting statistics belong to the fitting partition.
    return frame.drop(columns=drop).select_dtypes(include=[np.number]), y


def prepare_principal(family):
    from benchmarks.official_baselines.datasets import REAL_WORLD_SPECS
    spec = REAL_WORLD_SPECS[family]
    if Path(spec['train']).resolve() == Path(spec['test']).resolve():
        raise ValueError('Use the separate PowerSystem diagnostic')
    train, ytrain = _raw_frame(spec['train'], spec['train_field'])
    test, ytest = _raw_frame(spec['test'], spec['test_field'])
    if ytrain.any():
        raise ValueError('Released training file is contaminated')
    common = [c for c in train if c in test]
    if not common:
        raise ValueError('No shared numeric features')
    split = int(.8 * len(train))
    return _case(train[common].iloc[:split].to_numpy(), train[common].iloc[split:].to_numpy(),
                 test[common].to_numpy(), ytest, {
                     'case_id': family, 'source_family': family,
                     'protocol': 'released-train-80-fit-20-calibration-separate-test-v1',
                     'train_sha256': digest(spec['train']), 'test_sha256': digest(spec['test']),
                     'feature_names': common})


def prepare_power(root):
    parts, labels, groups, positions = [], [], [], []
    for numbers in (range(1, 10), range(10, 13), range(13, 16)):
        frames, ys, gs, ps = [], [], [], []
        for number in numbers:
            frame, y = _raw_frame(Path(root) / f'data{number}.csv', 'marker')
            frames.append(frame); ys.extend(y)
            gs.extend([number] * len(frame)); ps.extend(range(len(frame)))
        parts.append(pd.concat(frames, ignore_index=True))
        labels.append(np.asarray(ys)); groups.append(np.asarray(gs)); positions.append(np.asarray(ps))
    if any(list(p.columns) != list(parts[0].columns) for p in parts):
        raise ValueError('PowerSystem feature schemas differ')
    # Partition overlap is screened before imputation, without consulting test labels.
    filtered, overlap = remove_overlap(parts)
    arrays, entities, selected_labels = [], [], []
    for index, frame in enumerate(filtered):
        indices = frame.index.to_numpy()
        if index < 2:
            indices = indices[labels[index][indices] == 0]
        arrays.append(parts[index].iloc[indices].to_numpy(dtype=float))
        entities.append(gap_entities(positions[index][indices], groups[index][indices]))
        selected_labels.append(labels[index][indices])
    return _case(*arrays, selected_labels[2], {
        'case_id': 'PowerSystem-source-grouped', 'source_family': 'PowerSystem',
        'protocol': 'source-files-1-9-fit-10-12-calibration-13-15-test-v1',
        'temporal_order_verified': False, 'scenario_independence_verified': False,
        'interpretation': 'row-disjoint sampled-order diagnostic only',
        'overlap_screen': overlap, 'feature_names': list(parts[0].columns)}, entities)


def evaluate(labels, scores, calibration_scores):
    labels, scores, calibration_scores = map(np.asarray, (labels, scores, calibration_scores))
    if (labels.ndim != 1 or scores.shape != labels.shape or calibration_scores.ndim != 1
            or not len(calibration_scores) or not np.isfinite(scores).all()
            or not np.isfinite(calibration_scores).all()):
        raise ValueError('Invalid metric arrays')
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Evaluation requires both classes')
    result = {'auprc': float(average_precision_score(labels, scores)),
              'auroc': float(roc_auc_score(labels, scores)),
              'anomaly_prevalence': float(labels.mean()),
              'evaluated_rows': len(labels), 'calibration_rows': len(calibration_scores)}
    for alpha, name in ((.01, '01'), (.05, '05')):
        threshold = float(np.quantile(calibration_scores, 1 - alpha, method='higher'))
        prediction = scores > threshold
        result.update({f'pauc_fpr_{name}': float(roc_auc_score(labels, scores, max_fpr=alpha)),
                       f'threshold_{name}': threshold,
                       f'calibration_fpr_{name}': float(np.mean(calibration_scores > threshold)),
                       f'fpr_{name}': float(prediction[labels == 0].mean()),
                       f'recall_{name}': float(prediction[labels == 1].mean())})
    return result


class FittedControl:
    """Existing configurations with one fit and boundary-aware held-out scoring."""

    def __init__(self, method, seed):
        if method not in METHODS:
            raise ValueError(f'Unknown method: {method}')
        self.method, self.seed = method, seed

    def fit(self, train, entities):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from benchmarks.official_baselines.local_controls import DYNSPHERE_CONFIGURATION
        from pendulum_detector import MultiConceptEuclideanDetector, MultiConceptPendulumDetector
        if self.method.startswith('dynsphere'):
            euclidean = 'euclidean' in self.method
            cls = MultiConceptEuclideanDetector if euclidean else MultiConceptPendulumDetector
            params = dict(DYNSPHERE_CONFIGURATION)
            params.update(n_concepts=1 if 'one_regime' in self.method else 4, random_state=self.seed)
            params['position_weight' if euclidean else 'angular_weight'] = 1.
            self.model = cls(**params).fit(train, entities, np.arange(len(train)))
        elif self.method == 'pca_reconstruction':
            self.model = PCA(n_components=min(len(train)-1, train.shape[1]),
                             svd_solver='full', random_state=self.seed).fit(train)
            variance = np.cumsum(np.nan_to_num(self.model.explained_variance_ratio_, nan=0.))
            count = min(len(variance), int(np.searchsorted(variance, .95, side='left')) + 1)
            self.components = self.model.components_[:count]
        elif self.method == 'kmeans_distance':
            self.model = KMeans(n_clusters=min(4, len(train)), n_init=20,
                                random_state=self.seed).fit(train)
        else:
            valid = np.flatnonzero(entities[1:] == entities[:-1])
            if len(valid) < 2:
                raise ValueError('Insufficient within-entity autoregressive pairs')
            self.model = Ridge(alpha=1.).fit(train[valid], train[valid+1])
        return self

    def score(self, values, entities):
        if self.method.startswith('dynsphere'):
            result = self.model.decision_function(values, entities, np.arange(len(values)))
        elif self.method == 'pca_reconstruction':
            centered = values - self.model.mean_
            result = np.sum((centered - centered @ self.components.T @ self.components) ** 2, axis=1)
        elif self.method == 'kmeans_distance':
            result = self.model.transform(values).min(axis=1)
        else:
            result = np.zeros(len(values))
            valid = np.flatnonzero(entities[1:] == entities[:-1])
            prediction = self.model.predict(values[valid]).reshape(len(valid), values.shape[1])
            result[valid+1] = np.mean((values[valid+1] - prediction) ** 2, axis=1)
        if np.shape(result) != (len(values),) or not np.isfinite(result).all():
            raise ValueError('Invalid model scores')
        return result


def event_summary(labels, predictions, valid, entities):
    count, detected, delays = 0, 0, []
    for entity in np.unique(entities):
        indices = np.flatnonzero(entities == entity)
        y = np.asarray(labels)[indices]
        starts = np.flatnonzero(np.diff(np.r_[0, y, 0]) == 1)
        stops = np.flatnonzero(np.diff(np.r_[0, y, 0]) == -1)
        for start, stop in zip(starts, stops):
            original = indices[start:stop]
            if not np.asarray(valid)[original].all():
                continue
            count += 1
            hits = np.flatnonzero(np.asarray(predictions)[original])
            if len(hits):
                detected += 1
                delays.append(int(hits[0]))
    return {'eligible_events': count, 'event_recall': detected / count if count else None,
            'mean_detected_delay_samples': float(np.mean(delays)) if delays else None}


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze(entries, output, seeds=SEEDS):
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT / p for p in ('pendulum_detector.py', 'benchmarks/official_baselines/local_controls.py',
                                 'benchmarks/official_baselines/datasets.py',
                                 'benchmarks/real_world_datasets/run_real_world_datasets.py')]
    doc = {'protocol': 'held-out-normal-calibration-v1', 'methods': METHODS, 'seeds': seeds,
           'cases': [{**entry, 'source_hashes': {p: digest(p) for p in entry['paths']}}
                     for entry in entries],
           'implementation_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)},
           'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn')},
           'python': sys.version, 'manuscript_ready': False,
           'test_trim': [100, 10], 'calibration_trim': [10, 10], 'fit_fraction': .8}
    doc['manifest_id'] = _identity(doc)
    _json(Path(output) / 'manifests' / (doc['manifest_id'] + '.json'), doc)
    return doc


def run_one(case, method, seed, output, manifest_id):
    identity = _identity({'manifest': manifest_id, 'case': case['metadata']['case_id'],
                          'method': method, 'seed': seed})
    folder = Path(output) / 'runs' / identity
    record_path, artifact_path = folder / 'run.json', folder / 'scores.npz'
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        if record['status'] == 'complete':
            if (record.get('run_identity') != identity or record.get('manifest_id') != manifest_id
                    or record.get('method_id') != method or record.get('seed') != seed
                    or record.get('artifact_sha256') != digest(artifact_path)):
                raise ValueError('Saved score artifact integrity mismatch')
            return record
    record = {'run_identity': identity, 'manifest_id': manifest_id,
              'case_id': case['metadata']['case_id'], 'source_family': case['metadata']['source_family'],
              'method_id': method, 'seed': seed, 'status': 'running', 'manuscript_ready': False,
              'run_directory': str(folder.resolve()), 'metadata': case['metadata']}
    _json(record_path, record)
    started = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            model = FittedControl(method, seed).fit(case['train'], case['entities'][0])
            fit_seconds = time.perf_counter() - started
            calibration_scores = model.score(case['calibration'], case['entities'][1])
            scores = model.score(case['test'], case['entities'][2])
        valid, calibration_valid = case['test_mask'], case['calibration_mask']
        metrics = evaluate(case['labels'][valid], scores[valid], calibration_scores[calibration_valid])
        # Event recall uses complete events and original entity boundaries, never compressed rows.
        for name in ('01', '05'):
            events = event_summary(case['labels'], scores > metrics[f'threshold_{name}'],
                                   valid, case['entities'][2])
            metrics.update({f'{k}_{name}': v for k, v in events.items()})
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f'scores.{os.getpid()}.npz'
        np.savez_compressed(temporary, scores=scores, calibration_scores=calibration_scores,
                            labels=case['labels'], test_mask=valid, calibration_mask=calibration_valid,
                            entities=case['entities'][2], calibration_entities=case['entities'][1])
        temporary.replace(artifact_path)
        record.update(status='complete', metrics=metrics, fit_seconds=fit_seconds,
                       elapsed_seconds=time.perf_counter()-started, artifact_sha256=digest(artifact_path))
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        _json(record_path, record)
        raise
    _json(record_path, record)
    return record
