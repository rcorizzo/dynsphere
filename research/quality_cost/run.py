"""Fresh-process CPU measurements; existing campaign artifacts are read-only."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import random
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from research.evaluation_protocols.campaign import ROOT, FittedControl, _identity, digest
from research.submission_campaign.studies import BASE, BASE_ID, load_case

METHODS = ('dynsphere_euclidean_one_regime', 'dynsphere_euclidean',
           'dynsphere_one_regime', 'dynsphere', 'pca_reconstruction',
           'kmeans_distance', 'linear_ridge_ar1', 'mtad_gat', 'catch')
SEEDS = (37, 38, 39)
FAMILIES = ('BATADAL', 'HAI-20.07', 'MSL', 'SMAP', 'SMD')


def select_cases(catalogue, families=FAMILIES):
    selected = []
    for family in families:
        rows = sorted((r for r in catalogue if r['source_family'] == family
                       and r['eligible']), key=lambda r: (r['test_rows'], r['case_id']))
        if not rows:
            raise ValueError('No eligible cases for ' + family)
        selected.append(rows[(len(rows) - 1) // 2])
    return selected


def cpu_environment(environment=None):
    env = dict(os.environ if environment is None else environment)
    env.update(DYNSPHERE_DEVICE='cpu', CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               LOKY_MAX_CPU_COUNT='1', PYTHONPATH=str(ROOT))
    return env


def common_support(scores):
    scores = np.asarray(scores)
    mask = np.zeros(len(scores), dtype=bool)
    if scores.ndim != 1 or len(scores) <= 200:
        raise ValueError('Insufficient common support')
    mask[100:-100] = True
    if not np.isfinite(scores[mask]).all():
        raise ValueError('Unobserved or non-finite common support')
    return mask


def write_new_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def capture_random_state():
    state = {'numpy': np.random.get_state(), 'python': random.getstate()}
    if 'torch' in sys.modules:
        state['torch'] = sys.modules['torch'].get_rng_state()
    return state


def check_serialization(model, score, reference, random_state):
    payload = pickle.dumps(model, protocol=4)
    restored = pickle.loads(payload)
    np.random.set_state(random_state['numpy'])
    random.setstate(random_state['python'])
    if 'torch' in random_state:
        sys.modules['torch'].set_rng_state(random_state['torch'])
    actual = score(restored)
    mask = common_support(reference)
    np.testing.assert_allclose(actual[mask], reference[mask], rtol=1e-7, atol=1e-9)
    return len(payload)


def summarize(records):
    frame = pd.DataFrame(records)
    cost = ['fit_seconds', 'score_seconds', 'process_peak_rss_bytes', 'serialized_bytes']
    quality = ['pauc_fpr_05', 'auprc', 'auroc']
    groups = ['source_family', 'case_id', 'method_id']
    if frame.empty:
        return pd.DataFrame(columns=groups + ['planned_seeds', 'completed_seeds'] + cost + quality)
    if frame.duplicated(['case_id', 'method_id', 'seed']).any():
        raise ValueError('Duplicate method/case/seed records')
    for name in cost + quality:
        if name not in frame:
            frame[name] = np.nan
    complete = frame[frame.status.eq('complete')]
    a = complete.groupby(groups)[cost].median()
    b = complete.groupby(groups)[quality].mean()
    counts = frame.groupby(groups).size().rename('planned_seeds').to_frame()
    counts['completed_seeds'] = complete.groupby(groups).size().reindex(counts.index, fill_value=0)
    return counts.join(a).join(b).reset_index()


def ensure_configuration(path, configuration):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != configuration:
            raise ValueError('Changed frozen configuration')
    else:
        write_new_json(path, configuration)


def source_files():
    paths = [ROOT / 'pendulum_detector.py', ROOT / 'benchmarks/official_baselines/registry.json']
    for directory in ('research/quality_cost', 'research/evaluation_protocols',
                      'research/submission_campaign', 'benchmarks/official_baselines'):
        paths.extend((ROOT / directory).glob('*.py'))
    paths.extend((ROOT / 'benchmarks/official_baselines/adapters').glob('*.py'))
    return sorted(set(paths))


def validate_manifest(doc):
    from research.submission_campaign.deep import upstream_snapshot
    if _identity({k: v for k, v in doc.items() if k != 'manifest_id'}) != doc['manifest_id']:
        raise ValueError('Changed campaign manifest')
    for name, expected in doc['sources'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Changed runner/model source: ' + name)
    for method, expected in doc['upstream'].items():
        actual = upstream_snapshot(ROOT / 'benchmarks/official_baselines/.external' / method,
                                   expected['revision'])
        if actual != expected:
            raise ValueError('Changed upstream source: ' + method)


def validate_worker(output, expected, rows):
    output = Path(output)
    record = json.loads((output / 'run.json').read_text())
    if record.get('status') != 'complete' or any(record.get(k) != v for k, v in expected.items()):
        raise ValueError('Worker identity or configuration mismatch')
    if digest(output / 'scores.npy') != record.get('scores_sha256'):
        raise ValueError('Worker score integrity mismatch')
    scores = np.load(output / 'scores.npy', allow_pickle=False)
    if scores.shape != (rows,):
        raise ValueError('Worker score length mismatch')
    common_support(scores)
    return record, scores


def local_trial(method, seed, inputs):
    from threadpoolctl import threadpool_limits
    # Finish core imports before model timers.
    import pendulum_detector  # noqa: F401
    from benchmarks.official_baselines import local_controls  # noqa: F401
    inputs = Path(inputs)
    train, test = [np.load(inputs / (n + '.npy'), allow_pickle=False) for n in ('train', 'test')]
    train_entities, test_entities = [np.load(inputs / (n + '_entities.npy'), allow_pickle=False)
                                     for n in ('train', 'test')]
    before = peak_rss_bytes()
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        model = FittedControl(method, seed).fit(train, train_entities)
        fit = time.perf_counter() - start
        start = time.perf_counter()
        scores = model.score(test, test_entities)
        scoring = time.perf_counter() - start
        peak = peak_rss_bytes()
        payload = pickle.dumps(model, protocol=4)
        restored = pickle.loads(payload).score(test, test_entities)
    np.testing.assert_allclose(restored, scores, rtol=1e-10, atol=1e-10)
    return dict(fit_seconds=fit, score_seconds=scoring, process_peak_rss_bytes=peak,
                prefit_peak_rss_bytes=before, serialized_bytes=len(payload),
                serialization_verified=True, serialization='pickle4-fitted-estimator',
                device='cpu', threads=1), scores


def deep_trial(method, seed, inputs, configuration):
    from torch import nn
    import torch
    from torch.utils.data import DataLoader
    from threadpoolctl import threadpool_limits
    from research.submission_campaign.deep import prime_and_seed
    from benchmarks.official_baselines.adapters.common import source_directory
    inputs = Path(inputs)
    train, calibration, test = [np.load(inputs / (n + '.npy'), allow_pickle=False)
                                for n in ('train', 'calibration', 'test')]
    prime_and_seed(method, seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    window = int(configuration['window_size'])
    batch = int(configuration.get('batch_size', 128))
    if method == 'mtad_gat':
        sys.path.insert(0, str(source_directory(method)))
        try:
            from mtad_gat import MTAD_GAT
            from utils import SlidingWindowDataset
        finally:
            sys.path.pop(0)
    else:
        from benchmarks.official_baselines.adapters.catch import _load_corrected_catch
        from research.submission_campaign.deep import catch_fit_frame
        cls = _load_corrected_catch()
    before = peak_rss_bytes()
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        if method == 'catch':
            model = cls(num_epochs=int(configuration['epochs']), seq_len=window, batch_size=batch)
            combined = catch_fit_frame(train, calibration)
            model.detect_fit(combined, combined)
            if str(model.device) != 'cpu':
                raise ValueError('CATCH did not use CPU')
        else:
            torch.manual_seed(seed)
            loader = DataLoader(SlidingWindowDataset(torch.as_tensor(train, dtype=torch.float32), window),
                                batch_size=batch, shuffle=True)
            model = MTAD_GAT(n_features=train.shape[1], window_size=window,
                             out_dim=train.shape[1], use_gatv2=True).to('cpu')
            optimizer = torch.optim.Adam(model.parameters(), lr=float(configuration.get('lr', 1e-3)))
            criterion = nn.MSELoss()
            model.train()
            for _ in range(int(configuration['epochs'])):
                for windows, targets in loader:
                    optimizer.zero_grad()
                    forecast, reconstruction = model(windows)
                    loss = criterion(forecast, targets.squeeze(1)) + criterion(reconstruction, windows)
                    loss.backward()
                    optimizer.step()
            model.eval()
        fit = time.perf_counter() - start

        def score(fitted):
            if method == 'catch':
                columns = [f'feature_{i}' for i in range(test.shape[1])]
                raw = np.asarray(fitted.detect_score(pd.DataFrame(test, columns=columns))[0]).reshape(-1)
                result = np.full(len(test), np.nan)
                result[:min(len(raw), len(test))] = raw[:len(test)]
                return result
            loader = DataLoader(SlidingWindowDataset(torch.as_tensor(test, dtype=torch.float32), window),
                                batch_size=batch, shuffle=False)
            values = []
            with torch.no_grad():
                for windows, targets in loader:
                    forecast, reconstruction = fitted(windows)
                    values.append((torch.mean((forecast - targets.squeeze(1))**2, dim=1)
                                   + torch.mean((reconstruction - windows)**2, dim=(1, 2))).numpy())
            result = np.full(len(test), np.nan)
            raw = np.concatenate(values)
            result[window:window + len(raw)] = raw
            return result

        scoring_random_state = capture_random_state()
        start = time.perf_counter()
        scores = score(model)
        scoring = time.perf_counter() - start
        peak = peak_rss_bytes()
        snapshot = copy.copy(model)
        # Training-only loaders and optimizers are not needed for CATCH inference.
        if method == 'catch':
            for name in ('optimizer', 'optimizerM', 'train_data_loader', 'valid_data_loader',
                         'thre_loader', 'temp_anomaly_criterion', 'freq_anomaly_criterion'):
                if hasattr(snapshot, name):
                    delattr(snapshot, name)
        serialized_bytes = check_serialization(snapshot, score, scores, scoring_random_state)
    return dict(fit_seconds=fit, score_seconds=scoring, process_peak_rss_bytes=peak,
                prefit_peak_rss_bytes=before, serialized_bytes=serialized_bytes,
                serialization_verified=True, serialization='pickle4-fitted-estimator-without-training-loaders-or-optimizers',
                device='cpu', threads=1, torch_version=torch.__version__), scores


def worker(args):
    output = Path(args.output)
    if (output / 'run.json').exists():
        raise FileExistsError('Existing run record must be preserved')
    config = json.loads(Path(args.configuration).read_text())
    doc = json.loads(args.manifest.read_text())
    validate_manifest(doc)
    if config != doc['configurations'][args.method]:
        raise ValueError('Changed frozen worker configuration')
    case = next(c for c in doc['cases'] if c['case_id'] == args.case_id)
    for name, expected in case['input_sha256'].items():
        if digest(args.inputs / name) != expected:
            raise ValueError('Changed worker input')
    if args.method in ('catch', 'mtad_gat'):
        result, scores = deep_trial(args.method, args.seed, args.inputs, config)
    else:
        result, scores = local_trial(args.method, args.seed, args.inputs)
    common_support(scores)
    np.save(output / 'scores.npy', scores, allow_pickle=False)
    result.update(status='complete', method_id=args.method, seed=args.seed,
                  manifest_id=doc['manifest_id'], case_id=args.case_id,
                  scores_sha256=digest(output / 'scores.npy'), configuration=config,
                  python=sys.version, numpy_version=np.__version__,
                  sklearn_version=importlib.metadata.version('scikit-learn'))
    write_new_json(output / 'run.json', result)


def freeze(output, families, timeout=3600.):
    from research.submission_campaign.deep import configurations, upstream_snapshot, declared_revisions
    base = json.loads((BASE / 'manifests' / (BASE_ID + '.json')).read_text())
    catalogue, cases = [], {}
    for entry in base['cases']:
        if entry['source_family'] not in families:
            continue
        try:
            case = load_case(entry)
        except ValueError as error:
            catalogue.append(dict(case_id=entry['case_id'], source_family=entry['source_family'],
                                  test_rows=0, eligible=False, reason=str(error)))
            continue
        catalogue.append(dict(case_id=entry['case_id'], source_family=entry['source_family'],
                              train_rows=len(case['train']), calibration_rows=len(case['calibration']),
                              test_rows=len(case['test']), features=case['train'].shape[1], eligible=True,
                              entry=entry))
    selected = select_cases(catalogue, families)
    configs = configurations()
    for entry in selected:
        case = load_case(entry['entry'])
        folder = output / 'inputs' / entry['case_id']
        folder.mkdir(parents=True)
        arrays = dict(train=case['train'], calibration=case['calibration'], test=case['test'],
                      labels=case['labels'], train_entities=case['entities'][0],
                      calibration_entities=case['entities'][1], test_entities=case['entities'][2])
        for name, array in arrays.items():
            np.save(folder / (name + '.npy'), array, allow_pickle=False)
        entry['input_sha256'] = {p.name: digest(p) for p in sorted(folder.glob('*.npy'))}
    sources = source_files()
    doc = dict(selection='lower-median-test-rows-with-case-id-ties; frozen-before-new-training',
               base_manifest=BASE_ID, cases=selected, catalogue=catalogue, methods=METHODS, seeds=SEEDS,
               configurations={m: configs.get(m, {'fixed_local_control': m}) for m in METHODS},
               sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
               upstream={m: upstream_snapshot(ROOT / 'benchmarks/official_baselines/.external' / m,
                                             declared_revisions()[m]) for m in ('catch', 'mtad_gat')},
               hardware=dict(platform=platform.platform(), machine=platform.machine(),
                             processor=platform.processor(), logical_cpus=os.cpu_count()),
               environment=cpu_environment({}),
               timing='model initialization+fitting and full scoring separately; excludes imports,input preparation,serialization,metrics,startup',
               memory='fresh-process peak RSS through full scoring; includes runtime/imports/inputs; before serialization validation',
               serialization='pickle protocol 4; fitted estimators including retained state; CATCH training loaders/optimizers excluded; score roundtrip verified',
               support='first-and-last-100-test-rows-excluded', manuscript_ready=False)
    doc['worker_timeout_seconds'] = timeout
    doc['manifest_id'] = _identity(doc)
    write_new_json(output / 'manifest.json', doc)
    return doc


def controller(args):
    from sklearn.metrics import average_precision_score, roc_auc_score
    output = Path(args.output).resolve()
    if args.resume:
        doc = json.loads((output / 'manifest.json').read_text())
        validate_manifest(doc)
        if args.timeout != doc['worker_timeout_seconds']:
            raise ValueError('Changed frozen worker timeout')
    else:
        output.mkdir(parents=True, exist_ok=False)
        doc = freeze(output, args.families, args.timeout)
    print(json.dumps({'manifest_id': doc['manifest_id'], 'cases': [c['case_id'] for c in doc['cases']],
                      'planned_runs': len(doc['cases']) * len(doc['methods']) * len(doc['seeds'])}), flush=True)
    if args.dry_run:
        return
    records = []
    # Complete inexpensive methods across every family before deep methods.
    for method in doc['methods']:
        configuration = output / ('configuration-' + method + '.json')
        ensure_configuration(configuration, doc['configurations'][method])
        for case in doc['cases']:
            inputs = output / 'inputs' / case['case_id']
            for name, expected in case['input_sha256'].items():
                if digest(inputs / name) != expected:
                    raise ValueError('Frozen input changed')
            labels = np.load(inputs / 'labels.npy', allow_pickle=False)
            for seed in doc['seeds']:
                folder = output / 'runs' / case['case_id'] / method / ('seed-' + str(seed))
                folder.mkdir(parents=True, exist_ok=True)
                expected = dict(case_id=case['case_id'], method_id=method, seed=seed,
                                manifest_id=doc['manifest_id'], configuration=doc['configurations'][method])
                if (folder / 'result.json').exists():
                    record = json.loads((folder / 'result.json').read_text())
                    if any(record.get(k) != v for k, v in expected.items()):
                        raise ValueError('Stored result identity mismatch')
                    if record['status'] == 'complete':
                        validate_worker(folder / record['worker_subdirectory'], expected, len(labels))
                    records.append(record)
                    continue
                executable = (ROOT / 'benchmarks/official_baselines/.envs' / method / 'bin/python'
                              if method in ('catch', 'mtad_gat') else Path(sys.executable))
                completed = sorted(folder.glob('attempt-*/run.json'))
                attempt = completed[0].parent if completed else folder / ('attempt-%04d' % (len(list(folder.glob('attempt-*'))) + 1))
                attempt.mkdir(exist_ok=True)
                command = [str(executable), '-m', 'research.quality_cost.run', '--worker',
                           '--method', method, '--seed', str(seed), '--inputs', str(inputs),
                           '--configuration', str(configuration), '--output', str(attempt),
                           '--manifest', str(output / 'manifest.json'), '--case-id', case['case_id']]
                record = dict(expected, source_family=case['source_family'], status='failed',
                              worker_subdirectory=attempt.name)
                try:
                    if not completed:
                        with (attempt / 'stdout.log').open('x') as stdout, (attempt / 'stderr.log').open('x') as stderr:
                            result = subprocess.run(command, cwd=ROOT, env=cpu_environment(),
                                                    stdout=stdout, stderr=stderr, timeout=args.timeout)
                        if result.returncode:
                            raise RuntimeError('Worker exit code ' + str(result.returncode))
                    measurement, scores = validate_worker(attempt, expected, len(labels))
                    record.update(measurement)
                    mask = common_support(scores)
                    if set(np.unique(labels[mask])) != {0, 1}:
                        raise ValueError('Common support lacks both label classes')
                    record.update(auprc=float(average_precision_score(labels[mask], scores[mask])),
                                  auroc=float(roc_auc_score(labels[mask], scores[mask])),
                                  pauc_fpr_05=float(roc_auc_score(labels[mask], scores[mask], max_fpr=.05)),
                                  test_rows=len(scores), evaluated_rows=int(mask.sum()))
                except subprocess.TimeoutExpired as error:
                    record.update(status='timeout', error=repr(error), timeout_seconds=args.timeout)
                except Exception as error:
                    record.update(status='failed', error=repr(error))
                write_new_json(folder / 'result.json', record)
                records.append(record)
                print(json.dumps({k: record.get(k) for k in ('case_id', 'method_id', 'seed', 'status', 'fit_seconds', 'score_seconds', 'error')}), flush=True)
                pd.DataFrame(records).to_csv(output / 'progress.csv', index=False)
    summarize(records).to_csv(output / 'per_case.csv', index=False)
    pd.DataFrame(records).to_csv(output / 'per_run.csv', index=False)
    complete = sum(r['status'] == 'complete' for r in records)
    completion = dict(expected=len(records), complete=complete,
                   failed=len(records)-complete, manuscript_ready=complete == len(records),
                   manifest_id=doc['manifest_id'])
    ensure_configuration(output / 'completion.json', completion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--families', nargs='+', default=list(FAMILIES))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--timeout', type=float, default=3600.)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--configuration', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--case-id')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int, default=37)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        controller(args)


if __name__ == '__main__':
    main()
