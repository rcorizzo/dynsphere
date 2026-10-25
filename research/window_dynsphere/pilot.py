"""Frozen, isolated, resumable local pilot. No external job submission."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .controls import WindowRidgeControl
from .data import prepare_training
from .detector import WindowConditionedDynamicsDetector


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT/'results/window_dynsphere_v1'
CONFIGURATIONS = {
    f'window_k{k}_reconstruction{weight}': {'n_regimes': k, 'reconstruction_weight': weight}
    for k in (1, 4) for weight in (0, 1)
}
CONFIGURATIONS.update({
    'channel_independent': {'n_regimes': 1, 'reconstruction_weight': 1., 'channel_independent': True},
    'window_ridge': {'variant': 'ridge'},
    'neural_forecast_head': {'variant': 'head', 'n_regimes': 1, 'reconstruction_weight': 0.},
    'previous_neural_predictor': {'variant': 'previous'},
})


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def source_identity():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/p for p in (
        'predictive_generative_detector.py', 'range_metrics.py',
        'benchmarks/official_baselines/metrics.py', 'benchmarks/official_baselines/datasets.py',
        'benchmarks/official_baselines/cached_extension_metrics.py',
        'benchmarks/official_baselines/extension_campaign.py',
        'benchmarks/real_world_datasets/run_real_world_datasets.py')]
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'torch', 'vus')}
    return document_hash({'files': {str(p.relative_to(ROOT)): file_hash(p) for p in files},
                          'versions': versions, 'python': sys.version})


def select_pilot_entries(entries):
    selected = []
    for family in ('Exathlon', 'GHL'):
        candidates = [e for e in entries if e['source_family'] == family]
        selected += sorted(candidates, key=lambda e: hashlib.sha256(e['case_id'].encode()).hexdigest())[:3]
    candidates = [e for e in entries if e['source_family'] == 'GutenTAG']
    for mechanism in sorted({e['mechanism'] for e in candidates}):
        selected.append(min((e for e in candidates if e['mechanism'] == mechanism),
                            key=lambda e: (e['generation_seed'], e['case_id'])))
    for family in ('BATADAL', 'HAI-20.07', 'PowerSystem'):
        selected += [e for e in entries if e['source_family'] == family]
    return sorted(selected, key=lambda e: (e['source_family'], e['case_id']))


def freeze_manifest(entries, output_dir, configurations=None):
    cases = []
    for original in entries:
        entry = dict(original)
        entry['source_hashes'] = {}
        for field in ('path', 'train', 'test'):
            if field in entry:
                path = Path(entry[field])
                actual = file_hash(path) if path.is_file() else None
                expected = entry.get('expected_hashes', {}).get(field)
                if expected and actual != expected:
                    raise ValueError(f'Frozen source hash mismatch: {path}')
                entry['source_hashes'][field] = actual
        cases.append(entry)
    doc = {'protocol': 'window-dynsphere-pilot-v1', 'seed': 37, 'cases': cases,
           'configurations': configurations or CONFIGURATIONS, 'source_identity': source_identity(),
           'manuscript_ready': False, 'status': 'exploratory'}
    doc['manifest_id'] = document_hash(doc)
    path = Path(output_dir)/'manifests'/f"{doc['manifest_id']}.json"
    if path.exists() and json.loads(path.read_text()) != doc:
        raise ValueError('Frozen manifest differs')
    atomic_json(path, doc)
    return path


def freeze_pilot(output_dir=DEFAULT_OUTPUT):
    from benchmarks.official_baselines.datasets import REAL_WORLD_SPECS
    frame = pd.read_csv(ROOT/'results/extension_audit_20261023/vus_seven_methods_per_run.csv')
    cases = []
    for case_id, group in frame[frame.source_family.isin(['Exathlon', 'GHL'])].groupby('case_id'):
        row = group.iloc[0]
        paths = list((ROOT/'benchmarks/tsb_ad_m/data/TSB-AD-M').rglob(row.trajectory_id+'.csv'))
        if len(paths) != 1:
            raise ValueError(f'Expected one frozen TSB source for {case_id}')
        original = json.loads((Path(row.run_directory)/'run.json').read_text())
        cases.append({'case_id': case_id, 'source_family': row.source_family, 'kind': 'tsb',
                      'path': str(paths[0]), 'expected_hashes': {'path': original['hyperparameters']['raw_source_sha256']['source']}})
    native = ROOT/'data/gutentag_extension_v1/native_manifest.json'
    for pair in json.loads(native.read_text())['cases']:
        cases.append({**pair, 'case_id': 'GutenTAG-native-v1/'+pair['case_name'],
                      'source_family': 'GutenTAG', 'kind': 'native',
                      'train': str((native.parent/pair['train']).resolve()),
                      'test': str((native.parent/pair['test']).resolve()),
                      'expected_hashes': {'train': pair['train_sha256'], 'test': pair['test_sha256']}})
    for family, paths in REAL_WORLD_SPECS.items():
        entry = {'case_id': family+'/complete', 'source_family': family, 'kind': 'principal',
                 'train': str(paths['train']), 'test': str(paths['test'])}
        if Path(entry['train']).resolve() == Path(entry['test']).resolve():
            entry['preflight_error'] = 'Historical training/test source overlap requires a disjoint protocol'
        cases.append(entry)
    selected = select_pilot_entries(cases)
    if len(selected) != 15 or len({e.get('mechanism') for e in selected if e['source_family']=='GutenTAG'}) != 6:
        raise ValueError('The declared 15-case pilot is incomplete')
    return freeze_manifest(selected, output_dir)


def _load(entry, seed):
    from benchmarks.official_baselines.extension_campaign import load_native_case, load_suffix_case
    from benchmarks.official_baselines.datasets import load_real_world_case
    if entry['kind'] == 'native':
        return load_native_case(Path(entry['train']), Path(entry['test']), entry['source_family'],
                                entry['case_name'], seed)
    if entry['kind'] == 'tsb':
        return load_suffix_case(Path(entry['path']), seed)
    return load_real_world_case(entry['source_family'], seed)


def _array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _read_manifest(path):
    doc = json.loads(Path(path).read_text())
    body = {k: v for k, v in doc.items() if k != 'manifest_id'}
    if doc.get('manifest_id') != document_hash(body) or doc.get('source_identity') != source_identity():
        raise ValueError('Frozen manifest/source identity no longer matches')
    return doc


def run_case(manifest_path, case_id, configuration_id, result_root):
    doc = _read_manifest(manifest_path)
    entry = next(e for e in doc['cases'] if e['case_id'] == case_id)
    cfg = dict(doc['configurations'][configuration_id])
    window = int(cfg.get('window', 100))
    for field, expected in entry.get('source_hashes', {}).items():
        if expected is None and Path(entry[field]).exists():
            raise ValueError(f'Input availability changed; freeze a new manifest: {field}')
        if expected is not None and file_hash(entry[field]) != expected:
            raise ValueError(f'Input hash changed: {field}')
    identity = document_hash({'manifest': doc['manifest_id'], 'case': entry, 'configuration': cfg})
    folder = Path(result_root)/'runs'/identity
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder/'run.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record['status'] == 'complete':
            expected = {'run_identity': identity, 'case_id': case_id, 'configuration_id': configuration_id,
                        'manifest_id': doc['manifest_id'], 'source_identity': doc['source_identity'],
                        'hyperparameters': cfg, 'seed': doc['seed']}
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError('Cached record identity integrity mismatch')
            if set(record.get('artifact_hashes', {})) != {'scores.npy', 'labels.npy'}:
                raise ValueError('Cached artifact integrity metadata is incomplete')
            for name, expected in record['artifact_hashes'].items():
                if not (folder/name).is_file() or file_hash(folder/name) != expected:
                    raise ValueError(f'Cached artifact integrity mismatch: {name}')
            atomic_json(folder/'latest_attempt.json', {'run_identity':identity,'status':'complete',
                                                       'recorded_at_ns':time.time_ns()})
            return {**record, 'reused': True}
    record = {'run_identity': identity, 'case_id': case_id, 'source_family': entry['source_family'],
              'configuration_id': configuration_id, 'hyperparameters': dict(cfg), 'seed': doc['seed'],
              'manifest_id': doc['manifest_id'], 'source_identity': doc['source_identity'],
              'run_directory': str(folder.resolve()), 'manuscript_ready': False, 'status': 'running'}
    try:
        if entry.get('preflight_error'):
            raise ValueError(entry['preflight_error'])
        case = _load(entry, doc['seed'])
        if case.case_id != case_id:
            raise ValueError('Loaded case identity differs from the frozen manifest')
        labels = np.asarray(case.labels)
        if len(labels) <= window or set(np.unique(labels[window:])) != {0, 1}:
            raise ValueError('Evaluation requires both label classes after the common warmup')
        train, test = np.asarray(case.train), np.asarray(case.test)
        ids, test_ids = np.zeros(len(train), dtype=int), np.zeros(len(test), dtype=int)
        prepare_training(train, ids, None, window, cfg.get('max_targets', 20000), doc['seed'])
        record['data_digest'] = _array_hash(train, test, labels)
    except (ValueError, FileNotFoundError) as exc:
        record.update(status='incompatible', error=f'{type(exc).__name__}: {exc}')
        atomic_json(record_path, record)
        atomic_json(folder/'latest_attempt.json', {'run_identity':identity,'status':record['status'],
                                                  'recorded_at_ns':time.time_ns()})
        return record
    atomic_json(record_path, record)
    started = time.perf_counter()
    try:
        variant = cfg.pop('variant', 'local')
        if variant == 'previous':
            from predictive_generative_detector import PredictiveGenerativeDetector
            model = PredictiveGenerativeDetector(geometry='euclidean', latent_dim=min(6, train.shape[1]),
                                                 hidden_dim=48, n_modes=1, epochs=15, sequence_length=128,
                                                 calibration_fraction=.2, random_state=doc['seed'])
            model.fit(train, ids)
            score_function = model.decision_function
        elif variant == 'ridge':
            model = WindowRidgeControl(window=window, random_state=doc['seed']).fit(train, ids)
            score_function = model.decision_function
        else:
            cfg.setdefault('random_state', doc['seed'])
            model = WindowConditionedDynamicsDetector(**cfg)
            neural_options = model.get_params()
            neural_options.pop('n_regimes')
            key = document_hash({'source': doc['source_identity'], 'data': record['data_digest'],
                                 'neural_configuration': neural_options})
            cache = Path(result_root)/'representation_cache'/key
            previous = None
            if cache.with_suffix('.json').exists():
                provenance = json.loads(cache.with_suffix('.json').read_text())
                if provenance['sha256'] != file_hash(cache.with_suffix('.pkl')):
                    raise ValueError('Representation cache integrity mismatch')
                previous = pickle.loads(cache.with_suffix('.pkl').read_bytes())
            model.fit(train, ids, representation_from=previous)
            if previous is None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.with_suffix('.pkl').write_bytes(pickle.dumps(model))
                atomic_json(cache.with_suffix('.json'), {'sha256': file_hash(cache.with_suffix('.pkl')),
                                                        'training_identity': model.training_identity_})
            record['representation_reused'] = previous is not None
            score_function = model.forecast_head_scores if variant == 'head' else model.decision_function
        record['fit_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        scores = score_function(test, test_ids)
        record['score_seconds'] = time.perf_counter() - started
        scores[:window] = np.nan
        if scores.shape != labels.shape or not np.isfinite(scores[window:]).all():
            raise ValueError('Nonfinite or unaligned post-warmup scores')
        np.save(folder/'scores.npy', scores, allow_pickle=False)
        np.save(folder/'labels.npy', labels, allow_pickle=False)
        from benchmarks.official_baselines.metrics import compute_metrics
        record['metrics'] = compute_metrics(labels[window:], scores[window:], .01, None)
        if record['metrics'].get('vus_backend') != 'official_vus':
            raise ValueError('Official VUS is required')
        record['evaluation_warmup_rows'] = window
        record['metadata'] = getattr(model, 'metadata_', {'base_method': 'PredictiveGenerativeDetector',
                                                        'calibration_fraction': .2})
        record['peak_memory_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform=='darwin' else 1024)
        record['artifact_hashes'] = {name: file_hash(folder/name) for name in ('scores.npy', 'labels.npy')}
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    atomic_json(record_path, record)
    atomic_json(folder/'latest_attempt.json', {'run_identity':identity,'status':record['status'],
                                              'recorded_at_ns':time.time_ns()})
    return record


def record_process_failure(manifest_path, case_id, configuration_id, result_root, reason):
    doc = _read_manifest(manifest_path)
    entry = next(e for e in doc['cases'] if e['case_id']==case_id)
    cfg = doc['configurations'][configuration_id]
    identity = document_hash({'manifest':doc['manifest_id'], 'case':entry, 'configuration':cfg})
    directory = Path(result_root)/'runs'/identity
    path = directory/'run.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    record = {**previous, 'run_identity':identity, 'case_id':case_id, 'configuration_id':configuration_id,
              'source_family':entry['source_family'], 'manifest_id':doc['manifest_id'],
              'source_identity':doc['source_identity'], 'seed':doc['seed'],
              'run_directory':str(directory.resolve()), 'manuscript_ready':False,
              'status':'timeout' if reason=='timeout' else 'failed',
              'error':f'Worker process did not complete: {reason}'}
    attempt = Path(result_root)/'attempt_failures'/identity/f'{time.time_ns()}.json'
    atomic_json(attempt,record)
    atomic_json(directory/'latest_attempt.json',record)
    if previous.get('status') != 'complete':
        atomic_json(path, record)
    return record


def reference_rows(records, result_root):
    from benchmarks.official_baselines.cached_extension_metrics import load_scores
    from benchmarks.official_baselines.metrics import compute_metrics
    tables = [ROOT/'results/extension_audit_20261023/per_run_all_available.csv',
              ROOT/'results/official_baselines_v1/aggregate/f87ae4a73384/per_run.csv']
    frames = [pd.read_csv(path) for path in tables if path.is_file()]
    if not frames:
        return [], [{'status':'pending', 'error':'Historical reference tables unavailable'}]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.method_id.isin(['dynsphere_euclidean','dynsphere_euclidean_one_regime',
                                      'mtad_gat','predictive_euclidean_single']) & (frame.seed==37)]
    roots = [ROOT/'results/extension_v1/runs',
             ROOT/'results/hpc_imports/au-extension-20261023/extension_baselines_v2/runs',
             ROOT/'results/official_baselines_v1/runs']
    case_records = {r['case_id']:r for r in records if r['status']=='complete'}
    rows, issues = [], []
    for case_id, completed in case_records.items():
        expected_labels = np.load(Path(completed['run_directory'])/'labels.npy', allow_pickle=False)
        for method in ('dynsphere_euclidean','dynsphere_euclidean_one_regime','mtad_gat','predictive_euclidean_single'):
            candidates = frame[(frame.case_id==case_id) & (frame.method_id==method)]
            if len(candidates) != 1:
                issues.append({'case_id':case_id,'method_id':method,'status':'pending',
                               'error':'Exactly one historical reference record was not found'})
                continue
            row = candidates.iloc[0].to_dict()
            try:
                arrays = None
                for root in roots:
                    relative = str(row['run_directory']).rsplit('/runs/',1)[-1]
                    if (root/relative/'run.json').is_file():
                        arrays = load_scores(row, root)
                        break
                if arrays is None:
                    raise FileNotFoundError('Historical reference score arrays unavailable')
                labels, scores = arrays
                if not np.array_equal(labels, expected_labels[1:]):
                    raise ValueError('Historical reference has different evaluation labels/support')
                labels, scores = labels[99:], scores[99:]
                key = document_hash({'arrays':_array_hash(labels,scores),'source':source_identity()})
                cache = Path(result_root)/'reference_metric_cache'/f'{key}.json'
                if cache.is_file():
                    metrics = json.loads(cache.read_text())
                else:
                    metrics = compute_metrics(labels,scores,.01,None)
                    if metrics['vus_backend'] != 'official_vus':
                        raise ValueError('Reference metrics require official VUS')
                    atomic_json(cache, metrics)
                rows.append({'case_id':case_id,'source_family':completed['source_family'],
                             'configuration_id':'historical_'+method,'seed':37,
                             'evaluation_warmup_rows':100,'reference_run_identity':row['run_identity'],
                             'training_budget':'historical protocol; see declared implementation',**metrics})
            except Exception as exc:
                issues.append({'case_id':case_id,'method_id':method,'status':'pending',
                               'error':f'{type(exc).__name__}: {exc}'})
    return rows, issues


def aggregate(manifest_path, result_root, include_references=False):
    doc = _read_manifest(manifest_path)
    records = [json.loads(p.read_text()) for p in (Path(result_root)/'runs').glob('*/run.json')]
    records = [r for r in records if r['manifest_id'] == doc['manifest_id']]
    current_records = []
    for record in records:
        attempt = Path(record['run_directory'])/'latest_attempt.json'
        if attempt.exists():
            latest = json.loads(attempt.read_text())
            if latest['run_identity'] != record['run_identity']:
                raise ValueError('Latest attempt identity integrity mismatch')
            if latest['status'] in ('failed','timeout'):
                record = {**record,'status':latest['status'],'error':latest.get('error','Worker attempt failed')}
        current_records.append(record)
    records = current_records
    entries = {e['case_id']:e for e in doc['cases']}
    seen = set()
    for r in records:
        key = (r['case_id'],r['configuration_id'])
        expected = document_hash({'manifest':doc['manifest_id'],'case':entries[r['case_id']],
                                  'configuration':doc['configurations'][r['configuration_id']]})
        if key in seen or r['run_identity'] != expected:
            raise ValueError('Aggregate record identity/coverage integrity mismatch')
        seen.add(key)
        if r['status']=='complete':
            for name, digest in r['artifact_hashes'].items():
                if file_hash(Path(r['run_directory'])/name) != digest:
                    raise ValueError('Aggregate artifact integrity mismatch')
    complete = [{**{k:r[k] for k in ('case_id','source_family','configuration_id','seed','run_identity',
                                    'fit_seconds','score_seconds','peak_memory_bytes')}, **r['metrics']}
                for r in records if r['status']=='complete']
    out = Path(result_root)/'summaries'/doc['manifest_id']
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'completion.json', {'expected_records': len(doc['cases'])*len(doc['configurations']),
                                      'observed_records': len(records),
                                      'outcomes': pd.Series([r['status'] for r in records]).value_counts().to_dict(),
                                      'manuscript_ready': False})
    atomic_json(out/'noncomplete.json', [r for r in records if r['status']!='complete'])
    if complete:
        frame = pd.DataFrame(complete)
        frame.to_csv(out/'per_run.csv', index=False)
        family = frame.groupby(['source_family','configuration_id'])[['auprc','auroc','vus_pr','vus_roc']].mean()
        family['case_count'] = frame.groupby(['source_family','configuration_id']).case_id.nunique()
        family.to_csv(out/'per_family_available.csv')
        counts = frame.groupby('case_id').configuration_id.nunique()
        common = counts[counts==len(doc['configurations'])].index
        matched = frame[frame.case_id.isin(common)]
        matched.to_csv(out/'per_run_matched.csv',index=False)
        matched_family = matched.groupby(['source_family','configuration_id'])[['auprc','auroc','vus_pr','vus_roc']].mean()
        matched_family['case_count'] = matched.groupby(['source_family','configuration_id']).case_id.nunique()
        matched_family.to_csv(out/'per_family_matched.csv')
    if include_references:
        references, issues = reference_rows(records,result_root)
        pd.DataFrame(references).to_csv(out/'historical_references_warmup100.csv',index=False)
        atomic_json(out/'reference_issues.json',issues)
    return str(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--case-id')
    parser.add_argument('--configuration-id')
    args = parser.parse_args()
    manifest = args.manifest or freeze_pilot(args.output_dir)
    doc = _read_manifest(manifest)
    if args.case_id:
        if not args.configuration_id:
            parser.error('--case-id requires --configuration-id')
        result = run_case(manifest, args.case_id, args.configuration_id, args.output_dir)
        print(json.dumps(result, indent=2), flush=True)
        return int(result['status']=='failed')
    if args.aggregate:
        print(aggregate(manifest, args.output_dir, include_references=True))
        return 0
    print(json.dumps({'manifest': str(manifest), 'cases': len(doc['cases']),
                      'configurations': len(doc['configurations']),
                      'logical_executions': len(doc['cases'])*len(doc['configurations']),
                      'source_identity': doc['source_identity'], 'manuscript_ready': False}), flush=True)
    if args.run or args.resume:
        process_failures = 0
        logs = args.output_dir/'logs'/doc['manifest_id']
        logs.mkdir(parents=True, exist_ok=True)
        order = {'GutenTAG':0, 'Exathlon':1, 'GHL':2, 'BATADAL':3, 'HAI-20.07':4, 'PowerSystem':5}
        for entry in sorted(doc['cases'], key=lambda e:(order[e['source_family']],e['case_id'])):
            for configuration in doc['configurations']:
                log = logs/(document_hash([entry['case_id'],configuration])+'.log')
                cmd = [sys.executable, '-m', 'research.window_dynsphere.pilot', '--manifest', str(manifest),
                       '--output-dir', str(args.output_dir), '--case-id', entry['case_id'],
                       '--configuration-id', configuration]
                with log.open('a') as handle:
                    try:
                        result = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                                env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                                                     'MKL_NUM_THREADS':'1', 'VECLIB_MAXIMUM_THREADS':'1'}, timeout=7200)
                        status = result.returncode
                    except subprocess.TimeoutExpired:
                        status = 'timeout'
                print(json.dumps({'case':entry['case_id'], 'configuration':configuration,
                                  'process_exit':status, 'log':str(log)}), flush=True)
                if status != 0:
                    process_failures += 1
                    record_process_failure(manifest,entry['case_id'],configuration,args.output_dir,status)
                aggregate(manifest, args.output_dir)
        aggregate(manifest,args.output_dir,include_references=True)
        return int(process_failures > 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
