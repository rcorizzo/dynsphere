"""Frozen PCA-95 sensitivity campaign, gated on completion of CPU timing."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.evaluation_protocols.campaign import ROOT, FittedControl, _identity, digest
from research.submission_campaign import deep
from research.submission_campaign.studies import BASE, BASE_ID, load_case
from research.submission_recovery.run import run_lock

DYN = ('dynsphere_euclidean_one_regime', 'dynsphere_euclidean', 'dynsphere_one_regime', 'dynsphere')
LOCAL = DYN + ('kmeans_distance', 'linear_ridge_ar1')
CPU = ('dagmm', 'omni_anomaly_tsb_ad')
GPU = ('catch', 'mtad_gat', 'anomaly_transformer', 'tranad')
METHODS = LOCAL + CPU + GPU
GROUPS = dict(local=LOCAL, external_cpu=CPU, gpu=GPU, all=METHODS)
FAMILIES = ('BATADAL', 'HAI-20.07', 'MSL', 'SMAP', 'SMD')
SEEDS = (37, 38, 39)
TIMING_ID = 'c2899e2c55b383ded3bd127821821844cc63f8b8131ee509830b7759b7a94398'
TIMING_PROOF = ROOT / 'results/quality_cost_20261026/five-family-cpu-v1/completion.json'
DIAGNOSTICS = ROOT / 'results/pca_diagnostics_20261026/fitting-only-v1/per_case.csv'
REFERENCE = ROOT / 'results/submission_gpu_l40_20261025/vus_20261025/per_run_matched.csv'
OUTPUT = ROOT / 'results/pca95_sensitivity_20261026/v1'
RECOVERY_RUNTIME = ROOT / 'results/submission_recovery_20261024/deployment-evidence/runtime-evidence.json'


def choose_dimension(ratios):
    ratios = np.asarray(ratios, dtype=float)
    if (ratios.ndim != 1 or not len(ratios) or not np.isfinite(ratios).all()
            or (ratios < 0).any() or not np.isclose(ratios.sum(), 1., atol=1e-7)):
        raise ValueError('Invalid full PCA variance spectrum')
    q95 = int(np.searchsorted(np.cumsum(ratios), .95, side='left') + 1)
    return min(len(ratios), max(5, q95))


def fit_projection(train, calibration, test):
    arrays = [np.asarray(a, dtype=float) for a in (train, calibration, test)]
    if (any(a.ndim != 2 or not np.isfinite(a).all() for a in arrays)
            or len({a.shape[1] for a in arrays}) != 1):
        raise ValueError('Invalid PCA input arrays')
    scaler = StandardScaler().fit(arrays[0])
    standardized = [scaler.transform(a) for a in arrays]
    pca = PCA(n_components=min(standardized[0].shape), svd_solver='full').fit(standardized[0])
    q = choose_dimension(pca.explained_variance_ratio_)
    projected = [pca.transform(a)[:, :q] for a in standardized]
    metadata = dict(q=q, q95=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .95) + 1),
                    retained_variance=float(pca.explained_variance_ratio_[:q].sum()),
                    fit_rows=len(arrays[0]), input_features=arrays[0].shape[1], whitening=False,
                    scaling_mean=scaler.mean_.tolist(), scaling_scale=scaler.scale_.tolist(),
                    pca_mean=pca.mean_.tolist(), components=pca.components_[:q].tolist())
    return projected, metadata


def dynsphere_model(method, seed, q):
    from benchmarks.official_baselines.local_controls import DYNSPHERE_CONFIGURATION
    from pendulum_detector import MultiConceptEuclideanDetector, MultiConceptPendulumDetector
    if method not in DYN:
        raise ValueError('Unknown DynSphere variant')
    euclidean = 'euclidean' in method
    parameters = dict(DYNSPHERE_CONFIGURATION)
    parameters.update(latent_dim=q, n_concepts=1 if 'one_regime' in method else 4,
                      random_state=seed)
    parameters['position_weight' if euclidean else 'angular_weight'] = 1.
    return (MultiConceptEuclideanDetector if euclidean else MultiConceptPendulumDetector)(**parameters)


def timing_guard(path=TIMING_PROOF):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError('The timing experiment has not finished; new training is gated')
    proof = json.loads(path.read_text())
    if (proof.get('manifest_id') != TIMING_ID or proof.get('expected') != 135
            or proof.get('complete', 0) + proof.get('failed', 0) != 135):
        raise RuntimeError('Incomplete or unrelated timing completion proof')


def jobs_for(cases, methods=METHODS, seeds=SEEDS):
    return [dict(case_id=c['case_id'], method_id=m, seed=s)
            for c in cases for m in methods for s in seeds]


def write_new_json(path, value):
    """Publish one complete JSON record without replacing an existing target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()


def source_files():
    from research.quality_cost.run import source_files as timing_sources
    return sorted(set(timing_sources() + list(Path(__file__).parent.glob('*.py'))
                      + list((ROOT / 'research/submission_recovery').glob('*.py'))
                      + [ROOT / 'range_metrics.py']))


def freeze(output, timeout=43200.):
    base_path = BASE / 'manifests' / (BASE_ID + '.json')
    base = json.loads(base_path.read_text())
    entries = {e['case_id']: e for e in base['cases'] if e['source_family'] in FAMILIES}
    dimensions = pd.read_csv(DIAGNOSTICS)
    if len(dimensions) != 60 or dimensions.case_id.duplicated().any():
        raise ValueError('Unexpected PCA diagnostic cohort')
    reference = pd.read_csv(REFERENCE)
    primary = sorted(set(reference.case_id))
    if len(primary) != 59:
        raise ValueError('Unexpected primary matched cohort')
    cases = []
    for row in dimensions.sort_values(['source_family', 'case_id']).to_dict('records'):
        entry = entries[row['case_id']]
        if entry['source_family'] != row['source_family']:
            raise ValueError('PCA diagnostic source family differs')
        cases.append(dict(entry=entry, case_id=row['case_id'], source_family=row['source_family'],
                          q=min(int(row['features']), max(5, int(row['q95']))), q95=int(row['q95'])))
    configurations = deep.configurations()
    configurations['omni_anomaly_tsb_ad']['validation_partition'] = 'reserved-normal-calibration-v1'
    doc = dict(base_manifest_id=BASE_ID, base_manifest_sha256=digest(base_path),
               cases=cases, primary_case_ids=primary, methods=list(METHODS), seeds=list(SEEDS),
               dimension_rule='max(5,q95)-capped-by-input-dimension; full fitting-only PCA spectrum',
               transformations=dict(dynsphere='internal latent dimension only; original-feature dynamics retained',
                                    competitors='fitting-only StandardScaler and PCA input projection; no whitening'),
               configurations={m: configurations.get(m, {'fixed_local_control': m}) for m in METHODS},
               sources={str(p.relative_to(ROOT)): digest(p) for p in source_files()},
               upstream={m: deep.upstream_snapshot(ROOT / 'benchmarks/official_baselines/.external' / m,
                                                  deep.declared_revisions()[m]) for m in deep.EXTERNAL},
               reference_path=str(REFERENCE.relative_to(ROOT)), reference_sha256=digest(REFERENCE),
               diagnostics_sha256=digest(DIAGNOSTICS), timing_manifest_id=TIMING_ID,
               runtime_evidence=json.loads(RECOVERY_RUNTIME.read_text()),
               support='first-and-last-100-test-rows-excluded', worker_timeout_seconds=timeout,
               primary_metric='standardized partial AUROC; max_fpr=.05; equal-family mean',
               exploratory=True, manuscript_ready=False)
    doc['manifest_id'] = _identity(doc)
    write_new_json(output / 'manifest.json', doc)
    return doc


def validate_manifest(doc):
    if _identity({k: v for k, v in doc.items() if k != 'manifest_id'}) != doc['manifest_id']:
        raise ValueError('Changed PCA-95 manifest')
    for name, expected in doc['sources'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Changed PCA-95 source: ' + name)
    for method, expected in doc['upstream'].items():
        if deep.upstream_snapshot(ROOT / 'benchmarks/official_baselines/.external' / method,
                                  expected['revision']) != expected:
            raise ValueError('Changed PCA-95 upstream: ' + method)


def validate_inputs(folder, case):
    metadata = json.loads((folder / 'inputs.json').read_text())
    if metadata['case_id'] != case['case_id'] or metadata['projection']['q'] != case['q']:
        raise ValueError('Changed PCA-95 input identity or dimension')
    for name, expected in metadata['sha256'].items():
        if digest(folder / name) != expected:
            raise ValueError('Changed PCA-95 frozen input')
    return metadata


def verify_input_binding(doc, case_id, path):
    if digest(path) != doc['input_metadata_sha256'][case_id]:
        raise ValueError('Changed frozen input metadata')


@contextmanager
def preparation_lock(root):
    # Array shards wait while one controller prepares shared immutable arrays.
    with (Path(root) / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def verify_runtime(doc, method, overlay=None):
    from research.submission_recovery.run import OVERLAY, overlay_files, versions
    evidence = doc['runtime_evidence']
    if method == 'dagmm':
        files = overlay_files(OVERLAY if overlay is None else overlay)
        if not files or _identity(files) != evidence['overlay_sha256']:
            raise ValueError('DAGMM compatibility overlay differs from frozen runtime')
    actual = versions()
    if actual != evidence['versions'][method]:
        raise ValueError('Corrected CPU runtime differs from frozen evidence')
    return actual


def prepare_execution(root, planning, proof):
    """Bind prepared arrays to a new immutable execution manifest after timing."""
    timing_guard(proof)
    path = root / 'execution-manifest.json'
    with preparation_lock(root):
        if path.exists():
            doc = json.loads(path.read_text())
            validate_manifest(doc)
            if doc['planning_manifest_id'] != planning['manifest_id']:
                raise ValueError('Execution and planning manifests differ')
            for case in doc['cases']:
                verify_input_binding(doc, case['case_id'], root / 'inputs' / case['case_id'] / 'inputs.json')
            return doc
        hashes = {}
        for case in planning['cases']:
            folder, _ = prepare_inputs(root, case)
            hashes[case['case_id']] = digest(folder / 'inputs.json')
        doc = {k: v for k, v in planning.items() if k != 'manifest_id'}
        doc.update(planning_manifest_id=planning['manifest_id'], input_metadata_sha256=hashes,
                   timing_completion_sha256=digest(proof))
        doc['manifest_id'] = _identity(doc)
        write_new_json(path, doc)
    return doc


def prepare_inputs(root, case):
    from benchmarks.official_baselines.contract import _atomic_save_array
    folder = root / 'inputs' / case['case_id']
    folder.mkdir(parents=True, exist_ok=True)
    with run_lock(folder):
        if (folder / 'inputs.json').is_file():
            return folder, validate_inputs(folder, case)
        values = load_case(case['entry'])
        with threadpool_limits(limits=1):
            transformed, projection = fit_projection(values['train'], values['calibration'], values['test'])
        if projection['q'] != case['q'] or projection['q95'] != case['q95']:
            raise ValueError('Recomputed PCA dimension differs from frozen diagnostics')
        arrays = {'labels.npy': values['labels']}
        for i, name in enumerate(('train', 'calibration', 'test')):
            arrays['raw/' + name + '.npy'] = values[name]
            arrays['projected/' + name + '.npy'] = transformed[i]
            arrays[name + '_entities.npy'] = values['entities'][i]
        for name, array in arrays.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                if not np.array_equal(np.load(target, allow_pickle=False), array):
                    raise ValueError('Partial cached PCA-95 input differs')
            else:
                _atomic_save_array(target, array)
        metadata = dict(case_id=case['case_id'], projection=projection,
                        sha256={name: digest(folder / name) for name in sorted(arrays)})
        write_new_json(folder / 'inputs.json', metadata)
    return folder, metadata


def expected_identity(doc, case, method, seed):
    expected = dict(manifest_id=doc['manifest_id'], case_id=case['case_id'],
                source_family=case['source_family'], method_id=method, seed=seed,
                variant=method + '+PCA95-min5', q=case['q'], configuration=doc['configurations'][method])
    if method in CPU:
        expected['runtime_evidence'] = doc['runtime_evidence']
    return expected


def validate_measurement(folder, expected, input_metadata):
    result = json.loads((folder / 'measurement.json').read_text())
    if (result.get('status') != 'complete' or any(result.get(k) != v for k, v in expected.items())
            or result.get('input_sha256') != input_metadata['sha256']):
        raise ValueError('PCA-95 worker identity or inputs differ')
    if digest(folder / 'scores.npy') != result['scores_sha256']:
        raise ValueError('PCA-95 score integrity differs')
    if expected['method_id'] in CPU:
        if result.get('runtime_profile') != expected['runtime_evidence']['versions'][expected['method_id']]:
            raise ValueError('Recorded CPU runtime differs from frozen evidence')
    return result, np.load(folder / 'scores.npy', allow_pickle=False)


def worker(args):
    timing_guard(args.timing_proof)
    doc = json.loads(args.manifest.read_text())
    validate_manifest(doc)
    case = next(c for c in doc['cases'] if c['case_id'] == args.case_id)
    if args.method not in doc['methods'] or args.seed not in doc['seeds']:
        raise ValueError('Undeclared PCA-95 worker')
    inputs = args.output / 'inputs' / case['case_id']
    verify_input_binding(doc, case['case_id'], inputs / 'inputs.json')
    metadata = validate_inputs(inputs, case)
    expected = expected_identity(doc, case, args.method, args.seed)
    args.worker_output.mkdir(parents=True, exist_ok=True)
    if (args.worker_output / 'measurement.json').exists():
        raise FileExistsError('Preserve completed PCA-95 worker')
    runtime_profile = verify_runtime(doc, args.method) if args.method in CPU else None
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise ValueError('Requested CUDA is unavailable')
    started = time.perf_counter()
    if args.method in LOCAL:
        input_root = inputs / ('raw' if args.method in DYN else 'projected')
        train, test = [np.load(input_root / (name + '.npy'), allow_pickle=False) for name in ('train', 'test')]
        train_entities, test_entities = [np.load(inputs / (name + '_entities.npy'), allow_pickle=False)
                                         for name in ('train', 'test')]
        with threadpool_limits(limits=1):
            if args.method in DYN:
                model = dynsphere_model(args.method, args.seed, case['q']).fit(train, train_entities)
                scores = model.decision_function(test, test_entities)
            else:
                model = FittedControl(args.method, args.seed).fit(train, train_entities)
                scores = model.score(test, test_entities)
        np.save(args.worker_output / 'scores.npy', scores, allow_pickle=False)
    else:
        if args.method == 'omni_anomaly_tsb_ad':
            from benchmarks.official_baselines.adapters import omni_anomaly
            from research.submission_recovery.run import omni_backend
            calibration = np.load(inputs / 'projected/calibration.npy', allow_pickle=False)
            omni_anomaly.official_backend = lambda train, test, config, seed: omni_backend(
                train, calibration, test, config, seed)
        deep.worker(args.method, inputs / 'projected', args.worker_output, args.seed,
                    doc['configurations'][args.method], doc['upstream'][args.method])
    result = dict(expected, status='complete', input_sha256=metadata['sha256'],
                  scores_sha256=digest(args.worker_output / 'scores.npy'),
                  backend_seconds=time.perf_counter() - started, requested_device=args.device,
                  runtime_profile=runtime_profile,
                  versions={name: importlib.metadata.version(name)
                            for name in ('numpy', 'scipy', 'scikit-learn', 'pandas')}, python=sys.version)
    write_new_json(args.worker_output / 'measurement.json', result)


def environment(method, device, seed):
    from research.quality_cost.run import cpu_environment
    result = cpu_environment()
    result['PYTHONHASHSEED'] = str(seed)
    if device == 'cuda':
        result['DYNSPHERE_DEVICE'] = 'cuda'
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            result['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
        else:
            result.pop('CUDA_VISIBLE_DEVICES', None)
    if method == 'dagmm':
        from research.submission_recovery.run import OVERLAY
        result['PYTHONPATH'] = str(OVERLAY) + os.pathsep + str(ROOT)
    return result


def execute(args, doc, case, method, seed):
    inputs = args.output / 'inputs' / case['case_id']
    verify_input_binding(doc, case['case_id'], inputs / 'inputs.json')
    metadata = validate_inputs(inputs, case)
    folder = args.output / 'runs' / case['case_id'] / method / ('seed-' + str(seed))
    folder.mkdir(parents=True, exist_ok=True)
    expected = expected_identity(doc, case, method, seed)
    with run_lock(folder):
        target = folder / 'result.json'
        if target.exists():
            record = json.loads(target.read_text())
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError('Changed PCA-95 result identity')
            if record['status'] == 'complete':
                validate_measurement(folder / record['worker_subdirectory'], expected, metadata)
            return record
        completed = sorted(folder.glob('attempt-*/measurement.json'))
        attempt = completed[0].parent if completed else folder / ('attempt-%04d' % (len(list(folder.glob('attempt-*'))) + 1))
        attempt.mkdir(exist_ok=True)
        record = dict(expected, status='failed', worker_subdirectory=attempt.name)
        try:
            if not completed:
                executable = (ROOT / 'benchmarks/official_baselines/.envs' / method / 'bin/python'
                              if method in deep.EXTERNAL else Path(sys.executable))
                command = [str(executable), '-m', 'research.pca95.run', '--worker', '--output', str(args.output),
                           '--manifest', str(args.manifest), '--timing-proof', str(args.timing_proof),
                           '--case-id', case['case_id'], '--method', method, '--seed', str(seed),
                           '--device', args.device, '--worker-output', str(attempt)]
                with (attempt / 'stdout.log').open('x') as stdout, (attempt / 'stderr.log').open('x') as stderr:
                    finished = subprocess.run(command, cwd=ROOT, env=environment(method, args.device, seed),
                                              stdout=stdout, stderr=stderr, timeout=doc['worker_timeout_seconds'])
                if finished.returncode:
                    raise RuntimeError('Worker exit ' + str(finished.returncode))
            measurement, scores = validate_measurement(attempt, expected, metadata)
            labels = np.load(inputs / 'labels.npy', allow_pickle=False)
            if scores.shape != labels.shape or len(scores) <= 200 or not np.isfinite(scores).all():
                raise ValueError('Invalid or unaligned PCA-95 scores')
            y, s = labels[100:-100], scores[100:-100]
            if set(np.unique(y)) != {0, 1}:
                raise ValueError('Common support lacks both classes')
            record.update(measurement, pauc_fpr_05=float(roc_auc_score(y, s, max_fpr=.05)),
                          auprc=float(average_precision_score(y, s)), auroc=float(roc_auc_score(y, s)))
        except subprocess.TimeoutExpired as error:
            record.update(status='timeout', error=repr(error))
        except Exception as error:
            record.update(status='failed', error=repr(error))
        write_new_json(target, record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--group', choices=GROUPS, default='local')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=43200.)
    parser.add_argument('--timing-proof', type=Path, default=TIMING_PROOF)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--worker-output', type=Path)
    parser.add_argument('--case-id')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid PCA-95 shard selection')
    args.manifest = args.output / 'manifest.json'
    if args.manifest.is_file():
        doc = json.loads(args.manifest.read_text())
        validate_manifest(doc)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        doc = freeze(args.output, args.timeout_seconds)
    jobs = jobs_for(doc['cases'], GROUPS[args.group], doc['seeds'])[args.shard_index::args.shard_count]
    print(json.dumps(dict(manifest_id=doc['manifest_id'], planned_runs=len(jobs_for(doc['cases'])),
                          group=args.group, selected_runs=len(jobs), training_started=False)), flush=True)
    if args.dry_run:
        return
    timing_guard(args.timing_proof)
    doc = prepare_execution(args.output, doc, args.timing_proof)
    args.manifest = args.output / 'execution-manifest.json'
    print(json.dumps(dict(execution_manifest=doc['manifest_id'], training_started=not args.prepare_only)), flush=True)
    if args.prepare_only:
        return
    cases = {c['case_id']: c for c in doc['cases']}
    outcomes = []
    for job in jobs:
        record = execute(args, doc, cases[job['case_id']], job['method_id'], job['seed'])
        outcomes.append({k: record[k] for k in ('case_id', 'method_id', 'seed', 'status')})
        print(json.dumps(dict(done=len(outcomes), of=len(jobs), **outcomes[-1])), flush=True)
    path = args.output / f'completion-{args.group}-shard-{args.shard_index}.json'
    if not path.exists():
        write_new_json(path, dict(manifest_id=doc['manifest_id'], outcomes=outcomes, manuscript_ready=False))


if __name__ == '__main__':
    main()
