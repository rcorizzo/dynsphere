"""Frozen campaign preparation, isolated fitting, and resumable execution."""
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
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from research.evaluation_protocols.campaign import ROOT, FittedControl, _identity, _json, digest
from research.submission_campaign.studies import grouped_scores
from . import data, protocol as p
from .models import DeepModel, require_support


def describe(doc):
    groups = doc['prepared']['cohort']
    seeds = len(doc['seeds'])
    return {'manifest_id': doc['manifest_id'], 'applications': len(groups),
            'test_executions': sum(len(g['tests']) for g in groups),
            'planned_fits': len(p.jobs(doc)), 'local_fits': len(p.jobs(doc, 'local')),
            'deep_fits': len(p.jobs(doc, 'deep')),
            'planned_score_records': sum(len(g['tests']) for g in groups)*seeds*len(p.METHODS),
            'seeds': doc['seeds'], 'manuscript_ready': False}


def verify_artifacts(record, directory, manifest_id, job):
    if (record.get('status') != 'complete' or record.get('manifest_id') != manifest_id
            or record.get('job') != job or not record.get('artifacts')):
        raise ValueError('Saved fit identity differs')
    for name, expected in record['artifacts'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or digest(Path(directory)/relative) != expected:
            raise ValueError('Saved score artifact integrity differs')
    return record


@contextmanager
def writer_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.writer.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def fit_identity(doc, job):
    return _identity({'manifest_id': doc['manifest_id'], 'job': job})


def record_worker_exit(output, doc, job, status):
    directory = Path(output)/'fits'/fit_identity(doc, job)
    path = directory/'fit.json'
    with writer_lock(directory):
        existing = json.loads(path.read_text()) if path.exists() else {}
        if existing.get('status') == 'complete':
            verify_artifacts(existing, directory, doc['manifest_id'], job)
            return
        if 'timeout_seconds' in status or status.get('returncode', 0):
            record = {'manifest_id': doc['manifest_id'], 'job': job,
                      'status': 'timeout' if 'timeout_seconds' in status else 'failed',
                      'error': str(status), 'artifacts': {}, 'scores': []}
            log = directory/'worker_exits'/(uuid.uuid4().hex+'.json')
            _json(log, record)
            if not existing or record['status'] == 'timeout':
                _json(path, record)


def require_compatibility(output):
    from research.submission_campaign.deep import upstream_snapshot, declared_revisions
    evidence = p.code_evidence()
    revisions = declared_revisions()
    for method in p.DEEP:
        candidates = list((Path(output)/'compatibility'/method).glob('*.json'))
        upstream = upstream_snapshot(ROOT/'benchmarks/official_baselines/.external'/method, revisions[method])
        expected_cfg = {**p.configuration()[method], 'epochs': 2}
        records = [json.loads(f.read_text()) for f in candidates]
        if not any(r.get('implementation') == evidence and r.get('status') == 'passed'
                   and r.get('upstream') == upstream and r.get('configuration') == expected_cfg for r in records):
            raise ValueError('Missing source-matched synthetic compatibility check: '+method)


def execute_fit(output, doc, job):
    output = Path(output)
    identity = fit_identity(doc, job)
    directory = output/'fits'/identity
    group = next(g for g in doc['prepared']['cohort'] if g['application'] == job['application'])
    with writer_lock(directory):
        checkpoint = directory/'fit.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get('status') == 'complete':
                return verify_artifacts(saved, directory, doc['manifest_id'], job)
        attempt = directory/'attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True)
        record = {'manifest_id': doc['manifest_id'], 'fit_id': identity, 'job': job,
                  'status': 'running', 'attempt': str(attempt.relative_to(directory)),
                  'artifacts': {}, 'scores': [], 'versions': runtime_versions(), 'manuscript_ready': False}
        started = time.perf_counter()
        try:
            train = data.load_input(output, job['application'], 'fit')
            cal = data.load_input(output, job['application'], 'calibration')
            with threadpool_limits(limits=1):
                if job['method'] in p.DEEP:
                    model = DeepModel(job['method'], doc['deep_configuration'][job['method']], job['seed'])
                    model.fit(train['values'], train['entities'], cal['values'], cal['entities'])
                    record['backend'] = model.metadata
                else:
                    model = FittedControl(job['method'], job['seed']).fit(train['values'], train['entities'])
                record['fit_seconds'] = time.perf_counter()-started
                cal_scores = model.score(cal['values'], cal['entities'])
                require_support(cal_scores, cal['mask'])
                reduced_cal = None
                if job['method'] == 'dynsphere_euclidean_one_regime':
                    groups = grouped_scores(model.model.decision_components(cal['values'], cal['entities']), cal['entities'], 1)
                    if not np.allclose(groups['combined_r8'], cal_scores, atol=1e-10, rtol=1e-10):
                        raise ValueError('Reduced calibration reference differs from E1')
                    reduced_cal = groups['motion_r8']
                for entry in group['tests']:
                    test = data.load_input(output, job['application'], entry['trace'])
                    scores = model.score(test['values'], test['entities'])
                    candidates = [(job['method'], scores, cal_scores)]
                    if reduced_cal is not None:
                        groups = grouped_scores(model.model.decision_components(test['values'], test['entities']), test['entities'], 1)
                        if not np.allclose(groups['combined_r8'], scores, atol=1e-10, rtol=1e-10):
                            raise ValueError('Reduced test reference differs from E1')
                        candidates.append((p.REDUCED, groups['motion_r8'], reduced_cal))
                    for method, values, calibration in candidates:
                        require_support(values, test['mask'])
                        path = attempt/(entry['trace']+'--'+method+'.npz')
                        temporary = path.with_suffix('.tmp')
                        with temporary.open('wb') as stream:
                            np.savez_compressed(stream, scores=values,
                                calibration_scores=calibration[cal['mask']], labels=test['labels'],
                                mask=test['mask'], entities=test['entities'])
                        temporary.replace(path)
                        relative = str(path.relative_to(directory))
                        record['artifacts'][relative] = digest(path)
                        record['scores'].append({'method_id': method, 'trace': entry['trace'], 'artifact': relative})
            record.update(status='complete', elapsed_seconds=time.perf_counter()-started)
        except Exception as exc:
            import traceback
            record.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                          elapsed_seconds=time.perf_counter()-started)
        _json(attempt/'fit.json', record)
        _json(checkpoint, record)
        print(json.dumps({'job': job, 'status': record['status'], 'seconds': record['elapsed_seconds'],
                          'error': record.get('error')}), flush=True)
        return record


def runtime_versions():
    result = {'python': sys.version.split()[0]}
    for name in ('numpy', 'pandas', 'scikit-learn', 'torch', 'vus'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def synthetic_inputs():
    rng = np.random.default_rng(123)
    return rng.normal(size=(1660, 19)), np.repeat([0, 1], 830), rng.normal(size=(390, 19)), np.repeat([0, 1, 2], 130)


def compatibility(output, method):
    from research.submission_campaign.deep import upstream_snapshot, declared_revisions
    started = time.perf_counter()
    cfg = {**p.configuration()[method], 'epochs': 2}
    train, entities, cal, ce = synthetic_inputs()
    model = DeepModel(method, cfg, 37).fit(train, entities, cal, ce)
    values = model.score(train[:520], np.repeat([0, 1], 260))
    require_support(values, data.support(np.repeat([0, 1], 260)))
    expected = 1462 if method == 'catch' else 1460
    if model.metadata['fit_windows'] != expected:
        raise ValueError('Synthetic fit windows crossed segment boundaries')
    if method == 'catch' and model.metadata['validation_windows'] != 93:
        raise ValueError('Synthetic independent validation windows differ')
    record = {'status': 'passed', 'method': method, 'implementation': p.code_evidence(),
              'configuration': cfg, 'versions': runtime_versions(), 'seconds': time.perf_counter()-started,
              'train_rows': len(train), 'calibration_rows': len(cal), 'backend': model.metadata,
              'upstream': upstream_snapshot(ROOT/'benchmarks/official_baselines/.external'/method,
                                            declared_revisions()[method]),
              'interpretation': 'synthetic interface check; not a benchmark result or full-size timing estimate'}
    p.immutable_json(Path(output)/'compatibility'/method/(_identity(record)+'.json'), record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('implementation', 'upstream')}), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=p.OUTPUT)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--tree', type=Path, required=True)
    sub.add_parser('freeze')
    sub.add_parser('dry-run')
    for name in ('local', 'deep'):
        child = sub.add_parser(name)
        child.add_argument('--shard-index', type=int, default=0)
        child.add_argument('--shard-count', type=int, default=1)
        child.add_argument('--limit', type=int, default=0)
    worker = sub.add_parser('deep-worker')
    worker.add_argument('--application', type=int, required=True)
    worker.add_argument('--method', choices=p.DEEP, required=True)
    worker.add_argument('--seed', type=int, choices=p.SEEDS, required=True)
    compat = sub.add_parser('compatibility')
    compat.add_argument('--method', choices=p.DEEP, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        return data.prepare(json.loads(args.tree.read_text()), args.output)
    if args.command == 'compatibility':
        return compatibility(args.output, args.method)
    if args.command == 'freeze':
        require_compatibility(args.output)
        doc = p.freeze(args.output)
    else:
        doc = p.validate(args.output, upstream=args.command in ('deep', 'deep-worker'))
    print(json.dumps(describe(doc)), flush=True)
    if args.command in ('freeze', 'dry-run'):
        return doc
    require_compatibility(args.output)
    if args.command == 'deep-worker':
        job = {'application': args.application, 'method': args.method, 'seed': args.seed}
        if job not in p.jobs(doc, 'deep'):
            raise ValueError('Worker job is outside frozen campaign')
        record = execute_fit(args.output, doc, job)
        if record['status'] != 'complete':
            raise RuntimeError(record.get('error', 'Deep worker incomplete'))
        return record
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid shard indices')
    selected = p.jobs(doc, args.command)[args.shard_index::args.shard_count]
    if args.limit:
        selected = selected[:args.limit]
    failures = 0
    for job in selected:
        if args.command == 'local':
            failures += execute_fit(args.output, doc, job)['status'] != 'complete'
        else:
            python = ROOT/'benchmarks/official_baselines/.envs'/job['method']/'bin/python'
            logs = args.output/'execution_logs'/uuid.uuid4().hex
            logs.mkdir(parents=True)
            command = [str(python), '-m', 'research.exathlon_native.run', '--output', str(args.output),
                       'deep-worker', '--application', str(job['application']),
                       '--method', job['method'], '--seed', str(job['seed'])]
            environment = {**os.environ, 'PYTHONHASHSEED': str(job['seed']), 'OMP_NUM_THREADS': '1',
                           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'LOKY_MAX_CPU_COUNT': '1'}
            with (logs/'stdout.log').open('w') as stdout, (logs/'stderr.log').open('w') as stderr:
                try:
                    process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, timeout=86400)
                    status = {'returncode': process.returncode}
                except subprocess.TimeoutExpired:
                    status = {'timeout_seconds': 86400}
            _json(logs/'execution.json', {'job': job, 'manifest_id': doc['manifest_id'], **status})
            record_worker_exit(args.output, doc, job, status)
            failures += 'timeout_seconds' in status or status.get('returncode', 0) != 0
            print(json.dumps({'job': job, 'logs': str(logs), **status}), flush=True)
    if failures:
        raise RuntimeError(f'{failures} frozen model fits did not complete; see preserved fit records')
    return describe(doc)


if __name__ == '__main__':
    main()
