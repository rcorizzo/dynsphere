"""Evaluate immutable extension scores without fitting or loading detectors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.official_baselines.metrics import compute_metrics


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def load_scores(row, runs_root):
    """Resolve transferred runs and verify frozen identity and pointwise metrics."""
    relative = str(row['run_directory']).rsplit('/runs/', 1)
    if len(relative) != 2:
        raise ValueError('Run directory must contain a /runs/ component')
    root = Path(runs_root).resolve()
    directory = (root / relative[1]).resolve()
    if root not in directory.parents:
        raise ValueError('Run directory escapes the supplied runs root')
    record = json.loads((directory / 'run.json').read_text())
    for field in ('run_identity', 'case_id', 'method_id', 'seed', 'source_family'):
        if str(record[field]) != str(row[field]):
            raise ValueError(f'Frozen run identity differs: {field}')
    if record['status'] != 'complete' or int(row['evaluation_warmup_rows']) != 1:
        raise ValueError('A completed run and the common one-row exclusion are required')
    labels = np.load(directory / 'case/private/labels.npy', allow_pickle=False)
    scores = np.load(directory / 'execution/scores.npy', allow_pickle=False)
    if (labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores)
            or not np.isfinite(scores).all() or not set(np.unique(labels)) <= {0, 1}):
        raise ValueError('Invalid or unaligned cached labels/scores')
    labels, scores = labels[1:].astype(np.int8), scores[1:].astype(float)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Both labels are required on the common evaluation support')
    actual = (average_precision_score(labels, scores), roc_auc_score(labels, scores))
    if not np.allclose(actual, [float(row['auprc']), float(row['auroc'])], atol=1e-10, rtol=1e-10):
        raise ValueError('Cached pointwise metrics differ from the frozen cohort')
    return labels, scores


def settings():
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(__file__).with_name('metrics.py'), root / 'range_metrics.py']
    return {'backend': 'official_vus', 'vus_version': importlib.metadata.version('vus'),
            'numpy_version': np.__version__,
            'sklearn_version': importlib.metadata.version('scikit-learn'),
            'event_budget': 0.01, 'vus_window': None,
            'support': 'suffix-only-first-row-excluded-v1',
            'metric_source_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()}


def metric_key(labels, scores, configuration):
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for value in (np.asarray(labels, dtype='<i8'), np.asarray(scores, dtype='<f8')):
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def evaluate(labels, scores, cache_dir):
    configuration = settings()
    key = metric_key(labels, scores, configuration)
    path = Path(cache_dir) / f'{key}.json'
    if path.exists():
        document = json.loads(path.read_text())
        if (document.get('metric_key') == key and document.get('status') == 'complete'
                and document.get('settings') == configuration):
            return {**document, 'reused': True}
    started = time.perf_counter()
    metrics = compute_metrics(labels, scores, configuration['event_budget'], None)
    if metrics['vus_backend'] != 'official_vus':
        raise ValueError('Official VUS evaluation is required; fallback output is rejected')
    if not all(np.isfinite(metrics[name]) for name in ('vus_pr', 'vus_roc')):
        raise ValueError('Official VUS returned nonfinite metrics')
    document = {**metrics, 'status': 'complete', 'metric_key': key, 'settings': configuration,
                'elapsed_seconds': time.perf_counter() - started,
                'manuscript_ready': False, 'reused': False}
    _write(path, document)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--families', nargs='+')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error('Invalid shard index/count')
    frame = pd.read_csv(args.input_csv)
    if frame['run_identity'].duplicated().any():
        raise ValueError('Duplicate run identities in input table')
    if args.families:
        frame = frame[frame.source_family.isin(args.families)]
        if set(args.families) - set(frame.source_family):
            raise ValueError('Requested family missing from input table')
    configuration = settings()
    groups = {}
    seal = {'input_csv_sha256': hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
            'run_inputs': {}}
    for row in frame.to_dict('records'):
        labels, scores = load_scores(row, args.runs_root)
        key = metric_key(labels, scores, configuration)
        groups.setdefault(key, []).append(row)
        seal['run_inputs'][row['run_identity']] = metric_key(
            labels, scores, {'support': 'suffix-only-first-row-excluded-v1'})
    seal_path = args.output_dir / 'input-seal.json'
    if seal_path.exists():
        if json.loads(seal_path.read_text()) != seal:
            raise ValueError('Input seal differs from the frozen score/label cohort')
    elif not args.dry_run:
        # The original table contains no score hashes. This initial seal protects
        # subsequent evaluation; it cannot establish earlier historical integrity.
        _write(seal_path, seal)
    keys = sorted(groups)[args.shard_index::args.shard_count]
    summary = {'selected_runs': len(frame), 'unique_metric_inputs': len(groups),
               'shard_metric_inputs': len(keys), 'shard_index': args.shard_index,
               'shard_count': args.shard_count, 'settings': configuration,
               'input_csv_sha256': hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
               'manuscript_ready': False}
    if args.dry_run:
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    rows, errors = [], []
    for index, key in enumerate(keys):
        try:
            labels, scores = load_scores(groups[key][0], args.runs_root)
            result = evaluate(labels, scores, args.output_dir / 'cache')
            for row in groups[key]:
                rows.append({**row, **{k: v for k, v in result.items()
                                       if k not in ('settings', 'elapsed_seconds')},
                             'metric_elapsed_seconds': result['elapsed_seconds']})
            print(json.dumps({'metric_input': index + 1, 'of': len(keys),
                              'family': groups[key][0]['source_family'],
                              'method': groups[key][0]['method_id'],
                              'reused': result['reused'], 'seconds': result['elapsed_seconds']}), flush=True)
        except Exception as exc:
            errors.append({'metric_key': key, 'error': f'{type(exc).__name__}: {exc}'})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_dir / f'metrics-shard-{args.shard_index:03d}.csv', index=False)
        _write(args.output_dir / f'completion-shard-{args.shard_index:03d}.json',
               {**summary, 'completed_runs': len(rows), 'errors': errors,
                'status': 'running', 'processed_metric_inputs': index + 1})
    summary.update(completed_runs=len(rows), errors=errors,
                   status='failed' if errors else 'complete')
    _write(args.output_dir / f'completion-shard-{args.shard_index:03d}.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    raise SystemExit(1 if main().get('status') == 'failed' else 0)
