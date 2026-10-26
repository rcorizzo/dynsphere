"""Paired accuracy sensitivity summaries; incomplete support remains explicit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run import (FAMILIES, METHODS, OUTPUT, ROOT, digest, expected_identity,
                  jobs_for, validate_inputs, validate_manifest, validate_measurement,
                  verify_input_binding, write_new_json)


def paired_summary(variants, baseline, primary_cases, seeds,
                   requested_metrics=('pauc_fpr_05', 'auprc', 'auroc')):
    keys = ['case_id', 'source_family', 'method_id', 'seed']
    for frame in (variants, baseline):
        if frame.duplicated(keys).any():
            raise ValueError('Duplicate sensitivity comparison records')
    complete = variants[variants.status.eq('complete') & variants.case_id.isin(primary_cases)]
    metrics = list(requested_metrics)
    optional = ['vus_status'] if 'vus_status' in complete else []
    pairs = complete.reindex(columns=keys + metrics + optional).merge(
              baseline.reindex(columns=keys + metrics), on=keys, how='inner',
                                            validate='one_to_one', suffixes=('_variant', '_original'))
    for metric in metrics:
        pairs['delta_' + metric] = pairs[metric + '_variant'] - pairs[metric + '_original']
    rows = []
    expected = {(case, seed) for case in primary_cases for seed in seeds}
    for method in sorted(set(variants.method_id)):
        current = pairs[pairs.method_id.eq(method)]
        ready = (set(zip(current.case_id, current.seed)) == expected
                 and set(current.source_family) == set(FAMILIES))
        row = dict(method_id=method, paired_runs=len(current), required_runs=len(expected),
                   primary_complete=ready, families=current.source_family.nunique())
        for metric in metrics:
            finite = np.isfinite(current[[metric + '_variant', metric + '_original']]).all(axis=1)
            if metric.startswith('vus_'):
                finite &= current.get('vus_status', pd.Series('failed', index=current.index)).eq('complete')
            row['paired_' + metric + '_runs'] = int(finite.sum())
            valid_metric = ready and bool(finite.all())
            row['complete_' + metric] = valid_metric
            for suffix in ('variant', 'original'):
                column = metric + '_' + suffix
                row['equal_family_' + column] = (float(current.groupby(['source_family', 'case_id'])[column]
                                                        .mean().groupby('source_family').mean().mean())
                                                if valid_metric else np.nan)
            row['equal_family_' + metric] = row['equal_family_' + metric + '_variant']
            row['equal_family_delta_' + metric] = (row['equal_family_' + metric + '_variant']
                                                   - row['equal_family_' + metric + '_original'])
        row['pointwise_complete'] = ready and all(row.get('complete_' + m, False)
                                                  for m in ('pauc_fpr_05', 'auprc', 'auroc'))
        row['primary_complete'] = ready and all(row['complete_' + m] for m in requested_metrics)
        rows.append(row)
    return pairs, pd.DataFrame(rows)


def cached_vus(output, labels, scores):
    from range_metrics import range_vus_scores
    version = importlib.metadata.version('vus')
    if version != '0.0.6':
        raise ValueError('The sensitivity VUS comparison requires vus==0.0.6')
    settings = dict(backend='official_vus', vus_version=version, numpy_version=np.__version__,
                    sklearn_version=importlib.metadata.version('scikit-learn'),
                    range_metrics_sha256=digest(ROOT / 'range_metrics.py'),
                    analysis_sha256=digest(Path(__file__)), support='100/100',
                    window='rounded-median-anomaly-segment-length-on-common-support')
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for array in (np.asarray(labels, dtype='<i8'), np.asarray(scores, dtype='<f8')):
        identity.update(str(array.shape).encode())
        identity.update(array.tobytes())
    path = output / 'vus-cache' / (identity.hexdigest() + '.json')
    if path.is_file():
        result = json.loads(path.read_text())
        if result['settings'] != settings:
            raise ValueError('Changed VUS cache settings')
        return result
    result = dict(settings=settings, metric_key=path.stem, status='failed')
    try:
        values = range_vus_scores(labels, scores, prefer_official=True)
        if (values.get('vus_backend') != 'official_vus'
                or not np.isfinite([values.get('vus_pr'), values.get('vus_roc')]).all()):
            raise ValueError('Official VUS unavailable: ' + repr(values))
        result.update(values, status='complete')
    except Exception as error:
        result['error'] = repr(error)
    write_new_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--vus', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    manifest = args.output / 'execution-manifest.json'
    if not manifest.exists():
        manifest = args.output / 'manifest.json'
    doc = json.loads(manifest.read_text())
    validate_manifest(doc)
    reference_path = ROOT / doc['reference_path']
    if digest(reference_path) != doc['reference_sha256']:
        raise ValueError('Changed original comparison reference')
    baseline = pd.read_csv(reference_path)
    cases = {c['case_id']: c for c in doc['cases']}
    records = []
    for path in sorted((args.output / 'runs').glob('*/*/seed-*/result.json')):
        record = json.loads(path.read_text())
        case = cases[record['case_id']]
        expected = expected_identity(doc, case, record['method_id'], record['seed'])
        if (record['method_id'] not in METHODS or record['seed'] not in doc['seeds']
                or any(record.get(k) != v for k, v in expected.items())):
            raise ValueError('Unexpected PCA-95 aggregate record')
        if record['status'] == 'complete':
            inputs = args.output / 'inputs' / case['case_id']
            verify_input_binding(doc, case['case_id'], inputs / 'inputs.json')
            metadata = validate_inputs(inputs, case)
            _, scores = validate_measurement(path.parent / record['worker_subdirectory'], expected, metadata)
            labels = np.load(inputs / 'labels.npy', allow_pickle=False)
            if scores.shape != labels.shape or not np.isfinite(scores).all() or len(scores) <= 200:
                raise ValueError('Invalid sensitivity metric support')
            from sklearn.metrics import average_precision_score, roc_auc_score
            y, s = labels[100:-100], scores[100:-100]
            actual = [roc_auc_score(y, s, max_fpr=.05), average_precision_score(y, s), roc_auc_score(y, s)]
            np.testing.assert_allclose(actual, [record[k] for k in ('pauc_fpr_05', 'auprc', 'auroc')],
                                       rtol=1e-11, atol=1e-11)
            if args.vus:
                temporal = cached_vus(args.output, labels[100:-100], scores[100:-100])
                record['vus_status'] = temporal['status']
                for name in ('vus_pr', 'vus_roc', 'vus_backend', 'vus_window', 'error'):
                    if name in temporal:
                        record['vus_' + name if name == 'error' else name] = temporal[name]
        records.append(record)
    columns = ['case_id', 'source_family', 'method_id', 'seed', 'status',
               'pauc_fpr_05', 'auprc', 'auroc']
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    for name in columns:
        if name not in frame:
            frame[name] = np.nan
    requested = ('pauc_fpr_05', 'auprc', 'auroc') + (('vus_pr', 'vus_roc') if args.vus else ())
    pairs, summary = paired_summary(frame, baseline, set(doc['primary_case_ids']), doc['seeds'], requested)
    destination = args.output / ('summary-vus' if args.vus else 'summary-pointwise')
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / 'per_run.csv', index=False)
    pairs.to_csv(destination / 'paired.csv', index=False)
    summary.to_csv(destination / 'primary_paired_summary.csv', index=False)
    planned = len(jobs_for(doc['cases'], doc['methods'], doc['seeds']))
    complete = int(frame.status.eq('complete').sum())
    vus_complete = int(frame.get('vus_status', pd.Series(dtype=str)).eq('complete').sum())
    status = dict(manifest_id=doc['manifest_id'], expected=planned, observed=len(frame),
                  complete=complete, noncomplete=len(frame) - complete, missing=planned - len(frame),
                  vus_complete=vus_complete, vus_requested=args.vus,
                  requested_metrics=requested,
                  manuscript_ready=False, exploratory=True,
                  model_runs_complete=len(frame) == planned,
                  full_primary_comparison=bool(len(summary) == len(METHODS) and summary.primary_complete.all()))
    # Derived status can be regenerated; immutable model and metric records are preserved.
    (destination / 'completion.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
