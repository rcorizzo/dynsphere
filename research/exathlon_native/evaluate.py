"""Independent cached metrics and coverage-aware application summaries."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from range_metrics import range_vus_scores
from research.evaluation_protocols.campaign import _identity, _json
from . import protocol as p
from .models import require_support
from .run import fit_identity, verify_artifacts

METRICS = ('pauc_fpr_05', 'auprc', 'auroc', 'vus_pr', 'vus_roc')


def pointwise(labels, scores, calibration):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    calibration = np.asarray(calibration, dtype=float)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid pointwise inputs')
    result = {'pauc_fpr_05': None, 'auprc': None, 'auroc': None,
              'evaluated_rows': len(labels), 'positive_rows': int(labels.sum()),
              'calibration_rows': len(calibration)}
    if set(np.unique(labels)) != {0, 1}:
        return result
    result.update(pauc_fpr_05=float(roc_auc_score(labels, scores, max_fpr=.05)),
                  auprc=float(average_precision_score(labels, scores)),
                  auroc=float(roc_auc_score(labels, scores)))
    if len(calibration):
        if not np.isfinite(calibration).all():
            raise ValueError('Invalid normal calibration scores')
        threshold = float(np.quantile(calibration, .95, method='higher'))
        pred = scores > threshold
        result.update(threshold_05=threshold, calibration_fpr_05=float(np.mean(calibration > threshold)),
                      fpr_05=float(pred[labels == 0].mean()), recall_05=float(pred[labels == 1].mean()))
    return result


def temporal_inputs(labels, scores, entities, mask):
    labels, scores, entities, mask = map(np.asarray, (labels, scores, entities, mask))
    require_support(scores, mask)
    ids = np.flatnonzero(mask)
    boundaries = np.flatnonzero((np.diff(ids) != 1) | (entities[ids[1:]] != entities[ids[:-1]]))+1
    parts = np.split(ids, boundaries)
    eligible, single = [], 0
    for part in parts:
        if not len(part):
            continue
        if set(np.unique(labels[part])) == {0, 1}:
            eligible.append({'indices': part, 'labels': labels[part], 'scores': scores[part]})
        else:
            single += len(part)
    return eligible, {'retained_rows': len(ids), 'vus_rows': sum(len(v['indices']) for v in eligible),
                      'vus_segments': len(eligible), 'single_class_rows': single}


def segment_vus(labels, scores, entities, mask, cache):
    if importlib.metadata.version('vus') != '0.0.6':
        raise ValueError('Official vus==0.0.6 is required')
    parts, coverage = temporal_inputs(labels, scores, entities, mask)
    rows = []
    for part in parts:
        y = np.asarray(part['labels'], dtype=np.int8)
        s = np.asarray(part['scores'], dtype='<f8')
        key = hashlib.sha256(b'official-vus-0.0.6-buffer100-segment-v1'+y.tobytes()+s.tobytes()).hexdigest()
        path = Path(cache)/(key+'.json')
        if path.exists():
            row = json.loads(path.read_text())
            if row.get('cache_key') != key or row.get('rows') != len(y):
                raise ValueError('Temporal cache identity differs')
        else:
            values = range_vus_scores(y, s, sliding_window=100, prefer_official=True)
            if values['vus_backend'] != 'official_vus' or not all(np.isfinite(values[m]) for m in ('vus_pr', 'vus_roc')):
                raise ValueError('Official temporal evaluation did not produce finite metrics')
            row = {'cache_key': key, 'rows': len(y), **values}
            _json(path, row)
        rows.append(row)
    segments = [{'entity': int(np.asarray(entities)[part['indices'][0]]),
                 'start': int(part['indices'][0]), 'stop': int(part['indices'][-1])+1,
                 'rows': len(part['indices']), 'cache_key': row['cache_key']}
                for part, row in zip(parts, rows)]
    return {**coverage, 'segments': segments,
            'vus_pr': float(np.average([r['vus_pr'] for r in rows], weights=[r['rows'] for r in rows])) if rows else None,
            'vus_roc': float(np.average([r['vus_roc'] for r in rows], weights=[r['rows'] for r in rows])) if rows else None,
            'vus_definition': 'official-0.0.6-segment-length-weighted-buffer100'}


def summarize(frame, seeds):
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    keys = ['method_id', 'application', 'trace']
    metrics = [m for m in METRICS if m in frame]
    records = []
    for key, group in frame.groupby(keys):
        if set(group.seed) != set(seeds) or len(group) != len(seeds) or group.pauc_fpr_05.isna().any():
            continue
        row = dict(zip(keys, key))
        for metric in metrics:
            row[metric] = float(group[metric].mean()) if group[metric].notna().all() else np.nan
        records.append(row)
    per_case = pd.DataFrame(records)
    if per_case.empty:
        return per_case, pd.DataFrame(), pd.DataFrame()
    per_app = per_case.groupby(['method_id', 'application'], as_index=False)[metrics].mean()
    counts = per_case.groupby(['method_id', 'application']).size().rename('execution_count').reset_index()
    per_app = per_app.merge(counts)
    for metric in metrics:
        counts = per_case.groupby(['method_id', 'application'])[metric].count().rename(metric+'_execution_count').reset_index()
        per_app = per_app.merge(counts)
    macro = per_app.groupby('method_id', as_index=False)[metrics].mean()
    macro = macro.merge(per_app.groupby('method_id').size().rename('application_count').reset_index())
    macro = macro.merge(per_case.groupby('method_id').size().rename('execution_count').reset_index())
    for metric in metrics:
        counts = per_app.groupby('method_id')[metric].count().rename(metric+'_application_count').reset_index()
        macro = macro.merge(counts)
        counts = per_case.groupby('method_id')[metric].count().rename(metric+'_execution_count').reset_index()
        macro = macro.merge(counts)
    return per_case, per_app, macro.sort_values('pauc_fpr_05', ascending=False)


def collect(output, doc, temporal=False, shard_index=0, shard_count=1):
    rows, inventory = [], []
    score_number = 0
    for group in doc['prepared']['cohort']:
        for method in doc['methods']:
            base = 'dynsphere_euclidean_one_regime' if method == p.REDUCED else method
            for seed in doc['seeds']:
                job = {'application': group['application'], 'method': base, 'seed': seed}
                folder = Path(output)/'fits'/fit_identity(doc, job)
                checkpoint = folder/'fit.json'
                record = json.loads(checkpoint.read_text()) if checkpoint.exists() else {'status': 'pending'}
                if record['status'] == 'complete':
                    verify_artifacts(record, folder, doc['manifest_id'], job)
                for entry in group['tests']:
                    identity = {'method_id': method, 'application': group['application'], 'trace': entry['trace'], 'seed': seed}
                    coverage = {**identity, 'status': record['status'], 'error': record.get('error')}
                    if record['status'] != 'complete':
                        inventory.append(coverage)
                        continue
                    matches = [r for r in record['scores'] if r['method_id'] == method and r['trace'] == entry['trace']]
                    if len(matches) != 1:
                        raise ValueError('Completed fit lacks exactly one expected score artifact')
                    path = folder/matches[0]['artifact']
                    with np.load(path, allow_pickle=False) as arrays:
                        labels, scores, mask = arrays['labels'], arrays['scores'], arrays['mask']
                        values = pointwise(labels[mask], scores[mask], arrays['calibration_scores'])
                        if temporal and score_number % shard_count == shard_index:
                            settings = {'artifact': record['artifacts'][matches[0]['artifact']], 'buffer': 100,
                                        'definition': 'official-0.0.6-segment-length-weighted'}
                            metric_path = Path(output)/'temporal_records'/(_identity(settings)+'.json')
                            if metric_path.exists():
                                saved = json.loads(metric_path.read_text())
                                if saved['settings'] != settings:
                                    raise ValueError('Temporal record settings differ')
                                temporal_values = saved['metrics']
                            else:
                                temporal_values = segment_vus(labels, scores, arrays['entities'], mask, Path(output)/'vus_cache')
                                _json(metric_path, {'settings': settings, 'metrics': temporal_values})
                            values.update(temporal_values)
                        else:
                            settings = {'artifact': record['artifacts'][matches[0]['artifact']], 'buffer': 100,
                                        'definition': 'official-0.0.6-segment-length-weighted'}
                            saved_path = Path(output)/'temporal_records'/(_identity(settings)+'.json')
                            if saved_path.exists():
                                values.update(json.loads(saved_path.read_text())['metrics'])
                        score_number += 1
                    coverage['status'] = 'complete' if values['pauc_fpr_05'] is not None else 'non-evaluable-support'
                    inventory.append(coverage)
                    rows.append({**identity, **values, 'score_artifact': str(path.relative_to(output))})
                    if temporal:
                        print(json.dumps({**identity, 'pauc_fpr_05': values['pauc_fpr_05'], 'vus_pr': values.get('vus_pr')}), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(inventory)


def export(output, frame, inventory, doc):
    directory = Path(output)/'summary'
    directory.mkdir(exist_ok=True)
    if frame.empty:
        frame = frame.reindex(columns=['method_id', 'application', 'trace', 'seed', *METRICS])
    frame.to_csv(directory/'per_run_available.csv', index=False)
    inventory.to_csv(directory/'completion.csv', index=False)
    available = summarize(frame, doc['seeds'])
    per_case = available[0]
    if per_case.empty:
        matched = frame.iloc[:0]
    else:
        counts = per_case.groupby(['application', 'trace']).method_id.nunique()
        eligible = set(counts[counts == len(doc['methods'])].index)
        matched = frame[[tuple(x) in eligible for x in frame[['application', 'trace']].to_numpy()]]
    temporal_status = {}
    matched = matched.copy()
    for metric in ('vus_pr', 'vus_roc'):
        complete = bool(len(matched)) and metric in matched and matched[metric].notna().all()
        temporal_status[metric] = 'same-cohort-complete' if complete else 'matched-aggregate-suppressed-incomplete-temporal-coverage'
        if not complete:
            matched[metric] = np.nan
    for label, source in [('available', frame), ('matched', matched)]:
        cases, apps, macro = summarize(source, doc['seeds'])
        for name, table in [('per_execution', cases), ('per_application', apps), ('equal_application', macro)]:
            table.to_csv(directory/(name+'_'+label+'.csv'), index=False)
        if not cases.empty:
            cases.groupby('method_id', as_index=False)[[m for m in METRICS if m in cases]].mean().to_csv(
                directory/('equal_execution_'+label+'.csv'), index=False)
            for label2, selected in [('pilot_applications', [1, 2]), ('additional_applications', [4, 5])]:
                part = source[source.application.isin(selected)]
                summarize(part, doc['seeds'])[2].to_csv(directory/(label2+'_'+label+'.csv'), index=False)
    status = inventory.status.value_counts().to_dict()
    report = {'manifest_id': doc['manifest_id'], 'expected_score_records': 150,
              'outcomes': {k: int(v) for k, v in status.items()},
              'vus_records': int(frame.vus_pr.notna().sum()) if 'vus_pr' in frame else 0,
              'matched_temporal_status': temporal_status,
              'matched_executions': int(matched[['application', 'trace']].drop_duplicates().shape[0]),
              'matched_applications': int(matched.application.nunique()) if len(matched) else 0,
              'weighting': 'three seeds, then executions within application, then equal applications',
              'manuscript_ready': False}
    _json(directory/'readiness.json', report)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=p.OUTPUT)
    parser.add_argument('--vus', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--metrics-only', action='store_true')
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('Invalid metric shard')
    doc = p.validate(args.output)
    frame, inventory = collect(args.output, doc, args.vus, args.shard_index, args.shard_count)
    if not args.metrics_only:
        export(args.output, frame, inventory, doc)


if __name__ == '__main__':
    main()
