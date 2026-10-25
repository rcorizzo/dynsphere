"""Run the bounded local campaign without changing prior scores or paper sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign import (ROOT, METHODS, SEEDS, _json, digest, freeze, prepare_nasa,
                       prepare_power, prepare_principal, run_one)


OUTPUT = ROOT / 'results/protocol_validation_20261024'
HISTORICAL = ROOT / 'results/official_baselines_v1/aggregate/f87ae4a73384/per_run.csv'
POWER = ROOT / 'benchmarks/real_world_datasets/data/power_system/binary'
METRICS = ['auprc', 'auroc', 'pauc_fpr_01', 'pauc_fpr_05', 'fpr_01', 'fpr_05',
           'recall_01', 'recall_05', 'event_recall_01', 'event_recall_05']


def entries():
    from benchmarks.official_baselines.datasets import REAL_WORLD_SPECS
    old = pd.read_csv(HISTORICAL)
    names = sorted(old[old.source_family.isin(['MSL', 'SMAP', 'SMD'])].trajectory_id.unique())
    available = {}
    for path in (ROOT / 'benchmarks/tsb_ad_m/data/TSB-AD-M').rglob('*.csv'):
        if path.stem in names:
            if path.stem in available:
                raise ValueError('Ambiguous NASA source filename')
            available[path.stem] = path
    if set(names) != set(available):
        raise ValueError('Missing frozen NASA source files')
    result = [{'case_id': name, 'source_family': name.split('_id_')[0].split('_', 1)[1],
               'kind': 'nasa', 'paths': [str(available[name])]} for name in names]
    for family in ('BATADAL', 'HAI-20.07'):
        spec = REAL_WORLD_SPECS[family]
        result.append({'case_id': family, 'source_family': family, 'kind': 'principal',
                       'paths': [str(spec['train']), str(spec['test'])]})
    result.append({'case_id': 'PowerSystem-source-grouped', 'source_family': 'PowerSystem',
                   'kind': 'power', 'paths': [str(POWER / f'data{i}.csv') for i in range(1, 16)]})
    return result


def historical_low_fpr(output):
    old = pd.read_csv(HISTORICAL)
    keys = ['dataset_id', 'trajectory_id', 'seed']
    counts = old.groupby(keys).method_id.nunique()
    common = old.merge(counts[counts == old.method_id.nunique()].reset_index()[keys], on=keys)
    case = common.groupby(['source_family', 'case_id', 'method_id'], as_index=False).pauc_fpr_05.mean()
    family = case.groupby(['source_family', 'method_id'], as_index=False).pauc_fpr_05.mean()
    family['rank'] = family.groupby('source_family').pauc_fpr_05.rank(ascending=False, method='min')
    family.to_csv(output / 'historical_pauc_fpr_05.csv', index=False)
    extension_path = ROOT / 'results/extension_audit_20261023/new_methods_vus/per_run_matched.csv'
    ext = pd.read_csv(extension_path)
    ext = ext.groupby(['source_family', 'case_id', 'method_id'], as_index=False).pauc_fpr_05.mean()
    ext = ext.groupby(['source_family', 'method_id'], as_index=False).pauc_fpr_05.mean()
    ext['rank'] = ext.groupby('source_family').pauc_fpr_05.rank(ascending=False, method='min')
    ext.to_csv(output / 'extension_pauc_fpr_05.csv', index=False)


def summarize(records, output, manifest):
    complete = [r for r in records if r['status'] == 'complete']
    incomplete = [{k: r[k] for k in r if k != 'metadata'} for r in records if r['status'] != 'complete']
    _json(output / 'noncomplete.json', incomplete)
    summary = {'manifest_id': manifest['manifest_id'], 'planned_records': len(records),
               'complete': len(complete), 'incompatible': sum(r['status'] == 'incompatible' for r in records),
               'failed': sum(r['status'] == 'failed' for r in records), 'manuscript_ready': False,
               'scope': 'local controls only; deep matched reruns pending',
               'historical_raw_scores': 'Latest paper-table run identities not located locally; no substitution',
               'matched_records': 0}
    # Completion/failure evidence is durable even when subsequent aggregation fails.
    _json(output / 'completion.json', summary)
    rows = [{**{k: r[k] for k in ('run_identity', 'case_id', 'source_family', 'method_id', 'seed')},
             **r['metrics'], 'protocol': r['metadata']['protocol'],
             'artifact_sha256': r['artifact_sha256'], 'run_directory': r['run_directory']}
            for r in complete]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'per_run.csv', index=False)
    if len(frame):
        if frame.duplicated(['case_id', 'method_id', 'seed']).any():
            raise ValueError('Duplicate case/method/seed records')
        available_case = frame.groupby(['source_family', 'case_id', 'method_id'], as_index=False)[METRICS].mean()
        available = available_case.groupby(['source_family', 'method_id'], as_index=False)[METRICS].mean()
        counts = available_case.groupby(['source_family', 'method_id']).case_id.nunique().rename('case_count')
        available = available.merge(counts, on=['source_family', 'method_id'])
        available['support'] = 'available-successful-records; unmatched descriptive summary'
        available.to_csv(output / 'per_family_available.csv', index=False)
        # Match seeds before case averaging so differing seed successes cannot change support.
        counts = frame.groupby(['case_id', 'seed']).method_id.nunique()
        support = counts[counts == len(METHODS)].reset_index()[['case_id', 'seed']]
        matched = frame.merge(support, on=['case_id', 'seed'], validate='many_to_one')
        matched.to_csv(output / 'per_run_matched.csv', index=False)
        summary['matched_records'] = len(matched)
        case = matched.groupby(['source_family', 'case_id', 'method_id'], as_index=False)[METRICS].mean()
        family = case.groupby(['source_family', 'method_id'], as_index=False)[METRICS].mean()
        family['case_count'] = family.source_family.map(case.groupby('source_family').case_id.nunique())
        for name in ('auprc', 'pauc_fpr_05'):
            family[name + '_rank'] = family.groupby('source_family')[name].rank(ascending=False, method='min')
        case.to_csv(output / 'per_case.csv', index=False)
        family.to_csv(output / 'per_family.csv', index=False)
        paired = []
        rng = np.random.default_rng(20261024)
        for source in ('MSL', 'SMAP', 'SMD'):
            sub = case[case.source_family == source]
            for metric in ('auprc', 'pauc_fpr_05'):
                pivot = sub.pivot(index='case_id', columns='method_id', values=metric)
                if 'dynsphere_euclidean_one_regime' not in pivot:
                    continue
                for method in sorted(set(pivot) - {'dynsphere_euclidean_one_regime'}):
                    difference = (pivot.dynsphere_euclidean_one_regime - pivot[method]).dropna().to_numpy()
                    if not len(difference):
                        continue
                    means = difference[rng.integers(0, len(difference), size=(10000, len(difference)))].mean(axis=1)
                    lo, hi = np.quantile(means, [.025, .975])
                    paired.append({'source_family': source, 'metric': metric, 'comparison': 'E1 minus '+method,
                                   'case_count': len(difference), 'difference': float(difference.mean()),
                                   'ci_low': float(lo), 'ci_high': float(hi),
                                   'interval': 'descriptive paired trajectory bootstrap; no multiplicity correction'})
        pd.DataFrame(paired).to_csv(output / 'paired_bootstrap.csv', index=False)
    _json(output / 'completion.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    selected = entries()
    if args.pilot:
        selected = [min((e for e in selected if e['source_family'] == family), key=lambda e: e['case_id'])
                    for family in ('MSL', 'SMAP', 'SMD', 'BATADAL', 'HAI-20.07', 'PowerSystem')]
    seeds = (37,) if args.pilot else SEEDS
    doc = freeze(selected, args.output, seeds)
    output = args.output / 'summaries' / doc['manifest_id'][:12]
    output.mkdir(parents=True, exist_ok=True)
    historical_low_fpr(output)
    # A pilot uses a separate manifest despite sharing source/configuration choices.
    if args.dry_run:
        print(json.dumps({'manifest_id': doc['manifest_id'], 'cases': len(selected),
                          'executions': len(selected)*len(METHODS)*len(seeds),
                          'seeds': seeds, 'output': str(output)}, indent=2), flush=True)
        return 0
    records = []
    for index, entry in enumerate(doc['cases']):
        for path, expected in entry['source_hashes'].items():
            if digest(path) != expected:
                raise ValueError('Frozen source changed before execution')
        try:
            if entry['kind'] == 'nasa':
                case = prepare_nasa(entry['paths'][0])
            elif entry['kind'] == 'power':
                case = prepare_power(POWER)
            else:
                case = prepare_principal(entry['source_family'])
            error = None
        except ValueError as exc:
            case, error = None, str(exc)
        for seed in seeds:
            for method in METHODS:
                base = {'case_id': entry['case_id'], 'source_family': entry['source_family'],
                        'method_id': method, 'seed': seed}
                if error is not None:
                    records.append({**base, 'status': 'incompatible', 'error': error})
                else:
                    try:
                        records.append(run_one(case, method, seed, args.output, doc['manifest_id']))
                    except Exception as exc:
                        records.append({**base, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        _json(output / 'progress.json', {'cases_processed': index + 1, 'cases': len(selected),
                                        'records': len(records), 'case_id': entry['case_id']})
        print(json.dumps({'cases_processed': index+1, 'cases': len(selected),
                          'case_id': entry['case_id'], 'preparation_error': error}), flush=True)
    summary = summarize(records, output, doc)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
