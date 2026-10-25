"""Disjoint Exathlon/GHL and native GutenTAG extension pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmarks.official_baselines.campaign import (
    campaign_document, partition_jobs, write_campaign_manifest,
)
from benchmarks.official_baselines.contract import EvaluationCase, write_run_record
from benchmarks.official_baselines.datasets import fit_training_scaler, source_name
from benchmarks.official_baselines.local_controls import LOCAL_SCORERS
from benchmarks.official_baselines.provenance import ImplementationSpec, load_registry
from benchmarks.official_baselines.run_expanded_evaluation import (
    CaseDescriptor, ExperimentJob, DEFAULT_MANIFEST, DEFAULT_REGISTRY, DEFAULT_TSB_ROOT,
    REPO_ROOT, execute_job,
)


METHODS = ('dynsphere', 'dynsphere_euclidean', 'dynsphere_one_regime',
           'dynsphere_euclidean_one_regime', 'pca_reconstruction', 'linear_ridge_ar1')
FAMILIES = ('Exathlon', 'GHL', 'GutenTAG')
PROTOCOL = 'disjoint-extension-v1'


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest() -> str:
    """Identify actual source contents, including an uncommitted pilot."""
    package = Path(__file__).resolve().parent
    files = list(REPO_ROOT.glob('*.py')) + list(package.glob('*.py'))
    files += list((package / 'adapters').glob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _observations(path: Path):
    frame = pd.read_csv(path)
    label = 'Label' if 'Label' in frame else 'is_anomaly'
    if label not in frame:
        raise ValueError(f'Missing anomaly label: {path}')
    labels = frame[label].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f'Expected finite binary labels: {path}')
    features = [name for name in frame if name not in (label, 'timestamp', 'time', 'entity_id')]
    values = frame[features].to_numpy(dtype=float)
    if values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError(f'Expected finite multivariate observations without row deletion: {path}')
    return values, labels.astype(int), features


def load_suffix_case(path: Path, seed: int) -> EvaluationCase:
    """Train on a clean declared prefix; score only the untouched suffix."""
    values, labels, features = _observations(path)
    match = re.search(r'_tr_(\d+)_', path.name)
    if match is None:
        raise ValueError(f'Missing declared prefix: {path}')
    stop = int(match.group(1))
    if not 16 <= stop <= len(values) - 4:
        raise ValueError(f'Insufficient prefix or suffix: {path}')
    if labels[:stop].any():
        raise ValueError(f'Training prefix is contaminated: {path}')
    train, test, scaling = fit_training_scaler(values[:stop], values[stop:])
    family = source_name(path)
    return EvaluationCase(
        case_id=f'TSB-AD-M-disjoint-v1/{path.stem}', dataset_id='TSB-AD-M-disjoint-v1',
        source_family=family, trajectory_id=path.stem, train=train, test=test,
        labels=labels[stop:], metadata={
            **scaling, 'seed': seed, 'protocol': PROTOCOL, 'row_id_start': stop,
            'row_id_stop': len(values), 'feature_names': features,
            'source_sha256': file_digest(path), 'source_file': path.name,
            'training_protocol': 'clean-declared-prefix',
            'score_context': 'suffix-only; no prefix concatenation',
        },
    )


def load_native_case(train_path: Path, test_path: Path, family: str,
                     case_name: str, seed: int, expected_hashes=None) -> EvaluationCase:
    if train_path.resolve() == test_path.resolve():
        raise ValueError('Training and test files must be distinct')
    if expected_hashes is not None:
        for kind, path in [('train', train_path), ('test', test_path)]:
            expected = expected_hashes.get(f'{kind}_sha256')
            if expected is not None and file_digest(path) != expected:
                raise ValueError(f'Native file hash mismatch: {path}')
    train, labels, features = _observations(train_path)
    test, test_labels, test_features = _observations(test_path)
    if labels.any():
        raise ValueError(f'Normal training file is contaminated: {train_path}')
    if len(train) < 16 or len(test) < 4 or features != test_features:
        raise ValueError('Native train/test files have incompatible features or lengths')
    train, test, scaling = fit_training_scaler(train, test)
    return EvaluationCase(
        case_id=f'{family}-native-v1/{case_name}', dataset_id=f'{family}-native-v1',
        source_family=family, trajectory_id=case_name, train=train, test=test,
        labels=test_labels, metadata={
            **scaling, 'seed': seed, 'protocol': PROTOCOL, 'feature_names': features,
            'training_protocol': 'separate-normal-training-file',
            'train_sha256': file_digest(train_path), 'test_sha256': file_digest(test_path),
            'row_id_start': 0, 'row_id_stop': len(test),
        },
    )


def read_native_pairs(manifest: Path, families: Sequence[str]) -> list[dict]:
    payload = json.loads(manifest.read_text())
    pairs = []
    for entry in payload['cases']:
        if entry['family'] not in families:
            continue
        pair = dict(entry)
        if pair['family'] == 'GutenTAG':
            from benchmarks.official_baselines.prepare_gutentag_extension import GENERATOR_VERSION
            if (payload.get('version') != GENERATOR_VERSION or
                    payload.get('generator') != 'official TimeEval/GutenTAG'):
                raise ValueError('Unexpected GutenTAG generator or version')
        for field in ('train', 'test'):
            path = (manifest.parent / pair[field]).resolve()
            expected = pair.get(f'{field}_sha256')
            if pair['family'] == 'GutenTAG' and not expected:
                raise ValueError(f'Missing generated file hash: {path}')
            if expected is not None and file_digest(path) != expected:
                raise ValueError(f'Native file hash mismatch: {path}')
            pair[field] = str(path)
        pairs.append(pair)
    return pairs


def select_tsb_paths(root: Path, names: Sequence[str], families: Sequence[str],
                     limit: int) -> list[Path]:
    if limit < 0:
        raise ValueError('The case limit must be nonnegative')
    available = {}
    for path in root.rglob('*.csv'):
        if path.name not in names:
            continue
        if path.name in available:
            raise ValueError(f'Duplicate dataset filename: {path.name}')
        available[path.name] = path
    counts = Counter()
    selected = []
    for name in names:
        family = source_name(Path(name))
        if family not in families or (limit and counts[family] >= limit):
            continue
        if name not in available:
            raise FileNotFoundError(f'Missing selected dataset: {name}')
        selected.append(available[name])
        counts[family] += 1
    return selected


def point_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and row-aligned')
    # A common one-row exclusion handles the AR(1) control's unavailable predecessor.
    labels, scores = labels[1:], scores[1:]
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Evaluation suffix must contain both label classes after warmup')
    return {
        'auprc': float(average_precision_score(labels, scores)),
        'auroc': float(roc_auc_score(labels, scores)),
        'pauc_fpr_05': float(roc_auc_score(labels, scores, max_fpr=0.05)),
        'anomaly_prevalence': float(labels.mean()), 'evaluated_rows': len(labels),
        'evaluation_warmup_rows': 1,
    }


def build_jobs(paths, pairs, result_root, methods, seeds, *, baseline_profile='historical-v1'):
    if baseline_profile not in ('historical-v1', 'audited-v2'):
        raise ValueError(f'Unknown baseline profile: {baseline_profile}')
    if not methods or not seeds or len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        raise ValueError('Methods and seeds must be nonempty and unique')
    registry = load_registry(DEFAULT_REGISTRY)
    if baseline_profile == 'audited-v2':
        overrides = {
            'tranad': {'pipeline_revision': 'audited-v2', 'epochs': 5},
            'anomaly_transformer': {'training_revision': 'released-budget-v2',
                                    'epochs': 10, 'd_model': 512, 'n_heads': 8},
        }
        for method, configuration in overrides.items():
            registry[method] = replace(registry[method], default_hyperparameters={
                **registry[method].default_hyperparameters, **configuration})
    reference = registry['pca_reconstruction']
    registry['linear_ridge_ar1'] = ImplementationSpec(
        method_id='linear_ridge_ar1', display_name='Linear ridge AR(1)', source_kind='local',
        implementation_category='repository-maintained', repository_url='.',
        revision=reference.revision, python='>=3.9', adapter_command=('__local__', 'linear_ridge_ar1'),
        default_hyperparameters={'ridge_alpha': 1.0}, fallback_ids=(),
        adapter_revision=reference.adapter_revision,
    )
    # External methods can be selected explicitly after the fidelity pilot.
    unknown = set(methods) - registry.keys()
    if unknown:
        raise ValueError(f'Unknown methods: {sorted(unknown)}')
    descriptors = []
    source_hashes = {
        f'TSB-AD-M-disjoint-v1/{path.stem}': {'source': file_digest(path)} for path in paths}
    source_hashes.update({f"{p['family']}-native-v1/{p['case_name']}": {
        'train': file_digest(Path(p['train'])), 'test': file_digest(Path(p['test']))} for p in pairs})
    for seed in seeds:
        for path in paths:
            descriptors.append(CaseDescriptor(
                f'TSB-AD-M-disjoint-v1/{path.stem}', 'TSB-AD-M-disjoint-v1',
                source_name(path), path.stem, seed,
                lambda path=path, seed=seed: load_suffix_case(path, seed)))
        for pair in pairs:
            family, name = pair['family'], pair['case_name']
            if family not in FAMILIES:
                raise ValueError(f'Unexpected native family: {family}')
            descriptors.append(CaseDescriptor(
                f'{family}-native-v1/{name}', f'{family}-native-v1', family, name, seed,
                lambda pair=pair, seed=seed: load_native_case(
                    Path(pair['train']), Path(pair['test']), pair['family'], pair['case_name'], seed,
                    expected_hashes=pair)))
    fingerprint = source_digest()
    jobs = [ExperimentJob(
        case=case, implementation=registry[method], hyperparameters={
            **registry[method].default_hyperparameters, 'extension_source_digest': fingerprint,
            'evaluation_protocol': PROTOCOL, 'raw_source_sha256': source_hashes[case.case_id],
            'baseline_profile': baseline_profile,
        }, result_root=result_root, timeout_seconds=14400, resume=True,
    ) for case in sorted(descriptors, key=lambda c: (c.case_id, c.seed)) for method in sorted(methods)]
    keys = [(j.case.case_id, j.case.seed, j.implementation.method_id) for j in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate extension case identities')
    return jobs


def run_pilot(paths, pairs, result_root, *, methods=METHODS, seeds=(37,),
              shard_index=0, shard_count=1, dry_run=False, with_vus=False,
              baseline_profile='historical-v1'):
    result_root = Path(result_root)
    jobs = build_jobs(paths, pairs, result_root, methods, seeds, baseline_profile=baseline_profile)
    if not jobs:
        raise ValueError('No extension cases selected')
    selection = partition_jobs(jobs, shard_index, shard_count)
    document = campaign_document(PROTOCOL, jobs, shard_count)
    document['extension_source_digest'] = source_digest()
    document['metrics_protocol'] = 'pointwise-plus-official-vus' if with_vus else 'pointwise-pilot-only'
    document['input_source_sha256'] = {str(p.name): file_digest(p) for p in paths}
    document['native_source_sha256'] = {
        f"{p['family']}/{p['case_name']}": {
            'train': file_digest(Path(p['train'])), 'test': file_digest(Path(p['test']))}
        for p in pairs}
    document['campaign_id'] = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
    folder = result_root / 'summaries' / document['campaign_id'][:12]
    manifest = result_root / 'campaigns' / (document['campaign_id'][:12] + '.json')
    write_campaign_manifest(manifest, document)
    summary = {'campaign_manifest': str(manifest), 'planned_runs': len(jobs),
               'selected_runs': len(selection), 'manuscript_ready': False,
               'families': dict(Counter(j.case.source_family for j in jobs))}
    if dry_run:
        return summary
    folder.mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for job in selection:
        record = execute_job(job)
        records.append(record)
        print(json.dumps({k: record.get(k) for k in
                          ('case_id', 'method_id', 'seed', 'status', 'elapsed_seconds', 'skipped')}), flush=True)
        if record['status'] != 'complete':
            continue
        try:
            case = job.case.loader()
            scores = np.load(Path(record['run_directory']) / 'execution' / 'scores.npy', allow_pickle=False)
            metrics = point_metrics(case.labels, scores)
            if with_vus:
                from benchmarks.official_baselines.metrics import compute_metrics
                metrics.update(compute_metrics(case.labels[1:], scores[1:], 0.01, None))
                if metrics.get('vus_backend') != 'official_vus':
                    raise ValueError('Full extension metrics require the official VUS backend')
            rows.append({**{k: record[k] for k in ('case_id', 'source_family', 'trajectory_id',
                         'method_id', 'seed', 'run_identity', 'elapsed_seconds', 'run_directory')},
                         'train_rows': len(case.train), 'scored_rows': len(case.test),
                         'feature_count': case.train.shape[1], **metrics, 'manuscript_ready': False})
        except Exception as exc:
            record = {**record, 'status': 'metric_failed', 'metric_error': str(exc)}
            records[-1] = record
    metrics_path = folder / f'point_metrics-shard-{shard_index:03d}.csv'
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    if rows:
        frame = pd.DataFrame(rows)
        # Seed means precede trajectory means; synthetic generation replicates are cases.
        grouped = frame.groupby(['source_family', 'method_id', 'trajectory_id'])[
            ['auprc', 'auroc', 'pauc_fpr_05']].mean()
        grouped.groupby(['source_family', 'method_id']).mean().to_csv(
            folder / f'per_family-shard-{shard_index:03d}.csv')
    summary.update(outcomes=dict(Counter(r['status'] for r in records)),
                   resumed_count=sum(bool(r.get('skipped')) for r in records), metrics_csv=str(metrics_path))
    write_run_record(folder / f'completion-shard-{shard_index:03d}.json', {
        **summary, 'records': records,
        'status': 'complete' if all(r['status'] == 'complete' for r in records) else 'failed',
    })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=['Exathlon', 'GHL'])
    parser.add_argument('--tsb-data-root', type=Path, default=DEFAULT_TSB_ROOT)
    parser.add_argument('--native-manifest', type=Path)
    parser.add_argument('--limit-per-family', type=int, default=2,
                        help='First files in the fixed manifest; zero selects all eligible files')
    parser.add_argument('--methods', nargs='+', default=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[37])
    parser.add_argument('--result-root', type=Path, default=REPO_ROOT / 'results' / 'extension_v1')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--with-vus', action='store_true')
    parser.add_argument('--baseline-profile', choices=['historical-v1', 'audited-v2'],
                        default='historical-v1')
    args = parser.parse_args(argv)
    try:
        if 'GutenTAG' in args.families and args.native_manifest is None:
            raise ValueError('GutenTAG requires a generated --native-manifest')
        names = pd.read_csv(DEFAULT_MANIFEST)['file_name'].tolist()
        paths = select_tsb_paths(args.tsb_data_root, names, args.families, args.limit_per_family)
        pairs = []
        if args.native_manifest is not None:
            pairs = read_native_pairs(args.native_manifest, args.families)
        summary = run_pilot(paths, pairs, args.result_root, methods=args.methods, seeds=args.seeds,
                            shard_index=args.shard_index, shard_count=args.shard_count,
                            dry_run=args.dry_run, with_vus=args.with_vus,
                            baseline_profile=args.baseline_profile)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return int(any(status != 'complete' for status in summary.get('outcomes', {})))
    except Exception as exc:
        print(f'Extension campaign failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
