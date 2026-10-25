"""Pinned cohort selection, immutable evidence, and campaign integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from research.evaluation_protocols.campaign import ROOT, METHODS as LOCAL, _identity, _json, digest

REVISION = '4101f6087f902fa150e392b65c957976faa84e40'
SEEDS = (37, 38, 39)
REDUCED = 'E1-motion-state-speed-r8'
DEEP = ('catch', 'mtad_gat')
METHODS = (*LOCAL, REDUCED, *DEEP)
OUTPUT = ROOT/'results/exathlon_native_frozen_20261025'
EXPECTED = {1: [68], 2: [60], 4: [61], 5: [63, 64]}


def immutable_json(path, document):
    path = Path(path)
    canonical = json.loads(json.dumps(document, sort_keys=True, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != canonical:
            raise ValueError('Existing immutable document differs: '+str(path))
    else:
        _json(path, canonical)
    return canonical


def verify_blob(path, expected):
    content = Path(path).read_bytes()
    actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    if actual != expected:
        raise ValueError('Pinned source blob differs: '+str(path))


def select_cohort(tree):
    if tree.get('truncated'):
        raise ValueError('Release inventory is truncated')
    groups, seen = {}, set()
    for row in tree['tree']:
        path = Path(row['path'])
        match = re.fullmatch(r'(\d+)_(\d+)_(\d+)_(\d+)\.zip', path.name)
        if not str(path).startswith('data/raw/app') or not match:
            continue
        app, kind, rate, number = map(int, match.groups())
        if path.stem in seen:
            raise ValueError('Ambiguous duplicate execution in release inventory')
        seen.add(path.stem)
        entry = {'path': row['path'], 'blob': row['sha'], 'bytes': row['size'],
                 'trace': path.stem, 'number': number, 'type': kind}
        group = groups.setdefault((app, rate), {'normal': [], 'disturbed': []})
        group['normal' if kind == 0 else 'disturbed'].append(entry)
    cohort, inventory = [], []
    for (app, rate), group in sorted(groups.items()):
        normals = sorted(group['normal'], key=lambda r: r['number'])
        tests = sorted(group['disturbed'], key=lambda r: r['number'])
        eligible = len(normals) >= 2 and bool(tests)
        inventory.append({'application': app, 'rate': rate, 'eligible': eligible,
                          'normal': normals, 'disturbed': tests,
                          'reason': 'eligible' if eligible else 'requires-two-normal-and-one-disturbed'})
        if eligible:
            cohort.append({'application': app, 'rate': rate, 'fit': normals[0],
                           'calibration': normals[1], 'tests': tests})
    return cohort, inventory


def code_evidence():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/p for p in (
        'pendulum_detector.py', 'range_metrics.py',
        'research/evaluation_protocols/campaign.py', 'research/submission_campaign/studies.py',
        'research/submission_campaign/deep.py', 'benchmarks/official_baselines/local_controls.py',
        'benchmarks/official_baselines/registry.json',
        'benchmarks/official_baselines/adapters/catch.py',
        'benchmarks/official_baselines/adapters/mtad_gat.py',
        'benchmarks/official_baselines/adapters/common.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}


def configuration():
    from research.submission_campaign.deep import configurations
    result = configurations()
    return {m: {**result[m], 'epochs': 10, 'window_size': 100, 'batch_size': 128,
                 'validation_partition': 'independent-execution-segment-aware-v1'} for m in DEEP}


def freeze(output=OUTPUT):
    import importlib.metadata
    from benchmarks.official_baselines.local_controls import DYNSPHERE_CONFIGURATION
    from research.submission_campaign.deep import declared_revisions, upstream_snapshot
    output = Path(output)
    prepared = json.loads((output/'prepared.json').read_text())
    inputs = {str(p.relative_to(output)): digest(p) for p in sorted((output/'inputs').rglob('*')) if p.is_file()}
    revisions = declared_revisions()
    doc = {'schema': 'exathlon-native-frozen-v1', 'release_revision': REVISION,
           'prepared': prepared, 'methods': METHODS, 'seeds': SEEDS,
           'local_configuration': DYNSPHERE_CONFIGURATION, 'deep_configuration': configuration(),
           'primary_trim': [100, 100], 'vus': {'backend': 'official_vus', 'version': '0.0.6',
           'window': 100, 'aggregation': 'within-segment-retained-length-weighted'},
           'implementation': code_evidence(), 'inputs': inputs,
           'upstream': {m: upstream_snapshot(ROOT/'benchmarks/official_baselines/.external'/m, revisions[m]) for m in DEEP},
           'preparation_versions': {m: importlib.metadata.version(m) for m in ('numpy', 'pandas', 'scikit-learn')},
           'manuscript_ready': False}
    doc['manifest_id'] = _identity(doc)
    return immutable_json(output/'manifest.json', doc)


def validate(output=OUTPUT, upstream=False):
    output = Path(output)
    doc = json.loads((output/'manifest.json').read_text())
    if _identity({k: v for k, v in doc.items() if k != 'manifest_id'}) != doc['manifest_id']:
        raise ValueError('Frozen manifest identity differs')
    if doc['implementation'] != code_evidence():
        raise ValueError('Frozen implementation source differs')
    for name, expected in doc['inputs'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or digest(output/relative) != expected:
            raise ValueError('Frozen input differs: '+name)
    if upstream:
        from research.submission_campaign.deep import upstream_snapshot
        for method, expected in doc['upstream'].items():
            actual = upstream_snapshot(ROOT/'benchmarks/official_baselines/.external'/method, expected['revision'])
            if actual != expected:
                raise ValueError('Frozen upstream source differs: '+method)
    return doc


def jobs(doc, group='all'):
    methods = LOCAL if group == 'local' else DEEP if group == 'deep' else (*LOCAL, *DEEP)
    return [{'application': row['application'], 'method': m, 'seed': seed}
            for row in doc['prepared']['cohort'] for m in methods for seed in doc['seeds']]
