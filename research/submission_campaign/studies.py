"""Matched grouped-evidence analysis of the frozen held-out E1 configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pendulum_detector import _rolling_entity_max
from research.evaluation_protocols.campaign import (
    ROOT, FittedControl, _identity, _json, digest, evaluate,
    prepare_nasa, prepare_power, prepare_principal)


BASE_ID = '502a1002c7788742def0a4bc13aeac1885b7376aec38ad5f3ddab3b07b2a0423'
BASE = ROOT / 'results/protocol_validation_20261024'
OUT = ROOT / 'results/submission_push_20261024'
STATE = ('position', 'regime_distance')
MOTION = ('euclidean_dynamics', 'raw_dynamics', 'state_speed', 'velocity')


def resolve_source(path):
    original = Path(path)
    candidate = ROOT / 'benchmarks' / str(path).split('/benchmarks/', 1)[-1]
    if '/benchmarks/' not in str(path):
        raise ValueError('Unexpected frozen source location')
    return original if original.is_file() else candidate


def load_case(entry):
    paths = [resolve_source(p) for p in entry['paths']]
    for original, path in zip(entry['paths'], paths):
        if digest(path) != entry['source_hashes'][original]:
            raise ValueError('Frozen input hash changed')
    if entry['kind'] == 'nasa':
        return prepare_nasa(paths[0])
    if entry['kind'] == 'power':
        return prepare_power(paths[0].parent)
    return prepare_principal(entry['source_family'])


def grouped_scores(components, entities, n_concepts):
    if n_concepts != 1:
        raise ValueError('Grouped component reuse requires one regime')
    state = sum(components[k] for k in STATE)
    motion = sum(components[k] for k in MOTION)
    verify_reference(state + motion, components['pre_propagation_score'])
    result = {}
    for radius in (0, 8):
        for name, values in [('state', state), ('motion', motion), ('combined', state + motion)]:
            result[f'{name}_r{radius}'] = _rolling_entity_max(values, entities, radius)
    return result


def verify_reference(actual, frozen):
    if np.shape(actual) != np.shape(frozen) or not np.allclose(actual, frozen, rtol=1e-10, atol=1e-10):
        raise ValueError('Reconstructed score differs from frozen E1 reference')


def run(output=OUT):
    manifest = json.loads((BASE / 'manifests' / (BASE_ID + '.json')).read_text())
    sources = [Path(__file__), ROOT/'research/evaluation_protocols/campaign.py', ROOT/'pendulum_detector.py']
    identity = _identity({'base': BASE_ID, 'sources': {str(p.relative_to(ROOT)): digest(p) for p in sources},
                          'groups': [STATE, MOTION], 'radii': [0, 8]})
    output = Path(output) / 'mechanism' / identity[:12]
    output.mkdir(parents=True, exist_ok=True)
    _json(output/'protocol.json', {'identity': identity, 'base_manifest': BASE_ID,
          'groups': {'state_locality': STATE, 'motion_state_speed': MOTION}, 'radii': [0, 8],
          'caveat': 'State/locality includes state-velocity distance; state-speed includes both groups',
          'manuscript_ready': False})
    entries = {e['case_id']: e for e in manifest['cases']}
    records = [json.loads(p.read_text()) for p in (BASE/'runs').glob('*/run.json')]
    refs = [r for r in records if r['manifest_id'] == BASE_ID
            and r['method_id'] == 'dynsphere_euclidean_one_regime' and r['status'] == 'complete']
    rows = []
    for i, reference in enumerate(sorted(refs, key=lambda r:(r['case_id'],r['seed']))):
        checkpoint = output / (reference['run_identity'] + '.json')
        score_path = checkpoint.with_suffix('.npz')
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text())
            if saved['identity'] != identity or digest(score_path) != saved['score_sha256']:
                raise ValueError('Mechanism checkpoint integrity mismatch')
            rows.extend(saved['rows'])
            continue
        case = load_case(entries[reference['case_id']])
        frozen_path = Path(reference['run_directory'])/'scores.npz'
        if digest(frozen_path) != reference['artifact_sha256']:
            raise ValueError('Frozen reference artifact changed')
        with np.load(frozen_path, allow_pickle=False) as frozen:
            if not np.array_equal(case['labels'], frozen['labels']):
                raise ValueError('Frozen reference labels changed')
            with threadpool_limits(limits=1):
                model = FittedControl(reference['method_id'], reference['seed']).fit(case['train'],case['entities'][0]).model
                test = grouped_scores(model.decision_components(case['test'],case['entities'][2]),case['entities'][2],model.n_concepts_)
                cal = grouped_scores(model.decision_components(case['calibration'],case['entities'][1]),case['entities'][1],model.n_concepts_)
            verify_reference(test['combined_r8'], frozen['scores'])
            verify_reference(cal['combined_r8'], frozen['calibration_scores'])
        current = []
        for name, values in test.items():
            metrics = evaluate(case['labels'][case['test_mask']], values[case['test_mask']],
                               cal[name][case['calibration_mask']])
            current.append({'case_id':reference['case_id'],'source_family':reference['source_family'],
                            'seed':reference['seed'],'configuration':name,**metrics})
        np.savez_compressed(score_path, **test)
        _json(checkpoint, {'identity':identity,'reference_run_identity':reference['run_identity'],
                          'score_sha256':digest(score_path),'rows':current})
        rows.extend(current)
        print(json.dumps({'completed_reference_fits':i+1,'of':len(refs),'case_id':reference['case_id']}),flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'per_run.csv',index=False)
    metrics = ['auprc','auroc','pauc_fpr_05','fpr_05','recall_05']
    case = frame.groupby(['source_family','case_id','configuration'],as_index=False)[metrics].mean()
    case.to_csv(output/'per_case.csv',index=False)
    case.groupby(['source_family','configuration'],as_index=False)[metrics].mean().to_csv(output/'per_family.csv',index=False)
    local = [r for r in records if r['manifest_id']==BASE_ID and r['status']=='complete']
    cost = pd.DataFrame([{k:r[k] for k in ['source_family','case_id','method_id','seed','fit_seconds','elapsed_seconds']}
                          | {'auprc':r['metrics']['auprc']} for r in local])
    cost.groupby(['source_family','method_id'],as_index=False)[['fit_seconds','elapsed_seconds','auprc']].median().to_csv(
        output/'local_recorded_cost.csv',index=False)
    _json(output/'completion.json',{'status':'complete','reference_fits':len(refs),'metric_records':len(rows),
                                   'identity':identity,'manuscript_ready':False})
    print(json.dumps({'output':str(output),'reference_fits':len(refs),'metric_records':len(rows)}),flush=True)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    run(parser.parse_args().output)
