import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.window_dynsphere.pilot import (
    atomic_json, freeze_manifest, run_case, select_pilot_entries, source_identity, record_process_failure,
)
from research.window_dynsphere.controls import WindowRidgeControl


def test_selection_is_outcome_independent_and_has_all_declared_groups():
    entries = []
    for family in ('Exathlon', 'GHL'):
        for suffix in ('a', 'b', 'c', 'd'):
            entries.append({'case_id': f'{family}/{suffix}', 'source_family': family})
    for mechanism in ('amplitude', 'frequency', 'mean', 'mode-correlation', 'pattern-shift', 'variance'):
        for seed in (101, 102, 103):
            entries.append({'case_id': f'GutenTAG/{mechanism}-{seed}', 'source_family': 'GutenTAG',
                            'mechanism': mechanism, 'generation_seed': seed})
    for family in ('BATADAL', 'HAI-20.07', 'PowerSystem'):
        entries.append({'case_id': family + '/complete', 'source_family': family})
    result = select_pilot_entries(entries)
    assert len(result) == 15
    assert all(e.get('generation_seed', 101) == 101 for e in result)
    assert result == select_pilot_entries(list(reversed(entries)))
    assert len([e for e in result if e['source_family'] == 'Exathlon']) == 3
    assert {e['mechanism'] for e in result if e['source_family'] == 'GutenTAG'} == {
        'amplitude', 'frequency', 'mean', 'mode-correlation', 'pattern-shift', 'variance'}


def test_frozen_manifest_and_missing_sources_are_explicit(tmp_path):
    entry = {'case_id': 'GutenTAG-native-v1/a', 'case_name': 'a', 'source_family': 'GutenTAG', 'kind': 'native',
             'train': str(tmp_path/'missing.csv'), 'test': str(tmp_path/'also_missing.csv')}
    p = freeze_manifest([entry], tmp_path, {'test': {'window': 8, 'epochs': 1}})
    assert p == freeze_manifest([entry], tmp_path, {'test': {'window': 8, 'epochs': 1}})
    record = run_case(p, entry['case_id'], 'test', tmp_path)
    assert record['status'] == 'incompatible'
    assert 'missing' in record['error'].lower()


def small_native(tmp_path, single_class=False):
    t = np.arange(400)
    train = pd.DataFrame({'x': np.sin(t/7), 'y': np.cos(t/13), 'Label': np.zeros(400, int)})
    test = train.iloc[:120].copy()
    if not single_class:
        test.loc[60:79, 'Label'] = 1
        test.loc[60:79, 'x'] += 3
    a, b = tmp_path/'train.csv', tmp_path/'test.csv'
    train.to_csv(a, index=False)
    test.to_csv(b, index=False)
    entry = {'case_id': 'GutenTAG-native-v1/test', 'source_family': 'GutenTAG', 'kind': 'native',
             'train': str(a), 'test': str(b), 'case_name': 'test'}
    config = {'test': {'window': 8, 'hidden_dim': 6, 'latent_dim': 3, 'epochs': 1,
                       'max_targets': 32, 'batch_size': 16, 'n_regimes': 1,
                       'reconstruction_weight': 1., 'random_state': 37}}
    return freeze_manifest([entry], tmp_path, config), entry


def test_actual_case_replays_and_corrupt_scores_are_rejected(tmp_path):
    p, entry = small_native(tmp_path)
    a = run_case(p, entry['case_id'], 'test', tmp_path)
    assert a['status'] == 'complete'
    b = run_case(p, entry['case_id'], 'test', tmp_path)
    assert b['reused']
    assert a['run_identity'] == b['run_identity']
    score_path = Path(a['run_directory'])/'scores.npy'
    np.save(score_path, np.zeros(120))
    with pytest.raises(ValueError, match='integrity'):
        run_case(p, entry['case_id'], 'test', tmp_path)


def test_input_mutation_cannot_reuse_previous_run(tmp_path):
    p, entry = small_native(tmp_path)
    Path(entry['test']).write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        run_case(p, entry['case_id'], 'test', tmp_path)


def test_single_class_after_warmup_remains_a_recorded_issue(tmp_path):
    p, entry = small_native(tmp_path, single_class=True)
    result = run_case(p, entry['case_id'], 'test', tmp_path)
    assert result['status'] == 'incompatible'
    assert 'both label classes' in result['error']


def test_manifest_source_identity_tampering_is_rejected(tmp_path):
    p, entry = small_native(tmp_path)
    doc = json.loads(p.read_text())
    doc['source_identity'] = 'incorrect'
    atomic_json(p, doc)
    with pytest.raises(ValueError, match='source|manifest'):
        run_case(p, entry['case_id'], 'test', tmp_path)


def test_window_ridge_scores_are_causal_and_calibrated():
    t = np.arange(400)
    x = np.column_stack([np.sin(t/7), np.cos(t/13)])
    model = WindowRidgeControl(window=8, max_targets=64).fit(x, np.zeros(400))
    y = x[:120].copy()
    a = model.decision_function(y, np.zeros(120))
    y[60:] += 50
    b = model.decision_function(y, np.zeros(120))
    np.testing.assert_allclose(a[:60], b[:60], equal_nan=True)
    assert np.isfinite(a[8:]).all() and np.isnan(a[:8]).all()


def test_cached_record_identity_corruption_is_rejected(tmp_path):
    p, entry = small_native(tmp_path)
    result = run_case(p, entry['case_id'], 'test', tmp_path)
    path = Path(result['run_directory'])/'run.json'
    result['case_id'] = 'wrong-case'
    atomic_json(path, result)
    with pytest.raises(ValueError, match='integrity'):
        run_case(p, entry['case_id'], 'test', tmp_path)


def test_neural_head_reuses_zero_weight_representation(tmp_path):
    p, entry = small_native(tmp_path)
    doc = json.loads(p.read_text())
    local = {**doc['configurations']['test'], 'reconstruction_weight': 0}
    head = {**local, 'reconstruction_weight': 0., 'variant': 'head'}
    p = freeze_manifest([entry], tmp_path, {'local': local, 'head': head})
    a = run_case(p, entry['case_id'], 'local', tmp_path)
    b = run_case(p, entry['case_id'], 'head', tmp_path)
    assert a['status'] == b['status'] == 'complete'
    assert not a['representation_reused'] and b['representation_reused']


def test_child_timeout_is_a_durable_outcome_record(tmp_path):
    p, entry = small_native(tmp_path)
    record = record_process_failure(p, entry['case_id'], 'test', tmp_path, 'timeout')
    assert record['status'] == 'timeout'
    assert json.loads((Path(record['run_directory'])/'run.json').read_text())['status'] == 'timeout'


@pytest.mark.parametrize('variant', ['local','head','ridge','previous'])
def test_every_default_configuration_preserves_frozen_hyperparameters(tmp_path, variant):
    p, entry = small_native(tmp_path)
    options = json.loads(p.read_text())['configurations']['test']
    options.pop('random_state')
    if variant != 'local':
        options['variant'] = variant
    p = freeze_manifest([entry], tmp_path, {'test': options})
    first = run_case(p, entry['case_id'], 'test', tmp_path)
    assert first['status'] == 'complete'
    second = run_case(p, entry['case_id'], 'test', tmp_path)
    assert second['reused'] and second['hyperparameters'] == options


def test_missing_input_appearing_requires_a_new_frozen_manifest(tmp_path):
    p, entry = small_native(tmp_path)
    Path(entry['train']).unlink()
    p = freeze_manifest([entry], tmp_path, {'test': {'window':8,'epochs':1}})
    pd.DataFrame({'x':np.arange(400.),'y':np.arange(400.),'Label':np.zeros(400)}).to_csv(entry['train'],index=False)
    with pytest.raises(ValueError, match='availability|hash'):
        run_case(p,entry['case_id'],'test',tmp_path)


def test_failed_resume_attempt_is_recorded_without_destroying_success(tmp_path):
    p, entry = small_native(tmp_path)
    first = run_case(p,entry['case_id'],'test',tmp_path)
    failed = record_process_failure(p,entry['case_id'],'test',tmp_path,'input hash rejected')
    assert failed['status']=='failed'
    historical = json.loads((Path(first['run_directory'])/'run.json').read_text())
    assert historical['status']=='complete'
    assert list((tmp_path/'attempt_failures').rglob('*.json'))
