import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.official_baselines import cached_extension_metrics as cached


def _record(tmp_path):
    run = tmp_path/'runs'/'case'/'method'/'seed-37'/'abc'
    (run/'case/private').mkdir(parents=True)
    (run/'execution').mkdir()
    np.save(run/'case/private/labels.npy', [0, 0, 1, 1, 0, 1])
    np.save(run/'execution/scores.npy', [999., 0., 3., 4., 1., 2.])
    record = {'run_identity': 'abc', 'case_id': 'case', 'method_id': 'method',
              'seed': 37, 'source_family': 'GutenTAG', 'status': 'complete'}
    (run/'run.json').write_text(json.dumps(record))
    return {**record, 'run_directory': '/old/machine/results/extension_v1/runs/case/method/seed-37/abc',
            'auprc': 1.0, 'auroc': 1.0, 'evaluation_warmup_rows': 1,
            'elapsed_seconds': 42.0}


def test_portable_cache_loading_uses_frozen_identity_and_common_warmup(tmp_path):
    row = _record(tmp_path)
    labels, scores = cached.load_scores(row, tmp_path/'runs')
    np.testing.assert_array_equal(labels, [0, 1, 1, 0, 1])
    np.testing.assert_array_equal(scores, [0, 3, 4, 1, 2])


def test_changed_scores_cannot_silently_reuse_the_pointwise_cohort(tmp_path):
    row = _record(tmp_path)
    path = next((tmp_path/'runs').rglob('scores.npy'))
    np.save(path, [0, 10, 3, 4, 1, 2])
    with pytest.raises(ValueError, match='pointwise'):
        cached.load_scores(row, tmp_path/'runs')


def test_cache_key_binds_labels_scores_and_backend_settings():
    y = np.array([0, 1, 0, 1]); s = np.arange(4.)
    a = cached.metric_key(y, s, {'backend': 'official-v1'})
    assert a != cached.metric_key(y[::-1], s, {'backend': 'official-v1'})
    assert a != cached.metric_key(y, s+1, {'backend': 'official-v1'})
    assert a != cached.metric_key(y, s, {'backend': 'official-v2'})


def test_official_metrics_are_reused_without_loading_a_detector(tmp_path):
    pytest.importorskip('vus')
    row = _record(tmp_path)
    labels, scores = cached.load_scores(row, tmp_path/'runs')
    first = cached.evaluate(labels, scores, tmp_path/'metrics')
    second = cached.evaluate(labels, scores, tmp_path/'metrics')
    assert first['vus_backend'] == 'official_vus'
    assert second['reused'] is True
    assert first['vus_pr'] == second['vus_pr']
    assert first['manuscript_ready'] is False


def test_cli_deduplicates_exact_inputs_and_preserves_run_coverage(tmp_path):
    row = _record(tmp_path)
    table = tmp_path/'input.csv'
    pd.DataFrame([row]).to_csv(table, index=False)
    result = cached.main(['--input-csv', str(table), '--runs-root', str(tmp_path/'runs'),
                          '--output-dir', str(tmp_path/'out'), '--dry-run'])
    assert result['selected_runs'] == 1
    assert result['unique_metric_inputs'] == 1
    assert not list((tmp_path/'out').rglob('scores.npy'))


def test_initial_seal_detects_rank_preserving_mutation_on_resume(tmp_path):
    row = _record(tmp_path)
    table = tmp_path/'input.csv'
    pd.DataFrame([row]).to_csv(table, index=False)
    args = ['--input-csv', str(table), '--runs-root', str(tmp_path/'runs'),
            '--output-dir', str(tmp_path/'out')]
    cached.main(args)
    output = pd.read_csv(tmp_path/'out/metrics-shard-000.csv')
    assert output.loc[0, 'metric_elapsed_seconds'] >= 0
    assert output.loc[0, 'elapsed_seconds'] == 42.0
    path = next((tmp_path/'runs').rglob('scores.npy'))
    np.save(path, np.load(path)*2)
    with pytest.raises(ValueError, match='seal'):
        cached.main([*args, '--dry-run'])


def test_exact_duplicates_are_evaluated_once_across_shards_with_complete_coverage(tmp_path):
    import shutil
    row = _record(tmp_path)
    original = next((tmp_path/'runs').rglob('scores.npy')).parents[1]
    second = original.parent/'def'
    shutil.copytree(original, second)
    record = json.loads((second/'run.json').read_text())
    record.update(run_identity='def', seed=38)
    (second/'run.json').write_text(json.dumps(record))
    other = {**row, **record, 'run_directory': row['run_directory'].rsplit('/', 1)[0]+'/def'}
    table = tmp_path/'input.csv'
    pd.DataFrame([row, other]).to_csv(table, index=False)
    for shard in range(2):
        cached.main(['--input-csv', str(table), '--runs-root', str(tmp_path/'runs'),
                     '--output-dir', str(tmp_path/'out'), '--shard-count', '2',
                     '--shard-index', str(shard)])
    assert len(list((tmp_path/'out/cache').glob('*.json'))) == 1
    results = pd.concat(pd.read_csv(p) for p in (tmp_path/'out').glob('metrics-shard-*.csv'))
    assert set(results.run_identity) == {'abc', 'def'}
    assert len(results) == 2

