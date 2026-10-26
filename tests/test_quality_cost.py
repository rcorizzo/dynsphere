import importlib
import json

import numpy as np
import pytest


def runner():
    try:
        return importlib.import_module('research.quality_cost.run')
    except ModuleNotFoundError:
        pytest.fail('The isolated quality-cost runner is not implemented')


def test_selection_is_metadata_only_lower_median_with_stable_ties():
    rows = [{'case_id': name, 'source_family': 'MSL', 'test_rows': n,
             'eligible': eligible}
            for name, n, eligible in [('z', 20, True), ('b', 10, True),
                                      ('a', 10, True), ('bad', 1, False),
                                      ('c', 30, True)]]
    selected = runner().select_cases(rows, ('MSL',))
    assert [r['case_id'] for r in selected] == ['b']
    assert runner().select_cases(list(reversed(rows)), ('MSL',)) == selected
    with pytest.raises(ValueError, match='eligible'):
        runner().select_cases(rows, ('BATADAL',))


def test_cpu_worker_environment_pins_threads_without_mutating_parent():
    original = {'DYNSPHERE_DEVICE': 'cuda', 'OMP_NUM_THREADS': '8', 'PATH': 'saved'}
    env = runner().cpu_environment(original)
    assert original['DYNSPHERE_DEVICE'] == 'cuda'
    assert env['DYNSPHERE_DEVICE'] == 'cpu' and env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['OMP_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '1'
    assert env['MKL_NUM_THREADS'] == env['VECLIB_MAXIMUM_THREADS'] == '1'
    assert env['PATH'] == 'saved'


def test_support_excludes_boundaries_but_rejects_unobserved_interior():
    scores = np.arange(410, dtype=float)
    scores[:100] = np.nan
    scores[-100:] = np.nan
    mask = runner().common_support(scores)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(100, 310))
    scores[200] = np.nan
    with pytest.raises(ValueError, match='support'):
        runner().common_support(scores)


def test_existing_record_is_preserved(tmp_path):
    target = tmp_path / 'run.json'
    runner().write_new_json(target, {'status': 'original'})
    with pytest.raises(FileExistsError):
        runner().write_new_json(target, {'status': 'replacement'})
    assert json.loads(target.read_text()) == {'status': 'original'}


def test_serialization_replays_stochastic_score_without_changing_reference():
    module = runner()
    np.random.seed(13)
    state = module.capture_random_state()
    expected = np.random.uniform(size=410)
    payload = {'fitted_parameter': 7.}
    size = module.check_serialization(payload, lambda model: np.random.uniform(size=410),
                                      expected, state)
    assert size > 0
    assert payload == {'fitted_parameter': 7.}


def test_summary_keeps_cost_medians_and_quality_means_separate():
    records = []
    for seed, fit, quality in [(37, 1., .6), (38, 9., .8), (39, 2., .7)]:
        records.append(dict(status='complete', method_id='pca_reconstruction',
                            case_id='case', source_family='MSL', seed=seed,
                            fit_seconds=fit, score_seconds=fit / 2,
                            process_peak_rss_bytes=100., serialized_bytes=30.,
                            pauc_fpr_05=quality, auprc=quality, auroc=quality))
    frame = runner().summarize(records)
    assert frame.iloc[0].fit_seconds == 2.
    assert frame.iloc[0].score_seconds == 1.
    assert frame.iloc[0].pauc_fpr_05 == pytest.approx(.7)
    with pytest.raises(ValueError, match='Duplicate'):
        runner().summarize(records + [records[0]])


def test_local_measurement_roundtrip_and_full_length_scores(tmp_path):
    module = runner()
    rng = np.random.RandomState(5)
    train, test = rng.normal(size=(600, 3)), rng.normal(size=(410, 3))
    for name, values in [('train', train), ('test', test),
                         ('train_entities', np.zeros(600, dtype=int)),
                         ('test_entities', np.zeros(410, dtype=int))]:
        np.save(tmp_path / (name + '.npy'), values)
    record, scores = module.local_trial('dynsphere_euclidean_one_regime', 37, tmp_path)
    assert scores.shape == (410,) and np.isfinite(scores).all()
    assert record['fit_seconds'] > 0 and record['score_seconds'] > 0
    assert record['serialized_bytes'] > 0 and record['serialization_verified']
    assert record['process_peak_rss_bytes'] > 0


def test_summary_retains_completion_counts_and_all_failed_groups():
    record = dict(status='failed', method_id='catch', case_id='case',
                  source_family='HAI-20.07', seed=37)
    frame = runner().summarize([record])
    assert frame.iloc[0].planned_seeds == 1
    assert frame.iloc[0].completed_seeds == 0
    assert np.isnan(frame.iloc[0].fit_seconds)


def test_frozen_configuration_cannot_be_changed(tmp_path):
    target = tmp_path / 'configuration.json'
    runner().ensure_configuration(target, {'epochs': 10})
    runner().ensure_configuration(target, {'epochs': 10})
    with pytest.raises(ValueError, match='configuration'):
        runner().ensure_configuration(target, {'epochs': 2})


def test_worker_recovery_checks_identity_configuration_and_score_hash(tmp_path):
    module = runner()
    inputs = tmp_path / 'inputs'
    output = tmp_path / 'worker'
    inputs.mkdir()
    output.mkdir()
    np.save(inputs / 'test.npy', np.zeros((410, 2)))
    scores = np.arange(410, dtype=float)
    np.save(output / 'scores.npy', scores)
    expected = dict(method_id='catch', seed=37, manifest_id='frozen',
                    case_id='MSL/case', configuration={'epochs': 10})
    record = dict(expected, status='complete',
                  scores_sha256=module.digest(output / 'scores.npy'))
    module.write_new_json(output / 'run.json', record)
    recovered, actual = module.validate_worker(output, expected, 410)
    assert recovered == record
    np.testing.assert_array_equal(actual, scores)
    with pytest.raises(ValueError, match='identity'):
        module.validate_worker(output, dict(expected, seed=38), 410)
    np.save(output / 'scores.npy', scores + 1)
    with pytest.raises(ValueError, match='integrity'):
        module.validate_worker(output, expected, 410)


def test_complete_executed_sources_are_frozen():
    names = {str(p.relative_to(runner().ROOT)) for p in runner().source_files()}
    assert 'research/submission_campaign/deep.py' in names
    assert 'benchmarks/official_baselines/adapters/catch.py' in names
    assert 'benchmarks/official_baselines/registry.json' in names
