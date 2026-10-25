from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def api():
    from research.evaluation_protocols import campaign
    return campaign


def test_contaminated_prefix_is_rejected_without_normal_row_filtering(tmp_path):
    path = tmp_path / '001_SMD_id_1_Facility_tr_500_1st_700.csv'
    y = np.zeros(900, dtype=int)
    y[20] = 1
    pd.DataFrame({'a': np.arange(900), 'b': np.arange(900) * 2, 'Label': y}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='contaminated'):
        api().prepare_nasa(path)


def test_nasa_reserves_calibration_and_does_not_fit_test_statistics(tmp_path):
    path = tmp_path / '001_MSL_id_1_Sensor_tr_500_1st_700.csv'
    x = np.zeros((900, 2)); x[:400, 0] = np.arange(400)
    x[400:, 0] = 1e6; x[:, 1] = 3
    y = np.zeros(900, dtype=int); y[700:710] = 1
    pd.DataFrame({'a': x[:, 0], 'b': x[:, 1], 'Label': y}).to_csv(path, index=False)
    case = api().prepare_nasa(path)
    assert len(case['train']) == 400 and len(case['calibration']) == 100
    assert len(case['test']) == 400
    assert abs(case['train'][:, 0].mean()) < 1e-12
    assert case['calibration'][0, 0] > 8000
    assert case['metadata']['test_source_row_start'] == 500


def test_preprocessing_imputes_with_training_medians_only():
    train = np.array([[0., np.nan], [2., np.nan], [np.nan, np.nan]])
    cal = np.array([[np.nan, 100.]])
    test = np.array([[np.nan, -100.]])
    a, b, c, info = api().preprocess(train, cal, test)
    assert a.shape == (3, 1)
    np.testing.assert_allclose(b, [[0.]])
    np.testing.assert_allclose(c, [[0.]])
    assert info['retained_features'] == [0]


def test_group_overlap_removal_preserves_gap_entities():
    train = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [0.] * 4})
    cal = pd.DataFrame({'a': [2., 10.], 'b': [0., 0.]})
    test = pd.DataFrame({'a': [3., 20.], 'b': [0., 0.]})
    parts, info = api().remove_overlap([train, cal, test])
    assert parts[0].a.tolist() == [1., 4.]
    assert parts[1].a.tolist() == [2., 10.]
    assert info['removed_rows'] == [2, 0, 0]
    np.testing.assert_array_equal(api().gap_entities(parts[0].index.to_numpy(), np.zeros(2)), [0, 1])


def test_boundary_mask_is_applied_per_entity():
    mask = api().boundary_mask(np.array([0] * 6 + [1] * 7), left=2, right=1)
    np.testing.assert_array_equal(np.flatnonzero(mask), [2, 3, 4, 8, 9, 10, 11])


def test_low_fpr_metric_and_calibration_ties_are_explicit():
    result = api().evaluate(np.array([0, 0, 1, 1]), np.array([0., 1., 2., 3.]), np.array([1.] * 100))
    assert result['pauc_fpr_05'] == 1.
    assert result['threshold_05'] == 1.
    assert result['fpr_05'] == 0.
    assert result['recall_05'] == 1.
    assert result['calibration_fpr_05'] == 0.


def test_single_class_evaluation_is_rejected():
    with pytest.raises(ValueError, match='both classes'):
        api().evaluate(np.zeros(4), np.arange(4.), np.arange(20.))


@pytest.mark.parametrize('method', ['dynsphere', 'dynsphere_euclidean', 'dynsphere_one_regime',
                                  'dynsphere_euclidean_one_regime', 'pca_reconstruction',
                                  'kmeans_distance', 'linear_ridge_ar1'])
def test_fitted_models_match_existing_single_entity_controls(method):
    from benchmarks.official_baselines.local_controls import LOCAL_SCORERS
    rng = np.random.default_rng(4)
    train, test = rng.normal(size=(180, 4)), rng.normal(size=(150, 4))
    expected, _ = LOCAL_SCORERS[method](train, test, 37)
    model = api().FittedControl(method, 37).fit(train, np.zeros(len(train), dtype=int))
    actual = model.score(test, np.zeros(len(test), dtype=int))
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_ar1_never_learns_or_scores_across_entity_boundaries():
    x = np.array([[0.], [1.], [2.], [100.], [101.], [102.]])
    entities = np.array([0, 0, 0, 1, 1, 1])
    model = api().FittedControl('linear_ridge_ar1', 37).fit(x, entities)
    scores = model.score(x, entities)
    assert scores[0] == 0 and scores[3] == 0
    assert scores[4] < .01


def test_manifest_changes_when_source_changes(tmp_path):
    source = tmp_path / 'source.csv'; source.write_text('a,b\n1,2\n')
    entries = [{'case_id': 'toy', 'source_family': 'MSL', 'kind': 'nasa', 'paths': [str(source)]}]
    a = api().freeze(entries, tmp_path / 'out')
    source.write_text('a,b\n1,3\n')
    b = api().freeze(entries, tmp_path / 'out')
    assert a['manifest_id'] != b['manifest_id']


def test_corrupted_score_artifact_cannot_resume_as_complete(tmp_path):
    rng = np.random.default_rng(8)
    case = api()._case(rng.normal(size=(100, 3)), rng.normal(size=(100, 3)),
                       rng.normal(size=(400, 3)), np.r_[np.zeros(250), np.ones(150)],
                       {'case_id': 'toy', 'source_family': 'MSL'})
    row = api().run_one(case, 'pca_reconstruction', 37, tmp_path, 'manifest')
    assert row['status'] == 'complete'
    artifact = Path(row['run_directory']) / 'scores.npz'
    artifact.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        api().run_one(case, 'pca_reconstruction', 37, tmp_path, 'manifest')


def test_event_summary_keeps_entity_boundaries_and_excludes_trimmed_events():
    y = np.array([0, 1, 1, 0, 1, 1, 0, 1])
    pred = np.array([0, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
    valid = np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    result = api().event_summary(y, pred, valid, np.array([0] * 4 + [1] * 4))
    assert result['eligible_events'] == 2
    assert result['event_recall'] == .5
    assert result['mean_detected_delay_samples'] == 1.


def summary_row(method, case, value, seed=37):
    from research.evaluation_protocols.run import METRICS
    return {'run_identity': f'{case}-{method}-{seed}', 'case_id': case, 'source_family': 'MSL',
            'method_id': method, 'seed': seed, 'status': 'complete',
            'metrics': {k: value for k in METRICS}, 'metadata': {'protocol': 'toy'},
            'artifact_sha256': 'toy', 'run_directory': '/tmp/toy'}


def test_missing_reference_still_writes_failure_and_completion_reports(tmp_path):
    from research.evaluation_protocols.run import summarize
    records = [summary_row('pca_reconstruction', 'a', 1.),
               {'case_id': 'a', 'source_family': 'MSL', 'method_id': 'dynsphere_euclidean_one_regime',
                'seed': 37, 'status': 'failed', 'error': 'fixture'}]
    result = summarize(records, tmp_path, {'manifest_id': 'toy'})
    assert result['failed'] == 1
    assert (tmp_path / 'noncomplete.json').is_file()
    assert (tmp_path / 'completion.json').is_file()
    assert result['matched_records'] == 0


def test_rankings_use_shared_successful_seed_case_support(tmp_path):
    from research.evaluation_protocols.run import summarize
    records = [summary_row(method, 'a', .9 if method == 'dynsphere_euclidean_one_regime' else 1.)
               for method in api().METHODS]
    records += [summary_row(method, 'b', 0.) for method in api().METHODS
                if method != 'dynsphere_euclidean_one_regime']
    records += [{'case_id': 'b', 'source_family': 'MSL', 'method_id': 'dynsphere_euclidean_one_regime',
                 'seed': 37, 'status': 'failed', 'error': 'fixture'}]
    summarize(records, tmp_path, {'manifest_id': 'toy'})
    f = pd.read_csv(tmp_path / 'per_family.csv').set_index('method_id')
    assert f.loc['pca_reconstruction', 'auprc'] == 1.
    assert f.loc['dynsphere_euclidean_one_regime', 'auprc'] == .9
    assert set(f.case_count) == {1}
    available = pd.read_csv(tmp_path / 'per_family_available.csv').set_index('method_id')
    assert available.loc['pca_reconstruction', 'case_count'] == 2
    assert available.loc['dynsphere_euclidean_one_regime', 'case_count'] == 1
