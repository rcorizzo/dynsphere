"""Checks for disjoint extension data and row-aligned temporal controls."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.official_baselines import extension_campaign as extension


def _tsb_file(tmp_path, *, contaminated=False):
    path = tmp_path / '174_Exathlon_id_1_Facility_tr_16_1st_19.csv'
    labels = [0] * 19 + [1] * 5
    if contaminated:
        labels[2] = 1
    pd.DataFrame({'sensor': np.arange(24), 'other': np.arange(24) * 2,
                  'Label': labels}).to_csv(path, index=False)
    return path


def test_suffix_split_excludes_training_rows_and_fits_scaling_on_prefix(tmp_path):
    case = extension.load_suffix_case(_tsb_file(tmp_path), 37)
    assert case.train.shape == (16, 2)
    assert case.test.shape == (8, 2)
    assert case.metadata['row_id_start'] == 16
    assert case.metadata['row_id_stop'] == 24
    np.testing.assert_array_equal(case.labels, [0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_allclose(case.train.mean(axis=0), [0, 0], atol=1e-12)
    assert case.test[0, 0] == pytest.approx((16 - 7.5) / np.sqrt(21.25))


def test_contaminated_prefix_is_rejected_without_deleting_time_steps(tmp_path):
    with pytest.raises(ValueError, match='contaminated'):
        extension.load_suffix_case(_tsb_file(tmp_path, contaminated=True), 37)


def test_missing_observations_are_rejected_without_collapsing_time(tmp_path):
    path = _tsb_file(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[18, 'sensor'] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='finite'):
        extension.load_suffix_case(path, 37)


def test_native_pair_excludes_timestamp_and_preserves_label_alignment(tmp_path):
    train = tmp_path / 'train_no_anomaly.csv'
    test = tmp_path / 'test.csv'
    pd.DataFrame({'timestamp': range(20), 'value-0': range(20),
                  'value-1': range(20), 'is_anomaly': [0]*20}).to_csv(train, index=False)
    pd.DataFrame({'timestamp': range(4), 'value-0': [1, 2, 3, 4],
                  'value-1': [1, 2, 3, 4], 'is_anomaly': [0, 1, 1, 0]}).to_csv(test, index=False)
    case = extension.load_native_case(train, test, 'GutenTAG', 'frequency-g101', 37)
    assert case.train.shape == (20, 2)
    np.testing.assert_array_equal(case.labels, [0, 1, 1, 0])
    assert case.metadata['feature_names'] == ['value-0', 'value-1']
    with pytest.raises(ValueError, match='distinct'):
        extension.load_native_case(train, train, 'GutenTAG', 'same', 37)


def test_ar1_residual_is_assigned_to_the_predicted_row():
    from benchmarks.official_baselines.local_controls import score_linear_ridge_ar1
    # Constant normal dynamics predict zero. The spike occurs only at row 2.
    scores, metadata = score_linear_ridge_ar1(np.zeros((20, 2)),
                                            np.array([[0, 0], [0, 0], [4, 0], [0, 0]]), 37)
    np.testing.assert_array_equal(scores, [0, 0, 8, 0])
    assert metadata['warmup_rows'] == 1


def test_case_selection_uses_manifest_order_and_rejects_duplicate_files(tmp_path):
    path = _tsb_file(tmp_path)
    names = [path.name]
    assert extension.select_tsb_paths(tmp_path, names, ['Exathlon'], 1) == [path]
    (tmp_path / 'duplicate').mkdir()
    (tmp_path / 'duplicate' / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='Duplicate'):
        extension.select_tsb_paths(tmp_path, names, ['Exathlon'], 1)


def test_cached_metrics_use_shared_warmup_and_reject_nonfinite_scores():
    metrics = extension.point_metrics(np.array([0, 0, 1, 1]), np.array([99, 0, 1, 2]))
    assert metrics['auprc'] == 1
    assert metrics['evaluated_rows'] == 3
    with pytest.raises(ValueError, match='finite'):
        extension.point_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, np.nan, 2]))


def test_generation_design_uses_independent_replicates_and_normal_training():
    from benchmarks.official_baselines.prepare_gutentag_extension import generation_config
    config = generation_config(length=4000)
    assert len(config['timeseries']) == 6
    for case in config['timeseries']:
        assert case['semi-supervised'] is True
        assert len(case['base-oscillations']) == 3
        assert case['anomalies'][0]['exact-position'] >= 1000


def test_end_to_end_pilot_records_scores_protocol_and_exact_resume(tmp_path):
    path = _tsb_file(tmp_path)
    summary = extension.run_pilot([path], [], tmp_path / 'results',
                                  methods=['pca_reconstruction'], seeds=[37])
    assert summary['outcomes'] == {'complete': 1}
    result = pd.read_csv(Path(summary['metrics_csv']))
    assert len(result) == 1
    record = next((tmp_path / 'results' / 'runs').rglob('run.json'))
    payload = json.loads(record.read_text())
    assert payload['case_id'].startswith('TSB-AD-M-disjoint-v1/')
    assert payload['hyperparameters']['extension_source_digest']
    resumed = extension.run_pilot([path], [], tmp_path / 'results',
                                  methods=['pca_reconstruction'], seeds=[37])
    assert resumed['resumed_count'] == 1


def test_method_order_does_not_change_scheduler_partition(tmp_path):
    path = _tsb_file(tmp_path)
    first = extension.build_jobs([path], [], tmp_path/'runs',
                                 ['dynsphere', 'pca_reconstruction'], [37])
    second = extension.build_jobs([path], [], tmp_path/'runs',
                                  ['pca_reconstruction', 'dynsphere'], [37])
    assert [j.implementation.method_id for j in first] == [j.implementation.method_id for j in second]


def test_label_correction_cannot_resume_stale_private_labels(tmp_path):
    path = _tsb_file(tmp_path)
    root = tmp_path / 'results'
    extension.run_pilot([path], [], root, methods=['pca_reconstruction'], seeds=[37])
    frame = pd.read_csv(path)
    frame.loc[17, 'Label'] = 1
    frame.to_csv(path, index=False)
    corrected = extension.run_pilot([path], [], root, methods=['pca_reconstruction'], seeds=[37])
    assert corrected['resumed_count'] == 0


def test_generated_manifest_rejects_modified_data_and_wrong_generator_version(tmp_path):
    train = tmp_path/'train.csv'
    test = tmp_path/'test.csv'
    train.write_text('timestamp,value-0,value-1,is_anomaly\n0,1,2,0\n')
    test.write_text('timestamp,value-0,value-1,is_anomaly\n0,1,2,1\n')
    manifest = tmp_path/'native_manifest.json'
    payload = {'generator': 'official TimeEval/GutenTAG', 'version': '1.5.0', 'cases': [{
        'family': 'GutenTAG', 'case_name': 'frequency-g101',
        'train': 'train.csv', 'test': 'test.csv',
        'train_sha256': extension.file_digest(train), 'test_sha256': extension.file_digest(test)}]}
    manifest.write_text(json.dumps(payload))
    assert len(extension.read_native_pairs(manifest, ['GutenTAG'])) == 1
    test.write_text(test.read_text().replace(',1,2,1', ',4,2,1'))
    with pytest.raises(ValueError, match='hash'):
        extension.read_native_pairs(manifest, ['GutenTAG'])
    payload['version'] = '0.0.0'
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='version'):
        extension.read_native_pairs(manifest, ['GutenTAG'])


def test_official_vus_does_not_mark_the_pilot_manuscript_ready(tmp_path):
    pytest.importorskip('vus')
    result = extension.run_pilot([_tsb_file(tmp_path)], [], tmp_path/'results',
                                 methods=['pca_reconstruction'], seeds=[37], with_vus=True)
    assert result['outcomes'] == {'complete': 1}
    frame = pd.read_csv(result['metrics_csv'])
    assert not frame['manuscript_ready'].any()


def test_audited_profile_reaches_external_adapter_configuration(tmp_path):
    jobs = extension.build_jobs([_tsb_file(tmp_path)], [], tmp_path/'results',
                                ['tranad', 'anomaly_transformer'], [37],
                                baseline_profile='audited-v2')
    by_method = {j.implementation.method_id: j for j in jobs}
    tranad_job = by_method['tranad']
    assert tranad_job.implementation.default_hyperparameters['pipeline_revision'] == 'audited-v2'
    at = by_method['anomaly_transformer'].implementation.default_hyperparameters
    assert (at['epochs'], at['d_model'], at['n_heads']) == (10, 512, 8)
    assert at['training_revision'] == 'released-budget-v2'
    assert tranad_job.hyperparameters['baseline_profile'] == 'audited-v2'


def test_unknown_baseline_profile_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='profile'):
        extension.build_jobs([_tsb_file(tmp_path)], [], tmp_path/'results',
                              ['tranad'], [37], baseline_profile='unknown')

