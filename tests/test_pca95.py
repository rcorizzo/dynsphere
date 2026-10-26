"""Protect fitting-only dimension selection and isolated sensitivity execution."""
import importlib
import json

import numpy as np
import pandas as pd
import pytest


def protocol():
    try:
        return importlib.import_module('research.pca95.run')
    except ModuleNotFoundError:
        pytest.fail('PCA-95 sensitivity runner is not implemented')


def test_dimension_reaches_threshold_preserves_floor_and_caps_features():
    module = protocol()
    assert module.choose_dimension([.30, .25, .20, .10, .05, .04, .03, .03]) == 7
    assert module.choose_dimension([1., 0., 0., 0., 0., 0.]) == 5
    assert module.choose_dimension([.8, .2, 0.]) == 3
    with pytest.raises(ValueError):
        module.choose_dimension([np.nan, 1.])


def test_projection_uses_fitting_statistics_and_leaves_input_arrays_unchanged():
    rng = np.random.RandomState(7)
    train = rng.normal(size=(80, 8))
    original = train.copy()
    cal, test = rng.normal(size=(25, 8)), rng.normal(size=(30, 8))
    arrays, metadata = protocol().fit_projection(train, cal, test)
    changed, second = protocol().fit_projection(train, cal + 1000, test - 1000)
    np.testing.assert_array_equal(train, original)
    np.testing.assert_allclose(arrays[0], changed[0])
    assert metadata == second
    assert arrays[0].shape[1] == metadata['q']
    assert metadata['retained_variance'] >= .95
    assert not np.allclose(arrays[1], changed[1])


def test_dynsphere_only_changes_latent_dimension_and_preserves_raw_dynamics():
    module = protocol()
    model = module.dynsphere_model('dynsphere_euclidean_one_regime', 37, 7)
    assert model.latent_dim == 7 and model.n_concepts == 1
    assert model.dynamic_weight == 1.8 and model.raw_dynamic_weight == 1.4
    assert model.propagation_radius == 8 and model.ridge_alpha == 1.
    rng = np.random.RandomState(9)
    train = rng.normal(size=(120, 8))
    model.fit(train, np.zeros(len(train), dtype=int))
    assert model.raw_models_[0].n_features_in_ == 16
    assert model.models_[0].n_features_in_ == 14
    assert np.isfinite(model.decision_function(train, np.zeros(len(train), dtype=int))).all()


def test_execution_waits_for_complete_timing_proof(tmp_path):
    proof = tmp_path / 'completion.json'
    with pytest.raises(RuntimeError, match='timing'):
        protocol().timing_guard(proof)
    proof.write_text(json.dumps(dict(expected=135, complete=105, failed=0,
                                    manifest_id=protocol().TIMING_ID)))
    with pytest.raises(RuntimeError, match='timing'):
        protocol().timing_guard(proof)
    proof.write_text(json.dumps(dict(expected=135, complete=134, failed=1,
                                    manifest_id=protocol().TIMING_ID)))
    protocol().timing_guard(proof)


def test_matrix_has_all_approved_variants_without_duplicate_pca_baseline():
    module = protocol()
    cases = [{'case_id': str(i)} for i in range(60)]
    jobs = module.jobs_for(cases, module.METHODS, (37, 38, 39))
    assert len(jobs) == 2160
    assert len({(j['case_id'], j['method_id'], j['seed']) for j in jobs}) == 2160
    assert 'pca_reconstruction' not in {j['method_id'] for j in jobs}
    assert {j['method_id'] for j in jobs} == {
        'dynsphere', 'dynsphere_one_regime', 'dynsphere_euclidean',
        'dynsphere_euclidean_one_regime', 'kmeans_distance', 'linear_ridge_ar1',
        'catch', 'mtad_gat', 'dagmm', 'omni_anomaly_tsb_ad',
        'anomaly_transformer', 'tranad'}


def test_immutable_json_cannot_replace_a_record(tmp_path):
    path = tmp_path / 'record.json'
    protocol().write_new_json(path, {'value': 1})
    with pytest.raises(FileExistsError):
        protocol().write_new_json(path, {'value': 2})
    assert json.loads(path.read_text()) == {'value': 1}


def test_partial_pairing_does_not_produce_five_family_primary_mean():
    try:
        analysis = importlib.import_module('research.pca95.aggregate')
    except ModuleNotFoundError:
        pytest.fail('PCA-95 paired aggregation is not implemented')
    variants = pd.DataFrame([dict(case_id='a', source_family='BATADAL', method_id='kmeans_distance',
                                 seed=37, status='complete', pauc_fpr_05=.7, auprc=.5, auroc=.8)])
    baseline = pd.DataFrame([dict(case_id='a', source_family='BATADAL', method_id='kmeans_distance',
                                 seed=37, pauc_fpr_05=.6, auprc=.4, auroc=.7)])
    pairs, summaries = analysis.paired_summary(variants, baseline, {'a'}, (37, 38, 39))
    assert pairs.iloc[0].delta_pauc_fpr_05 == pytest.approx(.1)
    assert summaries.iloc[0].paired_runs == 1
    assert not summaries.iloc[0].primary_complete
    assert np.isnan(summaries.iloc[0].equal_family_pauc_fpr_05)
    with pytest.raises(ValueError, match='Duplicate'):
        analysis.paired_summary(pd.concat([variants, variants]), baseline, {'a'}, (37, 38, 39))


def test_worker_measurement_rejects_changed_configuration_and_scores(tmp_path):
    module = protocol()
    scores = np.arange(410, dtype=float)
    np.save(tmp_path / 'scores.npy', scores)
    expected = dict(method_id='kmeans_distance', seed=37, q=7, configuration={'fixed': True})
    metadata = {'sha256': {'raw/train.npy': 'frozen'}}
    module.write_new_json(tmp_path / 'measurement.json', dict(expected, status='complete',
                          input_sha256=metadata['sha256'], scores_sha256=module.digest(tmp_path / 'scores.npy')))
    module.validate_measurement(tmp_path, expected, metadata)
    with pytest.raises(ValueError, match='identity'):
        module.validate_measurement(tmp_path, dict(expected, q=5), metadata)
    np.save(tmp_path / 'scores.npy', scores + 1)
    with pytest.raises(ValueError, match='integrity'):
        module.validate_measurement(tmp_path, expected, metadata)


def test_frozen_execution_binds_prepared_input_metadata(tmp_path):
    module = protocol()
    path = tmp_path / 'inputs.json'
    path.write_text('{"q": 7}')
    doc = {'input_metadata_sha256': {'case': module.digest(path)}}
    module.verify_input_binding(doc, 'case', path)
    path.write_text('{"q": 5}')
    with pytest.raises(ValueError, match='metadata'):
        module.verify_input_binding(doc, 'case', path)


def test_complete_pairing_averages_families_equally():
    analysis = importlib.import_module('research.pca95.aggregate')
    baseline, variants = [], []
    families = ['BATADAL', 'HAI-20.07', 'MSL', 'SMAP', 'SMD']
    for index, family in enumerate(families):
        for seed in (37, 38, 39):
            row = dict(case_id=family, source_family=family, method_id='kmeans_distance', seed=seed,
                       pauc_fpr_05=.5 + .1 * index, auprc=.3, auroc=.7)
            baseline.append(row)
            variants.append(dict(row, status='complete', pauc_fpr_05=row['pauc_fpr_05'] + .05))
    _, summary = analysis.paired_summary(pd.DataFrame(variants), pd.DataFrame(baseline),
                                         set(families), (37, 38, 39))
    assert summary.iloc[0].primary_complete
    assert summary.iloc[0].equal_family_pauc_fpr_05 == pytest.approx(.75)
    assert summary.iloc[0].equal_family_delta_pauc_fpr_05 == pytest.approx(.05)


def test_shared_preparation_waits_for_another_shard(tmp_path):
    import threading
    module = protocol()
    proof = tmp_path / 'timing.json'
    proof.write_text(json.dumps(dict(expected=135, complete=135, failed=0, manifest_id=module.TIMING_ID)))
    planning = dict(cases=[], sources={}, upstream={})
    planning['manifest_id'] = module._identity(planning)
    returned, errors = threading.Event(), []

    def prepare():
        try:
            module.prepare_execution(tmp_path, planning, proof)
        except Exception as error:
            errors.append(error)
        finally:
            returned.set()

    thread = threading.Thread(target=prepare)
    try:
        with module.run_lock(tmp_path):
            thread.start()
            assert not returned.wait(.05), 'A contending shard must wait for preparation'
    finally:
        thread.join(2.)
    assert returned.is_set() and not errors
    assert (tmp_path / 'execution-manifest.json').is_file()


def test_cpu_worker_rejects_changed_overlay_or_dependency_versions(tmp_path):
    module = protocol()
    from research.submission_recovery.run import overlay_files, versions
    source = tmp_path / 'compatibility.py'
    source.write_text('value = 1\n')
    evidence = dict(overlay_sha256=module._identity(overlay_files(tmp_path)), versions={'dagmm': versions()})
    doc = {'runtime_evidence': evidence}
    module.verify_runtime(doc, 'dagmm', tmp_path)
    source.write_text('value = 2\n')
    with pytest.raises(ValueError, match='overlay'):
        module.verify_runtime(doc, 'dagmm', tmp_path)
    source.write_text('value = 1\n')
    evidence['versions']['dagmm']['python'] = '0.invalid'
    with pytest.raises(ValueError, match='runtime'):
        module.verify_runtime(doc, 'dagmm', tmp_path)


def test_failed_requested_vus_cannot_mark_primary_comparison_complete():
    analysis = importlib.import_module('research.pca95.aggregate')
    families = ['BATADAL', 'HAI-20.07', 'MSL', 'SMAP', 'SMD']
    variants, baseline = [], []
    for family in families:
        for seed in (37, 38, 39):
            row = dict(case_id=family, source_family=family, method_id='catch', seed=seed,
                       pauc_fpr_05=.7, auprc=.5, auroc=.8)
            variants.append(dict(row, status='complete', vus_status='failed'))
            baseline.append(dict(row, vus_pr=.4, vus_roc=.8))
    _, summary = analysis.paired_summary(pd.DataFrame(variants), pd.DataFrame(baseline),
                                         set(families), (37, 38, 39),
                                         requested_metrics=('pauc_fpr_05', 'auprc', 'auroc', 'vus_pr', 'vus_roc'))
    row = summary.iloc[0]
    assert row.pointwise_complete and not row.primary_complete
    assert not row.complete_vus_pr and row.paired_vus_pr_runs == 0
