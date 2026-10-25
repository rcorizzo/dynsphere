import numpy as np
import pytest

from benchmarks.official_baselines import extension_components as study
from benchmarks.official_baselines.local_controls import DYNSPHERE_CONFIGURATION
from pendulum_detector import MultiConceptEuclideanDetector


def test_e1_component_removal_matches_explicit_zero_weight_refit():
    rng = np.random.RandomState(7)
    train, test = rng.normal(size=(80, 3)), rng.normal(size=(20, 3))
    configuration = {**DYNSPHERE_CONFIGURATION, 'n_concepts': 1, 'random_state': 37}
    model = MultiConceptEuclideanDetector(**configuration).fit(train, np.zeros(80))
    scores = study.score_configurations(model, test)
    zero_raw = MultiConceptEuclideanDetector(**{
        **configuration, 'raw_dynamic_weight': 0}).fit(train, np.zeros(80))
    np.testing.assert_allclose(scores['without_raw_dynamics'],
                               zero_raw.decision_function(test, np.zeros(20)))
    no_propagation = MultiConceptEuclideanDetector(**{
        **configuration, 'propagation_radius': 0}).fit(train, np.zeros(80))
    np.testing.assert_allclose(scores['radius_0'],
                               no_propagation.decision_function(test, np.zeros(20)))
    assert len(scores) == 11


def test_multiple_regimes_cannot_use_single_regime_component_subtraction():
    rng = np.random.RandomState(3)
    model = MultiConceptEuclideanDetector(n_concepts=2).fit(rng.normal(size=(80, 3)), np.zeros(80))
    with pytest.raises(ValueError, match='single'):
        study.score_configurations(model, rng.normal(size=(20, 3)))


def test_study_matches_frozen_full_scores_and_writes_all_configurations(tmp_path):
    import pandas as pd
    from tests.test_extension_campaign import _tsb_file
    from benchmarks.official_baselines.extension_campaign import run_pilot
    result = run_pilot([_tsb_file(tmp_path)], [], tmp_path/'original',
                        methods=['dynsphere_euclidean_one_regime'], seeds=[37])
    summary = study.main(['--input-csv', result['metrics_csv'],
                          '--runs-root', str(tmp_path/'original/runs'),
                          '--output-dir', str(tmp_path/'study')])
    assert summary['completed_reference_runs'] == 1
    frame = pd.read_csv(summary['per_run_csv'])
    assert len(frame) == 11
    assert not frame['manuscript_ready'].any()
