"""Coverage-aware pointwise, segment-temporal, and application aggregation."""
import numpy as np
import pandas as pd
import pytest


def test_calibration_quantile_uses_independent_normal_scores():
    from research.exathlon_native.evaluate import pointwise
    result = pointwise(np.array([0, 0, 1, 1]), np.array([0., 2., 3., 4.]), np.arange(100.))
    assert result['threshold_05'] == 95.
    assert result['calibration_fpr_05'] == .04
    assert result['fpr_05'] == 0. and result['recall_05'] == 0.
    assert result['pauc_fpr_05'] == 1.
    assert pointwise(np.zeros(4), np.arange(4.), np.array([]))['pauc_fpr_05'] is None


def test_vus_inputs_preserve_gaps_and_single_class_coverage():
    from research.exathlon_native.evaluate import temporal_inputs
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    e = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    mask = np.array([True, True, False, True, True, False, True, True])
    inputs, coverage = temporal_inputs(y, np.arange(8.), e, mask)
    assert [v['indices'].tolist() for v in inputs] == [[0, 1], [3, 4]]
    assert coverage == {'retained_rows': 6, 'vus_rows': 4, 'vus_segments': 2, 'single_class_rows': 2}


def test_temporal_support_holes_split_even_one_entity():
    from research.exathlon_native.evaluate import temporal_inputs
    y = np.array([0, 1, 0, 1, 0, 1])
    result, coverage = temporal_inputs(y, np.arange(6.), np.zeros(6), np.array([1, 1, 0, 0, 1, 1], dtype=bool))
    assert [v['indices'].tolist() for v in result] == [[0, 1], [4, 5]]


def test_equal_application_mean_and_three_seed_requirement():
    from research.exathlon_native.evaluate import summarize
    rows = [{'method_id': 'a', 'application': app, 'trace': trace, 'seed': seed,
             'pauc_fpr_05': value, 'auprc': value, 'auroc': value}
            for app, trace, value in [(1, 'one', 1.), (5, 'five-a', 0.), (5, 'five-b', 0.)]
            for seed in (37, 38, 39)]
    rows += [{'method_id': 'b', 'application': 1, 'trace': 'one', 'seed': 37,
              'pauc_fpr_05': 1., 'auprc': 1., 'auroc': 1.}]
    per_case, per_app, macro = summarize(pd.DataFrame(rows), [37, 38, 39])
    assert set(macro.method_id) == {'a'}
    assert macro.iloc[0].pauc_fpr_05 == .5
    assert len(per_case) == 3 and len(per_app) == 2


def test_temporal_metric_denominators_are_reported_separately():
    from research.exathlon_native.evaluate import summarize
    rows = [{'method_id': 'a', 'application': app, 'trace': str(app), 'seed': seed,
             'pauc_fpr_05': .7, 'vus_pr': value}
            for app, value in [(1, .8), (5, np.nan)] for seed in (37, 38, 39)]
    cases, apps, macro = summarize(pd.DataFrame(rows), [37, 38, 39])
    assert macro.iloc[0].application_count == 2
    assert macro.iloc[0].vus_pr_application_count == 1
    assert macro.iloc[0].vus_pr_execution_count == 1


def test_empty_campaign_still_exports_readiness(tmp_path):
    from research.exathlon_native.evaluate import export
    inventory = pd.DataFrame([{'status': 'pending'} for _ in range(150)])
    result = export(tmp_path, pd.DataFrame(), inventory,
                    {'manifest_id': 'x', 'methods': ['a'], 'seeds': [37, 38, 39]})
    assert result['matched_executions'] == 0 and result['matched_applications'] == 0
    assert result['outcomes'] == {'pending': 150}


def test_matched_temporal_means_require_the_same_cohort_for_every_method(tmp_path):
    from research.exathlon_native.evaluate import export
    rows = [{'method_id': m, 'application': a, 'trace': str(a), 'seed': s,
             'pauc_fpr_05': .7, 'vus_pr': .8 if (m == 'a') == (a == 1) else np.nan}
            for m in ('a', 'b') for a in (1, 5) for s in (37, 38, 39)]
    frame = pd.DataFrame(rows)
    inventory = frame[['method_id', 'application', 'trace', 'seed']].assign(status='complete')
    export(tmp_path, frame, inventory, {'manifest_id': 'x', 'methods': ['a', 'b'], 'seeds': [37, 38, 39]})
    result = pd.read_csv(tmp_path/'summary/equal_application_matched.csv')
    assert result.vus_pr.isna().all()


def test_temporal_records_identify_each_eligible_segment(tmp_path):
    pytest.importorskip('vus')
    from research.exathlon_native.evaluate import segment_vus
    y = np.r_[np.zeros(100), np.ones(50), np.zeros(100)].astype(int)
    scores = y+np.linspace(0, .1, len(y))
    result = segment_vus(y, scores, np.full(len(y), 7), np.ones(len(y), dtype=bool), tmp_path)
    assert result['segments'][0]['entity'] == 7
    assert result['segments'][0]['start'] == 0 and result['segments'][0]['stop'] == 250
    assert result['segments'][0]['rows'] == 250
