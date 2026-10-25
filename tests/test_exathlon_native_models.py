"""Boundaries, independent validation, and resumable score evidence."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_windows_never_cross_a_segment_boundary():
    from research.exathlon_native.models import window_starts
    e = np.array([0, 0, 0, 1, 1, 1, 1])
    assert window_starts(e, 3, forecast=False).tolist() == [0, 3, 4]
    assert window_starts(e, 3, forecast=True).tolist() == [3]
    assert window_starts(np.array([], dtype=int), 100, True).tolist() == []
    with pytest.raises(ValueError, match='contiguous'):
        window_starts(np.array([0, 1, 0]), 2, False)


def test_independent_calibration_split_is_exact_and_scoped():
    from research.exathlon_native.models import reserved_partition
    original = lambda *args: ('old', 'split')
    module = SimpleNamespace(train_val_split=original)
    train = pd.DataFrame({'x': np.arange(13.)})
    cal = pd.DataFrame({'x': np.arange(7.)+100})
    with reserved_partition(module, cal):
        a, b = module.train_val_split(train, .8, None)
        assert a.x.tolist() == list(np.arange(13.))
        assert b.x.tolist() == list(np.arange(7.)+100)
    assert module.train_val_split is original


def test_padding_is_unobserved_and_cannot_satisfy_common_support():
    from research.exathlon_native.models import align_observed, require_support
    scores = align_observed(7, np.array([2., 3., 4.]), warmup=3)
    assert np.isnan(scores[:3]).all() and np.isnan(scores[-1])
    require_support(scores, np.array([False, False, False, True, True, True, False]))
    with pytest.raises(ValueError, match='observed'):
        require_support(scores, np.ones(7, dtype=bool))


def test_dry_run_counts_fits_and_scored_records_separately():
    from research.exathlon_native.run import describe
    groups = [{'application': a, 'tests': [{} for _ in range(n)]} for a, n in [(1, 1), (2, 1), (4, 1), (5, 2)]]
    info = describe({'manifest_id': 'x', 'prepared': {'cohort': groups}, 'seeds': [37, 38, 39]})
    assert info['planned_fits'] == 108
    assert info['planned_score_records'] == 150
    assert info['local_fits'] == 84 and info['deep_fits'] == 24


def test_resume_rejects_corrupt_artifacts_and_wrong_job(tmp_path):
    from research.exathlon_native.run import verify_artifacts
    from research.evaluation_protocols.campaign import digest
    artifact = tmp_path/'scores.npz'
    np.savez(artifact, scores=np.arange(3.))
    job = {'application': 1, 'method': 'catch', 'seed': 37}
    record = {'manifest_id': 'frozen', 'job': job, 'status': 'complete',
              'artifacts': {'scores.npz': digest(artifact)}}
    verify_artifacts(record, tmp_path, 'frozen', job)
    with pytest.raises(ValueError):
        verify_artifacts(record, tmp_path, 'frozen', {**job, 'seed': 38})
    artifact.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='artifact'):
        verify_artifacts(record, tmp_path, 'frozen', job)


def test_frozen_input_drift_is_detected(tmp_path):
    from research.exathlon_native import protocol as p
    from research.evaluation_protocols.campaign import _identity, digest
    (tmp_path/'inputs').mkdir()
    path = tmp_path/'inputs/x'
    path.write_bytes(b'original')
    doc = {'implementation': p.code_evidence(), 'inputs': {'inputs/x': digest(path)}}
    doc['manifest_id'] = _identity(doc)
    (tmp_path/'manifest.json').write_text(json.dumps(doc))
    assert p.validate(tmp_path)['manifest_id'] == doc['manifest_id']
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='input'):
        p.validate(tmp_path)


def test_synthetic_catch_check_has_enough_batches_for_upstream_update_schedule():
    from research.exathlon_native.run import synthetic_inputs
    from research.exathlon_native.models import window_starts
    train, entities, cal, ce = synthetic_inputs()
    assert int(np.ceil(len(window_starts(entities, 100, False))/128)) >= 10
    assert len(cal) != int(len(train)/4)


def test_worker_timeout_is_visible_in_campaign_completion(tmp_path):
    from research.exathlon_native.run import record_worker_exit, fit_identity
    doc = {'manifest_id': 'frozen'}
    job = {'application': 1, 'method': 'catch', 'seed': 37}
    record_worker_exit(tmp_path, doc, job, {'timeout_seconds': 86400})
    path = tmp_path/'fits'/fit_identity(doc, job)/'fit.json'
    assert json.loads(path.read_text())['status'] == 'timeout'


def test_compatibility_rejects_untested_upstream_source(tmp_path):
    from research.exathlon_native.protocol import code_evidence
    from research.exathlon_native.run import require_compatibility
    for method in ('catch', 'mtad_gat'):
        folder = tmp_path/'compatibility'/method
        folder.mkdir(parents=True)
        (folder/'wrong.json').write_text(json.dumps({'implementation': code_evidence(),
            'status': 'passed', 'upstream': {'revision': 'untested-source'}}))
    with pytest.raises(ValueError, match='compatibility'):
        require_compatibility(tmp_path)
