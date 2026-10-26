import importlib
import json
import sys
import time

import numpy as np
import pytest


def module(name='run'):
    try:
        return importlib.import_module('research.quality_cost_extension.' + name)
    except ModuleNotFoundError:
        pytest.fail('The isolated four-method CPU extension is missing')


class ToyModel:
    def __init__(self, value):
        self.value = value

    def eval(self):
        return self


def toy_backend(train, test, configuration, seed):
    window = configuration['window_size']
    model = ToyModel(float(train.sum()) + seed + window)
    model.eval()
    values = np.asarray(test) + model.value
    return values


def test_statement_split_preserves_fitting_and_scoring_expressions():
    m = module('trials')
    original = toy_backend(np.array([2.]), np.array([3.]), {'window_size': 4}, 7)
    before, fit, score = m.split_backend(toy_backend, 'window')
    scope = dict(toy_backend.__globals__, train=np.array([2.]), test=np.array([3.]),
                 configuration={'window_size': 4}, seed=7)
    exec(before, scope)
    exec(fit, scope)
    assert '_output' not in scope
    exec(score, scope)
    np.testing.assert_array_equal(scope['_output'], original)


def test_extension_uses_exact_original_input_bundle_and_sixty_jobs():
    m = module()
    doc = m.new_manifest()
    assert doc['original_manifest_id']=='c2899e2c55b383ded3bd127821821844cc63f8b8131ee509830b7759b7a94398'
    jobs = list(m.jobs(doc))
    assert len(jobs)==60
    assert {x[0]['source_family'] for x in jobs}=={'BATADAL','HAI-20.07','MSL','SMAP','SMD'}
    assert {x[1] for x in jobs}=={'dagmm','omni_anomaly_tsb_ad','anomaly_transformer','tranad'}
    assert {x[2] for x in jobs}=={37,38,39}
    for case, _, _ in jobs:
        assert m.input_directory(doc,case).parent.name=='inputs'


def test_serialization_replays_stochastic_scores():
    m=module('trials')
    np.random.seed(7)
    state=m.random_state()
    expected=np.random.normal(size=410)
    size=m.pickle_roundtrip({'coefficient':2},lambda _:np.random.normal(size=410),expected,state)
    assert size>0


def test_reference_comparison_resets_numpy_randomness():
    m=module('trials')
    np.random.seed(19)
    scores=np.random.normal(size=410)
    m.verify_reference(scores,lambda:np.random.normal(size=410),19)
    with pytest.raises(AssertionError):
        m.verify_reference(scores+1,lambda:np.random.normal(size=410),19)


def test_worker_validation_rejects_changed_score_and_identity(tmp_path):
    m=module()
    expected={'case_id':'c','method_id':'tranad','seed':37,'manifest_id':'test'}
    np.save(tmp_path/'scores.npy',np.arange(410,dtype=float))
    result=dict(expected,status='complete',scores_sha256=m.digest(tmp_path/'scores.npy'))
    (tmp_path/'run.json').write_text(json.dumps(result))
    assert m.validate_worker(tmp_path,expected,410)[0]==result
    with pytest.raises(ValueError):
        m.validate_worker(tmp_path,dict(expected,seed=38),410)
    np.save(tmp_path/'scores.npy',np.zeros(410))
    with pytest.raises(ValueError):
        m.validate_worker(tmp_path,expected,410)


def test_timeout_stops_worker_before_returning(tmp_path):
    m=module()
    marker=tmp_path/'late'
    code='import time; from pathlib import Path; time.sleep(1); Path('+repr(str(marker))+').touch()'
    with pytest.raises(TimeoutError):
        m.execute([sys.executable,'-c',code],tmp_path,0.05,{})
    time.sleep(1.1)
    assert not marker.exists()


def test_existing_outcomes_and_failed_groups_are_retained(tmp_path):
    m=module()
    p=tmp_path/'result.json'
    m.write_new(p,{'status':'timeout'})
    with pytest.raises(FileExistsError):
        m.write_new(p,{'status':'complete'})
    assert json.loads(p.read_text())['status']=='timeout'
    rows=[dict(status='timeout',method_id='dagmm',case_id='c',source_family='HAI-20.07',seed=37)]
    frame=m.summarize(rows)
    assert frame.iloc[0].completed_seeds==0
    assert np.isnan(frame.iloc[0].fit_seconds)


def test_resume_recovers_completed_worker_without_retraining(tmp_path):
    m=module()
    expected={'case_id':'c','method_id':'tranad','seed':37,'manifest_id':'test'}
    attempt=tmp_path/'attempt-0001'
    attempt.mkdir()
    np.save(attempt/'scores.npy',np.arange(410,dtype=float))
    (attempt/'run.json').write_text(json.dumps(dict(expected,status='complete',scores_sha256=m.digest(attempt/'scores.npy'))))
    path,recovered=m.recover_attempt(tmp_path,expected,410)
    assert path==attempt and recovered
    assert len(list(tmp_path.glob('attempt-*')))==1


def test_resume_preserves_unfinished_attempt_without_automatic_retry(tmp_path):
    m=module()
    (tmp_path/'attempt-0001').mkdir()
    with pytest.raises(InterruptedError):
        m.recover_attempt(tmp_path,{},410)
    assert not (tmp_path/'attempt-0002').exists()


def test_shared_hardware_lock_rejects_second_controller(tmp_path,monkeypatch):
    m=module('trials')
    monkeypatch.setattr(m,'LOCK_ROOT',tmp_path)
    with m.hardware_lock('controller'):
        with pytest.raises(BlockingIOError):
            with m.hardware_lock('controller'):
                pytest.fail('Concurrent hardware controller acquired the lock')


def test_runtime_inventory_includes_transitive_packages():
    actual=module('trials').versions()
    assert actual['installed_distributions']['pytest']
    assert actual['installed_distributions']['joblib']


def test_transformer_restore_scope_contains_original_return_metadata():
    m=module('trials')
    scope=m.restoration_metadata('anomaly_transformer',{'epochs':10})
    # The unmodified adapter returns this value after scoring.
    return_program=compile("result = {'cpu_compatibility': cpu_compatibility, 'epochs': epochs}",'<adapter-return>','exec')
    exec(return_program,scope)
    assert scope['result']=={'cpu_compatibility':True,'epochs':10}
