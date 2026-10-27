import importlib
import json

import numpy as np
import pytest


def module(name):
    try:
        return importlib.import_module('research.explanation_study.'+name)
    except ModuleNotFoundError:
        pytest.fail('Explanation execution implementation is missing')


def test_rebinding_changes_test_input_without_mutating_original_closure():
    def factory():
        test=np.array([1.,2.]); scale=3.
        return lambda fitted:scale*test.sum()+fitted
    original=factory()
    rebound=module('models').rebind_test(original,np.array([4.,5.]))
    assert rebound(2)==29 and original(2)==11
    with pytest.raises(ValueError,match='test'):
        module('models').rebind_test(lambda fitted:fitted,np.ones(2))


def test_rng_replay_makes_repeated_stochastic_scoring_identical():
    m=module('models'); rng=m.get_rng()
    scorer=m.ReplayedScorer(lambda x:np.random.normal(size=len(x)),rng)
    a=scorer(np.zeros((512,2))); b=scorer(np.zeros((512,2)))
    np.testing.assert_array_equal(a,b)
    assert scorer.calls==2


def test_real_local_fitted_capture_matches_unmodified_original_trial(tmp_path):
    m=module('models');rng=np.random.default_rng(2)
    for name,n in [('train',400),('calibration',100),('test',512)]:
        np.save(tmp_path/(name+'.npy'),rng.normal(size=(n,3)))
        np.save(tmp_path/(name+'_entities.npy'),np.zeros(n,dtype=int))
    fitted,scores,evidence=m.fit_original('pca_reconstruction',37,tmp_path,
                                       {'fixed_local_control':'pca_reconstruction'},tmp_path)
    test=np.load(tmp_path/'test.npy')
    np.testing.assert_allclose(fitted.score(test),scores,rtol=0,atol=0)
    assert fitted.model is not None and evidence['serialization_verified']


def gate_fixture():
    methods=['dynsphere_euclidean_one_regime','pca_reconstruction','tranad']
    cases=[{'case_id':f,'source_family':f} for f in ['BATADAL','HAI-20.07','MSL','SMAP','SMD']]
    rows=[dict(method_id=m,case_id=c['case_id'],source_family=c['source_family'],seed=37,
               status='complete',technical_verified=True) for c in cases for m in methods]
    return methods,cases,rows


def test_pilot_gate_blocks_missing_or_invalid_runs_and_focal_coverage():
    m=module('run'); methods,cases,rows=gate_fixture()
    assert m.technical_gate(rows,methods,cases)['passed']
    assert not m.technical_gate(rows[:-1],methods,cases)['passed']
    rows[0]['status']='timeout'
    assert not m.technical_gate(rows,methods,cases)['passed']
    rows[0]['status']='failed'
    assert not m.technical_gate(rows,methods,cases)['passed']


def test_pilot_gate_retains_resource_timeouts_without_selecting_winners():
    m=module('run');methods,cases,rows=gate_fixture()
    rows[-1].update(status='timeout',error='resource limit')
    assert m.technical_gate(rows,methods,cases)['passed']
    rows[0]['technical_verified']=False
    assert not m.technical_gate(rows,methods,cases)['passed']


def test_exclusive_records_and_interrupted_attempts_are_not_overwritten(tmp_path):
    m=module('run'); path=tmp_path/'record.json'
    m.write_new(path,{'original':True})
    with pytest.raises(FileExistsError):
        m.write_new(path,{'original':False})
    assert json.loads(path.read_text())=={'original':True}
    directory=tmp_path/'run';directory.mkdir();(directory/'started.json').write_text('{}')
    assert m.existing_state(directory)== 'interrupted'


def test_binding_validation_rejects_changed_file(tmp_path):
    m=module('run');path=tmp_path/'source';path.write_text('a')
    doc={str(path):m.digest(path)}
    m.verify_files(doc)
    path.write_text('b')
    with pytest.raises(ValueError,match='Changed'):
        m.verify_files(doc)


def test_paired_aggregation_requires_identical_complete_seeds():
    m=module('run')
    rows=[dict(method_id=method,case_id=family,source_family=family,seed=seed,value=value)
          for method,value in [('e1',.3),('baseline',.6)]
          for family in ['A','B'] for seed in [37,38,39]]
    result=m.paired_metric(rows,'value','e1','baseline',[37,38,39],['A','B'])
    assert result['families']==['A','B'] and result['e1']==pytest.approx(.3)
    rows[-1]['value']=None
    result=m.paired_metric(rows,'value','e1','baseline',[37,38,39],['A','B'])
    assert result['families']==['A'] and not result['full_family_support']


def test_native_decomposition_checks_propagation_in_addition_to_sum():
    from pendulum_detector import MultiConceptEuclideanDetector
    m=module('models');rng=np.random.default_rng(3)
    x=rng.normal(size=(512,3));model=MultiConceptEuclideanDetector(n_concepts=1,latent_dim=3).fit(x,np.zeros(512))
    evidence=m.native_diagnostics(model,x)
    assert evidence['accounting_verified'] and evidence['propagation_verified']
    assert abs(sum(evidence['components'].values())-evidence['target'])<1e-8


def test_interrupted_json_write_does_not_publish_truncated_terminal_record(tmp_path,monkeypatch):
    m=module('run');path=tmp_path/'result.json'
    def broken(value,stream,**kwargs):
        stream.write('{partial');raise OSError('interrupted write')
    monkeypatch.setattr(m.json,'dump',broken)
    with pytest.raises(OSError):m.write_new(path,{'value':1})
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_recovery_refuses_to_touch_a_live_worker(tmp_path):
    import os
    m=module('run');path=tmp_path/'worker-owner.json'
    path.write_text(json.dumps({'pid':os.getpid(),'token':'test-owner'}))
    with pytest.raises(RuntimeError,match='live'):
        m.require_stopped(path)
    assert json.loads(path.read_text())['pid']==os.getpid()


def test_event_summary_retains_detected_and_stable_subsets_and_counts():
    m=module('run')
    events=[dict(edit_score_increase=b,score_stable=s,explanation_seconds=1.,
                 resolutions={axis:dict(infidelity=v,ap=v,precision_at_2=v,stability=v,
                  random_infidelity=v,random_ap=v,random_precision_at_2=v,random_stability=v,
                  second_reference_infidelity=v) for axis in ('sensor','time')})
            for b,s,v in [(True,True,.2),(False,False,.8)]]
    row=m.summarize_events(events)
    assert row['sensor_ap']==pytest.approx(.5)
    assert row['sensor_ap_score_increase_subset']==pytest.approx(.2)
    assert row['sensor_stability_stable_score_subset']==pytest.approx(.2)
    assert row['score_increase_events']==row['stable_score_events']==1
    assert row['metric_coverage']['sensor_ap']=={'available':2,'total':2}


def test_stored_event_metric_tampering_is_rejected():
    c=importlib.import_module('research.explanation_study.core');m=module('run')
    rng=np.random.default_rng(8);clean=rng.normal(size=(512,3));x,truth=c.inject(clean,np.ones(3),'mean',2)
    result=c.explain_event(lambda a:float((a[240:272]**2).sum()),x,clean,truth,
                          [np.zeros_like(x),np.ones_like(x)],np.ones(3),9)
    m.verify_event(result)
    result['resolutions']['sensor']['infidelity']=99.
    with pytest.raises(ValueError,match='metric'):m.verify_event(result)


def test_valid_worker_terminal_is_recovered_when_parent_record_is_missing(tmp_path):
    m=module('run');directory=tmp_path/'trial';(directory/'worker').mkdir(parents=True)
    expected={'manifest_id':'frozen','method_id':'tranad','case_id':'A','source_family':'A','seed':37}
    record=dict(expected,status='timeout',technical_verified=False,phase='explanation')
    m.write_new(directory/'started.json',expected)
    m.write_new(directory/'worker/run.json',record)
    assert m.recover_terminal(directory,expected)==record
    assert not (directory/'result.json').exists()


def test_process_timeout_stops_descendant_before_it_writes(tmp_path):
    import sys,time
    m=module('run');marker=tmp_path/'descendant-alive'
    child="import time;from pathlib import Path;time.sleep(.8);Path("+repr(str(marker))+").write_text('alive')"
    parent="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);time.sleep(30)"
    with pytest.raises(TimeoutError):m.execute([sys.executable,'-c',parent],tmp_path,.2,{})
    time.sleep(1.)
    assert not marker.exists()
    assert (tmp_path/'worker-owner.json').exists()


def test_all_method_aggregate_does_not_hide_metric_level_missingness():
    m=module('run')
    rows=[dict(method_id=method,case_id=f,source_family=f,seed=s,v=.5)
          for method in ['e1','a','b'] for f in ['F1','F2'] for s in [37,38,39]]
    rows[-1]['v']=None
    result=m.common_metric(rows,'v',['e1','a','b'],[37,38,39],['F1','F2'])
    assert result['families']==['F1'] and not result['full_family_support']
    assert set(result['means'])=={'e1','a','b'}
