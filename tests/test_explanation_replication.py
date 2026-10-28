import copy
import json
from pathlib import Path

import pytest

from research.explanation_study import run as original


def test_replication_separates_mac_reference_without_mutation():
    from research.explanation_replication.run import derive
    parent = dict(manifest_id='linux-parent', methods=['dynsphere_euclidean_one_regime'],
                  cases=[dict(case_id='case',source_family='BATADAL')],seeds=[37,38,39],
                  original_records={'case|dynsphere_euclidean_one_regime|37':
                                    dict(status='complete',scores='/original.npy')},bindings={})
    before=copy.deepcopy(parent)
    child=derive(parent,Path('/tmp/new'),{})
    assert parent==before
    assert child['mac_reference_records']==parent['original_records']
    assert child['original_records']['case|dynsphere_euclidean_one_regime|37']['status']=='separate-host'
    assert child['manifest_id']!='linux-parent'
    assert len(child['trial_ids'])==3
    assert child['mac_score_equivalence_required'] is False


def test_stage_selection_and_exclusive_reservation(tmp_path):
    from research.explanation_replication.run import select_trial,reserve
    doc=dict(trial_ids=['a|m|37','a|m|38','a|m|39'],production_output=str(tmp_path))
    assert select_trial(doc,'pilot',1)=='a|m|37'
    assert select_trial(doc,'remaining',2)=='a|m|39'
    with pytest.raises(ValueError):select_trial(doc,'pilot',0)
    with pytest.raises(ValueError):select_trial(doc,'pilot',2)
    first=reserve(doc,'a|m|37',tmp_path)
    assert first.is_dir()
    with pytest.raises(FileExistsError):reserve(doc,'a|m|37',tmp_path)
    with pytest.raises(ValueError):reserve(doc,'../a|m|37',tmp_path)
    with pytest.raises(ValueError):reserve(doc,'a|m|37',tmp_path/'other')


def test_gate_rejects_missing_or_changed_terminal_evidence(tmp_path):
    from research.explanation_replication.run import validate_gate
    terminal=tmp_path/'result.json';original.write_new(terminal,{'status':'complete'})
    gate=dict(passed=True,manifest_id='id',evidence={str(terminal):original.digest(terminal)})
    doc=dict(manifest_id='id')
    validate_gate(doc,gate)
    with pytest.raises(ValueError):validate_gate(doc,dict(gate,evidence={}))
    terminal.write_text('{}')
    with pytest.raises(ValueError):validate_gate(doc,gate)


def test_replication_binding_rejects_mutation(tmp_path):
    from research.explanation_replication.run import derive,validate
    bound=tmp_path/'input';bound.write_bytes(b'normal')
    parent=dict(manifest_id='parent',methods=['m'],cases=[dict(case_id='c')],
                seeds=[37,38,39],original_records={},bindings={str(bound):original.digest(bound)})
    doc=derive(parent,tmp_path/'out',{})
    validate(doc)
    bound.write_bytes(b'changed')
    with pytest.raises(ValueError):validate(doc)


def test_gate_rejects_forged_pass_with_incomplete_trial_coverage(tmp_path):
    from research.explanation_replication.run import validate_gate
    terminal=tmp_path/'result.json'
    original.write_new(terminal,dict(case_id='c',method_id='m',seed=37,status='timeout'))
    doc=dict(manifest_id='id',methods=['m'],cases=[dict(case_id='c',source_family='BATADAL')],
             trial_ids=['c|m|37'],production_output=str(tmp_path))
    gate=dict(passed=True,manifest_id='id',evidence={str(terminal):original.digest(terminal)})
    with pytest.raises(ValueError):validate_gate(doc,gate)


def test_event_provenance_and_native_accounting_rejected_when_changed():
    from research.explanation_replication.run import verify_event_identity
    expected=dict(event_id='mean',sha256='input')
    event=dict(event_id='mean',input_sha256='input',target=3.,native=dict(
        accounting_verified=True,propagation_verified=True,target=3.,components=dict(a=1.,b=2.),
        target_row=250,origin_row=251))
    verify_event_identity(expected,'dynsphere',event)
    for changed in (dict(event,input_sha256='other'),dict(event,event_id='other'),
                    dict(event,native=dict(event['native'],components=dict(a=1.)))):
        with pytest.raises(ValueError):verify_event_identity(expected,'dynsphere',changed)


def test_hard_budget_records_timeout_and_kills_only_own_process_group(tmp_path):
    import subprocess,sys
    code='from pathlib import Path;import time;from research.explanation_replication.run import start_guard;start_guard(Path({!r}),.1);time.sleep(5)'.format(str(tmp_path))
    process=subprocess.run([sys.executable,'-c',code],start_new_session=True,timeout=10)
    assert process.returncode==-9
    assert json.loads((tmp_path/'hard-timeout.json').read_text())['phase']=='hard-process-limit'


def test_repeated_full_scores_required_for_restored_models(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from research.explanation_replication.run import repeated_scores
    np.save(tmp_path/'test.npy',np.ones((300,2)))
    fitted=SimpleNamespace(score=lambda x:np.ones(len(x)))
    assert len(repeated_scores(fitted,tmp_path))==300
    class Changed:
        calls=0
        def score(self,x):
            self.calls+=1
            return np.ones(len(x))*self.calls
    with pytest.raises(AssertionError):repeated_scores(Changed(),tmp_path)
