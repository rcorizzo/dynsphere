import copy
import importlib
import os

import numpy as np
import pytest


def recovery():
    try:
        return importlib.import_module('research.submission_recovery.run')
    except ModuleNotFoundError:
        pytest.fail('The isolated CPU recovery runner is not implemented')


def test_reserved_omni_validation_keeps_exact_fitting_rows():
    module=recovery()
    train=np.zeros((400,2));cal=np.ones((100,2))
    combined=module.reserved_data(train,cal,100)
    np.testing.assert_array_equal(combined[:400],train)
    np.testing.assert_array_equal(combined[400:],cal)
    assert int(.8*len(combined))==400
    with pytest.raises(ValueError,match='validation'):
        module.reserved_data(train,np.ones((80,2)),100)
    with pytest.raises(ValueError,match='boundary'):
        module.reserved_data(train,np.ones((102,2)),100)


def test_private_overlay_only_changes_new_worker_environment(tmp_path):
    module=recovery()
    environment={'PATH':'original-path','PYTHONPATH':'original-python-path'}
    old=environment.copy()
    result=module.worker_environment('dagmm',37,tmp_path,environment)
    assert environment==old
    assert result['PATH']=='original-path'
    assert result['PYTHONPATH'].split(os.pathsep)[0]==str(tmp_path)
    assert result['CUDA_VISIBLE_DEVICES']==''
    assert str(tmp_path) not in module.worker_environment('omni_anomaly_tsb_ad',37,tmp_path,environment)['PYTHONPATH']


def test_recovery_validation_rejects_wrong_reserved_input(tmp_path):
    module=recovery()
    from research.evaluation_protocols.campaign import _identity,_json,digest
    method='omni_anomaly_tsb_ad';seed=37
    entry={'case_id':'fixture','source_family':'MSL'}
    case={'train':np.zeros((400,2)),'calibration':np.ones((100,2)),
          'test':np.zeros((300,2)),'labels':np.r_[np.zeros(150),np.ones(150)]}
    config={'window_size':100,'validation_partition':'reserved-normal-calibration-v1'}
    doc={'manifest_id':'recovery','methods':[method],'seeds':[seed],
         'configurations':{method:config},'upstream_sources':{method:{'revision':'fixed'}},
         'runtime_evidence':{'overlay_sha256':'fixed'}}
    worker=tmp_path/'worker';worker.mkdir()
    np.save(worker/'scores.npy',np.arange(300.))
    inputs=module.input_hashes(case,method)
    record={'status':'complete','manifest_id':'recovery','case_id':'fixture','source_family':'MSL',
            'method_id':method,'seed':seed,'input_sha256':inputs,
            'run_identity':_identity({'manifest':'recovery','case':'fixture','seed':seed,'method':method}),
            'scores_sha256':digest(worker/'scores.npy')}
    metadata={**record,'configuration':config,'upstream_source':{'revision':'fixed'},
              'recovery_manifest_id':'recovery','runtime_evidence':doc['runtime_evidence']}
    _json(worker/'run.json',metadata)
    np.testing.assert_array_equal(module.validate_record(record,tmp_path,doc,entry,case),np.arange(300.))
    changed=copy.deepcopy(case);changed['calibration'][0,0]=7
    with pytest.raises(ValueError,match='input'):
        module.validate_record(record,tmp_path,doc,entry,changed)
    wrong=copy.deepcopy(doc);wrong['configurations'][method]['window_size']=50
    with pytest.raises(ValueError,match='configuration'):
        module.validate_record(record,tmp_path,wrong,entry,case)


def test_runtime_installer_targets_only_private_binary_packages(tmp_path):
    from research.submission_recovery.setup_runtime import install_command
    command=install_command(tmp_path/'uv',tmp_path/'python',tmp_path/'private')
    assert command==[str(tmp_path/'uv'),'pip','install','--python',str(tmp_path/'python'),
                     '--target',str(tmp_path/'private'),'--only-binary',':all:',
                     '--no-deps','--link-mode','copy','pandas==1.5.3','pytz==2024.1']


def test_cross_campaign_selection_is_explicit_and_rejects_duplicates():
    from research.submission_recovery.aggregate import select_records
    original=[{'method_id':'catch','case_id':'a','seed':37},
              {'method_id':'dagmm','case_id':'a','seed':37}]
    repaired=[{'method_id':'dagmm','case_id':'a','seed':37}]
    assert select_records(original,repaired)==[(original[0],'original'),(repaired[0],'recovery')]
    with pytest.raises(ValueError,match='Duplicate'):
        select_records(original,repaired+repaired)
    with pytest.raises(ValueError,match='Unexpected'):
        select_records(original,[{'method_id':'catch','case_id':'a','seed':37}])


def test_cross_campaign_refuses_changed_unapproved_configuration():
    from research.submission_recovery.aggregate import validate_manifests
    module=recovery()
    original={'base_manifest':{'cases':[]},'methods':['catch',*module.METHODS],
              'seeds':[37,38,39],'configurations':{'catch':{},'dagmm':{'epochs':20},
                    'omni_anomaly_tsb_ad':{'window_size':100}},
              'upstream_sources':{'catch':{},'dagmm':{},'omni_anomaly_tsb_ad':{}}}
    original['manifest_id']=module._identity(original)
    repaired=copy.deepcopy(original);repaired.pop('manifest_id')
    repaired['methods']=list(module.METHODS)
    repaired['configurations'].pop('catch');repaired['upstream_sources'].pop('catch')
    repaired['configurations']['omni_anomaly_tsb_ad']['validation_partition']='reserved-normal-calibration-v1'
    repaired['recovered_from_manifest']=original['manifest_id']
    repaired['manifest_id']=module._identity(repaired)
    validate_manifests(original,repaired)
    repaired['configurations']['dagmm']['epochs']=5
    repaired['manifest_id']=module._identity({k:v for k,v in repaired.items() if k!='manifest_id'})
    with pytest.raises(ValueError,match='configuration'):
        validate_manifests(original,repaired)


def test_run_lock_refuses_overlapping_writer_without_changing_record(tmp_path):
    from research.submission_recovery.run import run_lock
    record=tmp_path/'run.json';record.write_text('{"status":"running"}')
    with run_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with run_lock(tmp_path):
                record.write_text('overwritten')
    assert record.read_text()=='{"status":"running"}'
    with run_lock(tmp_path):
        assert record.is_file()


def test_worker_rechecks_its_actual_source_and_manifest(tmp_path,monkeypatch):
    module=recovery()
    source=tmp_path/'runner.py';source.write_text('original')
    doc={'code_hashes':{},'base_manifest':{'implementation_hashes':{}},
         'recovery_source_hashes':{'runner.py':module.digest(source)}}
    doc['manifest_id']=module._identity(doc)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    module.verify_worker_sources(doc,doc['manifest_id'])
    source.write_text('changed after controller freeze')
    with pytest.raises(ValueError,match='source'):
        module.verify_worker_sources(doc,doc['manifest_id'])
    source.write_text('original')
    doc['code_hashes']['extra.py']='new'
    with pytest.raises(ValueError,match='manifest'):
        module.verify_worker_sources(doc,doc['manifest_id'])
