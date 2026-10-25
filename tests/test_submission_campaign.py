import numpy as np
import pytest


def test_state_motion_groups_reconstruct_unpropagated_e1_score():
    from research.submission_campaign.studies import grouped_scores
    terms = {k: np.arange(30, dtype=float) * (i+1) for i,k in enumerate(
        ['position', 'regime_distance', 'euclidean_dynamics', 'raw_dynamics', 'state_speed', 'velocity'])}
    terms['transition'] = np.zeros(30)
    terms['pre_propagation_score'] = sum(terms.values())
    terms['score'] = terms['pre_propagation_score'].copy()
    result = grouped_scores(terms, np.zeros(30, dtype=int), n_concepts=1)
    np.testing.assert_allclose(result['state_r0'], np.arange(30)*3)
    np.testing.assert_allclose(result['motion_r0'], np.arange(30)*18)
    np.testing.assert_allclose(result['combined_r0'], terms['score'])
    assert result['state_r8'][10] == 54
    assert result['motion_r8'][10] == 324


def test_multiregime_component_reuse_is_rejected():
    from research.submission_campaign.studies import grouped_scores
    with pytest.raises(ValueError, match='one regime'):
        grouped_scores({}, np.zeros(3), n_concepts=4)


def test_frozen_combined_score_disagreement_blocks_study():
    from research.submission_campaign.studies import verify_reference
    with pytest.raises(ValueError, match='frozen'):
        verify_reference(np.arange(4.), np.arange(4.)+1)


def test_deep_comparison_excludes_padded_prefix_and_tail():
    from research.submission_campaign.deep import common_mask
    mask = common_mask(450)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(100,350))
    with pytest.raises(ValueError, match='support'):
        common_mask(200)


def test_worker_validation_rejects_wrong_seed_or_changed_scores(tmp_path):
    from research.submission_campaign.deep import validate_worker
    from research.evaluation_protocols.campaign import _json,digest
    np.save(tmp_path/'scores.npy',np.arange(250.))
    _json(tmp_path/'run.json',{'status':'complete','method_id':'mtad_gat','seed':38,
          'scores_sha256':digest(tmp_path/'scores.npy'),'input_sha256':{'train.npy':'a','test.npy':'b'}})
    with pytest.raises(ValueError,match='identity'):
        validate_worker(tmp_path,'mtad_gat',37,250,{'train.npy':'a','test.npy':'b'})


def test_tranad_import_is_primed_before_requested_seed():
    from research.submission_campaign.deep import prime_and_seed
    torch=pytest.importorskip('torch')
    pytest.importorskip('dgl',reason='TranAD replay is also tested in its dedicated environment')
    prime_and_seed('tranad',37)
    first=torch.rand(4)
    prime_and_seed('tranad',37)
    np.testing.assert_array_equal(first.numpy(),torch.rand(4).numpy())
    prime_and_seed('tranad',38)
    assert not np.array_equal(first.numpy(),torch.rand(4).numpy())


def test_catch_reserved_validation_preserves_exact_fitting_boundary():
    from research.submission_campaign.deep import catch_fit_frame
    train=np.zeros((400,3));cal=np.ones((100,3))
    combined=catch_fit_frame(train,cal)
    np.testing.assert_array_equal(combined.iloc[:400].to_numpy(),train)
    np.testing.assert_array_equal(combined.iloc[400:].to_numpy(),cal)
    with pytest.raises(ValueError,match='boundary'):
        catch_fit_frame(train,np.ones((80,3)))


def test_upstream_snapshot_detects_edited_and_untracked_source(tmp_path):
    import subprocess
    from research.submission_campaign import deep
    subprocess.run(['git','init',str(tmp_path)],check=True,capture_output=True)
    (tmp_path/'model.py').write_text('weight = 1\n')
    subprocess.run(['git','-C',str(tmp_path),'add','model.py'],check=True)
    subprocess.run(['git','-C',str(tmp_path),'-c','user.name=Test',
                    '-c','user.email=test@localhost','commit','-m','fixture'],check=True,capture_output=True)
    revision=subprocess.check_output(['git','-C',str(tmp_path),'rev-parse','HEAD'],text=True).strip()
    snapshot=getattr(deep,'upstream_snapshot',None)
    assert callable(snapshot), 'Actual upstream source must be fingerprinted'
    original=snapshot(tmp_path,revision)
    (tmp_path/'model.py').write_text('weight = 2\n')
    assert snapshot(tmp_path,revision)!=original
    (tmp_path/'model.py').write_text('weight = 1\n')
    (tmp_path/'extra.py').write_text('weight = 3\n')
    assert snapshot(tmp_path,revision)!=original
    with pytest.raises(ValueError,match='revision'):
        snapshot(tmp_path,'0'*40)


def test_aggregation_rejects_self_consistent_inputs_from_another_preparation(tmp_path):
    from research.submission_campaign import aggregate
    from research.evaluation_protocols.campaign import _json,_identity,digest
    folder=tmp_path/'worker';folder.mkdir()
    np.save(folder/'scores.npy',np.arange(250.))
    config={'window_size':100}
    wrong={'train.npy':'different-training-array','test.npy':'different-test-array'}
    doc={'manifest_id':'frozen','methods':['mtad_gat'],'seeds':[37],
         'configurations':{'mtad_gat':config},'upstream_sources':{'mtad_gat':{'revision':'a'}}}
    entry={'case_id':'case','source_family':'MSL'}
    record={'status':'complete','method_id':'mtad_gat','seed':37,'manifest_id':'frozen',
            'case_id':'case','source_family':'MSL','input_sha256':wrong,
            'run_identity':_identity({'manifest':'frozen','case':'case','seed':37,'method':'mtad_gat'}),
            'scores_sha256':digest(folder/'scores.npy')}
    _json(folder/'run.json',{**record,'configuration':config,'upstream_source':{'revision':'a'}})
    case={'train':np.zeros((400,2)),'test':np.ones((250,2)),'calibration':np.zeros((100,2)),
          'labels':np.r_[np.zeros(125),np.ones(125)]}
    validator=getattr(aggregate,'validate_deep_record',None)
    assert callable(validator), 'Aggregation must bind inputs to the frozen case'
    with pytest.raises(ValueError,match='input'):
        validator(record,folder.parent,doc,entry,case)
