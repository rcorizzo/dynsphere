import importlib

import numpy as np
import pytest


def core():
    try:
        return importlib.import_module('research.explanation_study.core')
    except ModuleNotFoundError:
        pytest.fail('Explanation metric/intervention implementation is missing')


def test_windows_reject_labels_boundaries_and_flat_focus():
    x=np.column_stack([np.arange(40.),np.arange(40.)**2])
    y=np.zeros(40); y[9]=1
    e=np.r_[np.zeros(20),np.ones(20)]
    starts=core().select_windows(x,y,e,np.ones(2),size=8,focus=(2,6))
    assert starts==[0,32]
    x[:]=0
    with pytest.raises(ValueError,match='windows'):
        core().select_windows(x,y,e,np.ones(2),size=8,focus=(2,6))


def test_edits_have_exact_support_and_do_not_mutate_input():
    x=np.column_stack([np.sin(np.arange(512)/7),np.cos(np.arange(512)/5),np.arange(512)/100])
    original=x.copy()
    for kind in ('mean','variance','reversal'):
        edit,truth=core().inject(x,np.ones(3),kind,17)
        np.testing.assert_array_equal(x,original)
        assert truth.shape==x.shape and truth.any()
        assert not truth[:240].any() and not truth[272:].any()
        assert truth.any(axis=0).sum()==2
        np.testing.assert_array_equal(edit[~truth],x[~truth])
        np.testing.assert_array_equal(core().inject(x,np.ones(3),kind,17)[0],edit)


def test_additive_infidelity_uses_independent_joint_queries():
    c=core(); x=np.ones((512,3)); ref=np.zeros_like(x)
    score=lambda a:float(a[128:384].sum())
    groups=c.group_masks(x.shape,'sensor')
    phi=c.attributions(score,x,ref,groups)
    np.testing.assert_array_equal(phi,[256.,256.,256.])
    masks=c.coalitions(3,32,19)
    assert np.all(masks.sum(1)>=2)
    truth=c.coalition_changes(score,x,ref,groups,masks)
    assert c.infidelity(phi,masks,truth)==pytest.approx(0.)
    assert c.infidelity(np.zeros(3),masks,truth)==pytest.approx(1.)
    assert c.infidelity(phi,masks,np.zeros(32)) is None


def test_temporal_groups_cover_analysis_interval_and_ground_truth():
    c=core(); masks=c.group_masks((512,4),'time')
    assert len(masks)==8
    total=np.sum(masks,axis=0)
    assert np.all(total[128:384]==1) and not total[:128].any() and not total[384:].any()
    truth=np.zeros((512,4),bool);truth[240:272,2]=True
    np.testing.assert_array_equal(c.group_truth(truth,masks),[False,False,False,True,True,False,False,False])


def test_fractional_ties_do_not_create_artificial_top_k_winners():
    c=core()
    assert c.precision_at_k(np.zeros(4),np.array([1,0,0,0]),2)==pytest.approx(.25)
    assert c.average_precision(np.zeros(4),np.array([1,0,0,0]))==pytest.approx(.25)
    assert c.average_precision([4,0,0,0],[1,0,0,0])==1.
    assert c.rank_agreement(np.ones(4),np.ones(4)) is None
    assert c.rank_agreement([1,2,3],[3,2,1])==pytest.approx(-1.)


def test_coalition_evaluation_rejects_nonfinite_scores():
    c=core();x=np.ones((512,2));groups=c.group_masks(x.shape,'sensor')
    with pytest.raises(ValueError,match='finite'):
        c.attributions(lambda a:float('nan'),x,np.zeros_like(x),groups)


def test_noise_and_donor_diagnostics_are_reported_separately():
    c=core(); rng=np.random.default_rng(4)
    clean=rng.normal(size=(512,4)); x,truth=c.inject(clean,np.ones(4),'mean',5)
    result=c.explain_event(lambda a:float(np.mean(a[240:272]**2)),
                         x,clean,truth,[np.zeros_like(x),np.ones_like(x)],np.ones(4),7)
    assert set(result['resolutions'])=={'sensor','time'}
    for item in result['resolutions'].values():
        assert 'infidelity' in item and 'second_reference_infidelity' in item
        assert 'random_infidelity' in item and 'random_ap' in item
        assert len(item['heldout_changes'])==32
    assert result['score_calls']>0 and result['explanation_seconds']>=0


def test_symmetric_focus_is_ineligible_for_reversal_without_score_selection():
    c=core();x=np.column_stack([np.arange(40.),np.arange(40.)**2])
    x[2:6]=np.array([[0,0],[1,1],[1,1],[0,0]])
    starts=c.select_windows(x,np.zeros(40),np.zeros(40),np.ones(2),size=8,focus=(2,6))
    assert starts==[1,32]
    z=np.column_stack([np.arange(512.),np.arange(512.),np.arange(512.)**2])
    z[240:272,0]=np.r_[np.arange(16),np.arange(16)[::-1]]
    edited,truth=c.inject(z,np.ones(3),'reversal',17)
    assert not truth[:,0].any() and truth[:,1].any() and truth[:,2].any()


def test_endpoint_only_changes_are_excluded_before_zero_taper_reversal():
    c=core();x=np.column_stack([np.arange(40.),np.arange(40.)**2])
    x[2:6]=np.array([[0,0],[0,0],[0,0],[1,1]])
    assert c.select_windows(x,np.zeros(40),np.zeros(40),np.ones(2),size=8,focus=(2,6))==[1,32]
