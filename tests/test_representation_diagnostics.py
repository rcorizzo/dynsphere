import numpy as np
import pytest


def test_spectrum_reports_rank_q95_and_heldout_without_test_fitting():
    from research.explanation_replication.representations import compactness
    train=np.array([[1,0],[-1,0],[0,1],[0,-1]],float)
    test=np.array([[0,10],[0,-10]],float)
    result=compactness(train,test)
    assert result['native_dimension']==2
    assert result['covariance']['q95']==2
    assert result['covariance']['ev1']==pytest.approx(.5)
    assert result['covariance']['effective_rank']==pytest.approx(2)
    assert result['covariance']['heldout_reconstruction']['2']==pytest.approx(1)
    assert result['covariance']['ev5_structurally_full'] is True


def test_constant_embeddings_unavailable_not_perfect():
    from research.explanation_replication.representations import compactness
    result=compactness(np.ones((40,5)),np.ones((40,5)))
    assert result['covariance']['status']=='unavailable'
    assert result['covariance']['q95'] is None


def test_context_selection_respects_labels_entities_and_stable_spacing():
    from research.explanation_replication.representations import context_starts
    x=np.arange(600)[:,None].astype(float)
    labels=np.zeros(600,int);labels[:10]=1
    entities=np.r_[np.zeros(300),np.ones(300)]
    starts=context_starts(x,labels,entities,limit=5)
    assert len(starts)==5
    assert starts[0]==10 and starts[-1]==344
    assert all((s+256<=300) or s>=300 for s in starts)


def test_linear_input_probe_is_trained_without_test_targets():
    from research.explanation_replication.representations import input_probe
    x=np.arange(40,dtype=float)[:,None]
    score=input_probe(x,x*2,x,x*2)
    assert score>.99
    assert input_probe(x,x*0,x,x*0) is None


def test_saved_budget_diagnostic_retains_signed_full_explanation():
    from research.explanation_replication.diagnostics import budget_curve
    part=dict(attributions=[3.,2.,-1.],heldout_masks=[[1,1,0],[0,1,1]],heldout_changes=[5.,1.])
    result=budget_curve(part)
    assert result['all']==pytest.approx(1/26)
    assert result['signed_full']==pytest.approx(0)
    assert result['1']>result['2']


def test_cross_seed_agreement_requires_same_events_and_all_seeds():
    from research.explanation_replication.diagnostics import cross_seed
    values={37:{'a':[3,2,1]},38:{'a':[3,2,1]},39:{'a':[3,2,1]}}
    assert cross_seed(values)==pytest.approx(1)
    assert cross_seed({37:values[37],38:values[38]}) is None
    assert cross_seed(dict(values,**{}))==pytest.approx(1)
    values[39]={'b':[3,2,1]}
    assert cross_seed(values) is None


def test_hook_cleanup_on_failure_and_success():
    from research.explanation_replication.representations import _hook_vector
    import torch
    from types import SimpleNamespace
    layer=torch.nn.Linear(2,2,bias=False)
    class Scorer:
        model=SimpleNamespace(model=SimpleNamespace(encoder=layer))
        def score(self,x):
            layer(torch.tensor(x,dtype=torch.float32))
            if len(x)==1:raise ValueError('score failed')
            return np.ones(len(x))
    fitted=Scorer()
    with pytest.raises(ValueError):_hook_vector(fitted,np.ones((1,2)),'anomaly_transformer')
    assert len(layer._forward_hooks)==0
    vector,_=_hook_vector(fitted,np.ones((3,2)),'anomaly_transformer')
    assert vector.shape==(2,)
    assert len(layer._forward_hooks)==0


def test_no_change_in_rng_after_representation_failure(tmp_path):
    from research.explanation_replication.representations import study
    from research.explanation_study.models import get_rng
    from types import SimpleNamespace
    arrays={'train':np.ones((300,2)),'train_entities':np.zeros(300),
            'test':np.ones((300,2)),'test_entities':np.zeros(300),'labels':np.zeros(300)}
    for name,value in arrays.items():np.save(tmp_path/(name+'.npy'),value)
    class BadModel:
        def _latent_state(self,x):
            np.random.rand(10)
            raise ValueError('extract failed')
        pca_=SimpleNamespace(transform=lambda x:x)
        feature_scaler_=SimpleNamespace(transform=lambda x:x)
    fitted=SimpleNamespace(model=SimpleNamespace(model=BadModel()))
    before=get_rng()['numpy']
    with pytest.raises(ValueError):study(fitted,'dynsphere',dict(inputs=str(tmp_path)),tmp_path)
    after=get_rng()['numpy']
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
