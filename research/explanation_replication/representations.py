"""Training-only compactness of prespecified context-level representations."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np


def context_starts(values,labels,entities,limit=128):
    n=len(values)
    if len(labels)!=n or len(entities)!=n:raise ValueError('Misaligned context metadata')
    bad=np.r_[0,np.cumsum(np.asarray(labels)!=0)]
    boundary=np.r_[0,np.cumsum(np.r_[False,np.asarray(entities)[1:]!=np.asarray(entities)[:-1]])]
    eligible=[i for i in range(n-255) if bad[i+256]==bad[i] and boundary[i+256]==boundary[i+1]]
    if not eligible:return []
    indices=np.unique(np.linspace(0,len(eligible)-1,min(limit,len(eligible))).astype(int))
    return [eligible[i] for i in indices]


def _spectrum(train,test,standardize):
    mean=train.mean(0);scale=train.std(0) if standardize else np.ones(train.shape[1])
    scale=np.where(scale>1e-12,scale,1.)
    x=(train-mean)/scale;y=(test-mean)/scale
    _,s,vt=np.linalg.svd(x,full_matrices=False);energy=float(np.sum(s*s))
    if energy<=1e-24:return dict(status='unavailable',reason='zero training covariance',q95=None)
    ev=s*s/energy;cumulative=np.cumsum(ev);q95=min(len(s),int(np.searchsorted(cumulative,.95))+1)
    positive=ev[ev>1e-15];rank=int(np.sum(s>max(x.shape)*np.finfo(float).eps*s[0]))
    effective=float(np.exp(-np.sum(positive*np.log(positive))))
    residual={};denom=float(np.sum(y*y))
    for q in sorted({1,2,5,q95}):
        q=min(q,train.shape[1],len(vt));basis=vt[:q]
        residual[str(q)]=None if denom<=1e-24 else float(1-np.sum((y-y@basis.T@basis)**2)/denom)
    return dict(status='complete',explained_variance_ratio=ev.tolist(),q95=q95,
        q95_fraction=float(q95/train.shape[1]),empirical_rank=rank,effective_rank=effective,
        normalized_effective_rank=effective/train.shape[1],ev1=float(cumulative[0]),
        ev2=float(cumulative[min(1,len(s)-1)]),ev5=float(cumulative[min(4,len(s)-1)]),
        ev5_structurally_full=train.shape[1]<=5,heldout_reconstruction=residual,
        training_center=mean.tolist(),training_scale=scale.tolist())


def compactness(train,test):
    train=np.asarray(train,float);test=np.asarray(test,float)
    if train.ndim!=2 or test.ndim!=2 or train.shape[1]!=test.shape[1] or min(len(train),len(test))<2:
        raise ValueError('Invalid representation matrices')
    if not np.isfinite(train).all() or not np.isfinite(test).all():raise ValueError('Nonfinite embedding')
    return dict(native_dimension=train.shape[1],train_contexts=len(train),test_contexts=len(test),
        maximum_estimable_rank=min(len(train)-1,train.shape[1]),
        covariance=_spectrum(train,test,False),correlation=_spectrum(train,test,True))


def input_probe(train_z,train_x,test_z,test_x):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    zs=StandardScaler().fit(train_z);xs=StandardScaler().fit(train_x)
    y=xs.transform(test_x);denom=float(np.sum((y-y.mean(0))**2))
    if denom<=1e-24:return None
    model=Ridge(alpha=1.).fit(zs.transform(train_z),xs.transform(train_x))
    prediction=np.asarray(model.predict(zs.transform(test_z))).reshape(y.shape)
    return float(1-np.sum((y-prediction)**2)/denom)


def _hook_vector(fitted,values,method):
    import torch
    model=fitted.model
    if method=='mtad_gat':layer=model.gru
    elif method=='omni_anomaly_tsb_ad':layer=model.model.encoder_mlp
    elif method=='anomaly_transformer':layer=model.model.encoder
    elif method=='tranad':layer=model.model.transformer_encoder
    elif method=='catch':layer=model.model.frequency_transformer
    else:raise ValueError('No prespecified representation layer')
    captured=[]
    def hook(module,inputs,output):
        out=output[1] if method=='mtad_gat' else output[0] if isinstance(output,tuple) else output
        if method=='omni_anomaly_tsb_ad':out=out[...,:model.model.n_latent]
        if not isinstance(out,torch.Tensor):raise ValueError('Non-tensor activation')
        captured.append(out.detach().cpu().numpy().copy())
    handle=layer.register_forward_hook(hook)
    try:scores=fitted.score(values)
    finally:handle.remove()
    if method=='tranad':
        if len(captured)%2:raise ValueError('Unexpected TranAD encoder calls')
        captured=captured[1::2]
    if not captured:raise ValueError('Layer did not execute')
    width=captured[0].shape[-1]
    if any(a.shape[-1]!=width for a in captured):raise ValueError('Changing activation width')
    pooled=np.concatenate([a.reshape(-1,width) for a in captured],axis=0).mean(0)
    return pooled,scores


def embedding(fitted,values,method):
    if method.startswith('dynsphere'):
        model=fitted.model.model
        return model._latent_state(model.pca_.transform(model.feature_scaler_.transform(values))).mean(0)
    if method=='pca_reconstruction':
        model=fitted.model
        return ((values-model.model.mean_)@model.components.T).mean(0)
    if method=='dagmm':raise ValueError('Archived DAGMM does not expose a bound compression tensor interface')
    if method in ('kmeans_distance','linear_ridge_ar1'):raise ValueError('No learned bottleneck representation')
    vector,first=_hook_vector(fitted,values,method)
    second=fitted.score(values)
    np.testing.assert_allclose(first,second,rtol=1e-7,atol=1e-9,equal_nan=True)
    return vector


def study(fitted,method,case,output):
    from research.explanation_study import run as original
    from research.explanation_study.models import get_rng,set_rng
    if method in ('kmeans_distance','linear_ridge_ar1','dagmm'):
        return dict(status='not_applicable' if method!='dagmm' else 'unavailable',
            reason='No learned bottleneck' if method!='dagmm' else 'No bound archived compression tensor interface')
    start=time.perf_counter();inputs=Path(case['inputs']);matrices={};raw={};starts={};previous=get_rng()
    try:
        for split in ('train','test'):
            values=np.load(inputs/(split+'.npy'),allow_pickle=False)
            entities=np.load(inputs/(split+'_entities.npy'),allow_pickle=False)
            labels=np.zeros(len(values),int) if split=='train' else np.load(inputs/'labels.npy',allow_pickle=False)
            selected=context_starts(values,labels,entities)
            if len(selected)<32:return dict(status='unavailable',reason='Fewer than32 eligible contexts',split=split)
            starts[split]=selected;contexts=[values[s:s+256] for s in selected]
            matrices[split]=np.asarray([embedding(fitted,x,method) for x in contexts])
            raw[split]=np.asarray([x.mean(0) for x in contexts])
            if split=='train':
                repeated=embedding(fitted,contexts[0],method)
                np.testing.assert_allclose(repeated,matrices[split][0],rtol=1e-7,atol=1e-9)
        path=Path(output)/'representations.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,**matrices,train_raw_mean=raw['train'],test_raw_mean=raw['test'])
        return dict(status='complete',metrics=compactness(matrices['train'],matrices['test']),
            input_mean_probe_r2=input_probe(matrices['train'],raw['train'],matrices['test'],raw['test']),
            starts=starts,context_rows=256,pooling='mean over all non-feature activation axes',
            representation_path=str(path),representation_sha256=original.digest(path),
            seconds=time.perf_counter()-start,claim='compactness/input retention; not interpretability or anomaly relevance')
    finally:set_rng(previous)
