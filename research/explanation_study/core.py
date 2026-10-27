"""Common sensor/time interventions, with explicit undefined metric handling."""
from __future__ import annotations

import time
import numpy as np


def select_windows(values, labels, entities, training_sd, size=512, focus=(240,272)):
    values,labels,entities=map(np.asarray,(values,labels,entities))
    if values.ndim!=2 or len(values)!=len(labels) or len(labels)!=len(entities):
        raise ValueError('Invalid evaluation arrays')
    if len(values)<2*size:
        raise ValueError('Insufficient nonoverlapping windows')
    bad=np.convolve((labels!=0).astype(int),np.ones(size,dtype=int),'valid')
    starts=np.flatnonzero(bad==0)
    def eligible(s):
        block=values[s+focus[0]:s+focus[1]]
        return (np.all(entities[s:s+size]==entities[s]) and
                np.count_nonzero((np.max(np.abs(block[1:-1]-block[::-1][1:-1]),axis=0)>1e-10)
                                 &(np.asarray(training_sd)>1e-12))>=2)
    first=next((int(s) for s in starts if eligible(s)),None)
    last=next((int(s) for s in starts[::-1] if first is not None and s>=first+size and eligible(s)),None)
    if first is None or last is None:
        raise ValueError('Insufficient eligible nonoverlapping windows')
    return [first,last]


def inject(clean, training_sd, kind, seed):
    clean=np.asarray(clean,dtype=float)
    sd=np.asarray(training_sd,dtype=float)
    if clean.shape[0]!=512 or not np.isfinite(clean).all():
        raise ValueError('Expected finite length512 context')
    block=clean[240:272]
    candidates=np.flatnonzero((sd>1e-12)&(np.max(np.abs(block[1:-1]-block[::-1][1:-1]),axis=0)>1e-10))
    if len(candidates)<2:
        raise ValueError('Two varying features are required')
    chosen=np.random.default_rng(seed).choice(candidates,2,replace=False)
    edited=clean.copy()
    taper=np.sin(np.linspace(0,np.pi,32))**2
    for j in chosen:
        current=clean[240:272,j]
        if kind=='mean':
            delta=2*sd[j]*taper
        elif kind=='variance':
            delta=2*(current-current.mean())*taper
        elif kind=='reversal':
            delta=(current[::-1]-current)*taper
        else:
            raise ValueError('Unknown edit')
        edited[240:272,j]+=delta
    truth=edited!=clean
    if not truth.any() or np.count_nonzero(truth.any(0))!=2:
        raise ValueError('Degenerate controlled edit')
    return edited,truth


def group_masks(shape, axis):
    if len(shape)!=2 or shape[0]!=512 or shape[1]<2:
        raise ValueError('Expected length512 with at least two features')
    result=[]
    if axis=='sensor':
        for j in range(shape[1]):
            mask=np.zeros(shape,bool); mask[128:384,j]=True; result.append(mask)
    elif axis=='time':
        for i in range(8):
            mask=np.zeros(shape,bool); mask[128+32*i:160+32*i]=True; result.append(mask)
    else:
        raise ValueError('Unknown explanation resolution')
    return np.asarray(result)


def group_truth(truth, masks):
    return np.asarray([bool(np.any(truth&mask)) for mask in masks])


def finite_score(score, values):
    value=float(score(values))
    if not np.isfinite(value):
        raise ValueError('Target score must be finite')
    return value


def replaced(x, reference, mask):
    return np.where(mask,reference,x)


def attributions(score, values, reference, masks):
    base=finite_score(score,values)
    return np.asarray([base-finite_score(score,replaced(values,reference,g)) for g in masks])


def coalitions(groups, count, seed):
    if groups<2:
        raise ValueError('At least two groups are required')
    rng=np.random.default_rng(seed); masks=np.zeros((count,groups),bool)
    for mask in masks:
        k=int(rng.integers(2,min(4,groups)+1)); mask[rng.choice(groups,k,replace=False)]=True
    return masks


def coalition_changes(score, values, reference, groups, masks):
    base=finite_score(score,values)
    return np.asarray([base-finite_score(score,replaced(values,reference,np.any(groups[m],axis=0)))
                       for m in masks])


def infidelity(phi, masks, actual):
    phi,masks,actual=map(np.asarray,(phi,masks,actual))
    if not np.isfinite(phi).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite explanation evaluation')
    denominator=float(np.mean(actual**2))
    return None if denominator<=1e-24 else float(np.mean((masks@phi-actual)**2)/denominator)


def precision_at_k(importance, truth, k=2):
    importance=np.asarray(importance);truth=np.asarray(truth,dtype=bool)
    k=min(k,len(importance)); boundary=np.sort(importance)[-k]
    greater=importance>boundary; ties=importance==boundary
    return float((truth[greater].sum()+(k-greater.sum())*truth[ties].mean())/k)


def average_precision(importance, truth):
    importance=np.asarray(importance); truth=np.asarray(truth,dtype=bool)
    if not truth.any():
        return None
    result=0.; previous=0.
    for threshold in np.unique(importance)[::-1]:
        chosen=importance>=threshold; hit=int(truth[chosen].sum())
        recall=hit/truth.sum(); result+=(recall-previous)*hit/chosen.sum();previous=recall
    return float(result)


def ranks(values):
    values=np.asarray(values); out=np.zeros(len(values))
    for value in np.unique(values):
        out[values==value]=np.count_nonzero(values<value)+(np.count_nonzero(values==value)-1)/2
    return out


def rank_agreement(a,b):
    a,b=ranks(a),ranks(b)
    return None if min(np.ptp(a),np.ptp(b))==0 else float(np.corrcoef(a,b)[0,1])


def explain_event(score, x, clean, truth, references, training_sd, seed):
    started=time.perf_counter(); calls=0
    def counted(values):
        nonlocal calls
        calls+=1
        return finite_score(score,values)
    base=counted(x); repeat=counted(x)
    if not np.isclose(base,repeat,rtol=1e-10,atol=1e-10):
        raise ValueError('Repeated target scoring is not deterministic under RNG replay')
    clean_score=counted(clean)
    noise=np.random.default_rng(seed).normal(size=x.shape)*np.asarray(training_sd)*.01
    perturbed=x+noise; perturbed_score=counted(perturbed)
    result=dict(target=base,clean_target=clean_score,edit_score_increase=base>clean_score,
                noise_score_change=perturbed_score-base,
                score_stable=bool(abs(perturbed_score-base)<=.05*abs(base)+1e-8),resolutions={})
    for i,axis in enumerate(('sensor','time')):
        groups=group_masks(x.shape,axis); annotated=group_truth(truth,groups)
        phi=attributions(counted,x,references[0],groups)
        noisy=attributions(counted,perturbed,references[0],groups)
        heldout=coalitions(len(groups),32,seed+100+i)
        actual=coalition_changes(counted,x,references[0],groups,heldout)
        other=coalition_changes(counted,x,references[1],groups,heldout)
        random_phi=np.random.default_rng(seed+200+i).permutation(phi)
        positive=np.maximum(phi,0); random_positive=np.maximum(random_phi,0)
        interventions=np.where(groups,references[0]-x,0.)
        scale=np.maximum(np.asarray(training_sd),1e-12)
        result['resolutions'][axis]=dict(attributions=phi.tolist(),noisy_attributions=noisy.tolist(),
            random_attributions=random_phi.tolist(),
            random_noisy_attributions=np.random.default_rng(seed+200+i).permutation(noisy).tolist(),
            truth=annotated.tolist(),
            infidelity=infidelity(phi,heldout,actual),second_reference_infidelity=infidelity(phi,heldout,other),
            random_infidelity=infidelity(random_phi,heldout,actual),
            ap=average_precision(positive,annotated),precision_at_2=precision_at_k(positive,annotated),
            random_ap=average_precision(random_positive,annotated),
            random_precision_at_2=precision_at_k(random_positive,annotated),
            stability=rank_agreement(phi,noisy),
            random_stability=rank_agreement(random_phi,np.random.default_rng(seed+200+i).permutation(noisy)),
            constant_attribution=bool(np.ptp(phi)==0),
            heldout_masks=heldout.tolist(),heldout_changes=actual.tolist(),second_reference_changes=other.tolist(),
            intervention_rms_standardized=float(np.sqrt(np.mean((interventions/scale)**2))))
    result.update(score_calls=calls,explanation_seconds=time.perf_counter()-started)
    return result
