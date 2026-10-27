"""Capture unchanged timing estimators and replay their original inference."""
from __future__ import annotations

from contextlib import contextmanager
import pickle
import random
import sys
import types
import numpy as np


def get_rng():
    state={'numpy':np.random.get_state(),'python':random.getstate()}
    if 'torch' in sys.modules:
        state['torch']=sys.modules['torch'].get_rng_state()
    return state


def set_rng(state):
    np.random.set_state(state['numpy']);random.setstate(state['python'])
    if 'torch' in state:
        sys.modules['torch'].set_rng_state(state['torch'])


def rebind_test(function, values):
    names=function.__code__.co_freevars
    if 'test' not in names:
        raise ValueError('Original scorer must close over test')
    def cell(value):
        return (lambda:value).__closure__[0]
    cells=tuple(cell(values) if name=='test' else original
                for name,original in zip(names,function.__closure__))
    return types.FunctionType(function.__code__,function.__globals__,
                              function.__name__,function.__defaults__,cells)


class ReplayedScorer:
    def __init__(self,function,state):
        self.function,self.state,self.calls=function,state,0

    def __call__(self,values):
        self.calls+=1; previous=get_rng()
        try:
            set_rng(self.state)
            return np.asarray(self.function(values),dtype=float).reshape(-1)
        finally:
            set_rng(previous)


class Fitted:
    def __init__(self,model,function,state):
        self.model=model;self.score=ReplayedScorer(function,state)

    def target(self,values):
        scores=self.score(values)
        if scores.shape!=(len(values),) or not np.isfinite(scores[240:272]).all():
            raise ValueError('Nonfinite or missing target scores')
        return float(np.max(scores[240:272]))


@contextmanager
def replace_attribute(obj,name,value):
    original=getattr(obj,name);setattr(obj,name,value)
    try:
        yield original
    finally:
        setattr(obj,name,original)


def fit_original(method,seed,inputs,configuration,output):
    """Original fitting/roundtrip code executes intact; capture its verified state."""
    from pathlib import Path
    from research.quality_cost_extension import trials
    local=method.startswith('dynsphere') or method in ('pca_reconstruction','kmeans_distance','linear_ridge_ar1')
    inputs=Path(inputs); captured={}
    if local:
        from research.evaluation_protocols.campaign import FittedControl
        from research.quality_cost import run
        original=pickle.dumps
        def watch(value,*args,**kwargs):
            if isinstance(value,FittedControl):captured['model']=value
            return original(value,*args,**kwargs)
        with replace_attribute(pickle,'dumps',watch):
            evidence,scores=run.local_trial(method,seed,inputs)
        model=captured['model']
        function=lambda values:model.score(values,np.zeros(len(values),dtype=int))
        state=get_rng()
    elif method in ('catch','mtad_gat'):
        from research.quality_cost import run
        original=run.check_serialization
        def watch(model,score,reference,state):
            captured.update(model=model,score=score,state=state)
            return original(model,score,reference,state)
        with replace_attribute(run,'check_serialization',watch):
            evidence,scores=run.deep_trial(method,seed,inputs,configuration)
        model=captured['model'];state=captured['state']
        function=lambda values:rebind_test(captured['score'],values)(model)
    elif method in ('tranad','anomaly_transformer','omni_anomaly_tsb_ad'):
        arrays=[np.load(inputs/(n+'.npy'),allow_pickle=False) for n in ('train','calibration','test')]
        original=trials.pickle_roundtrip
        def watch(model,score,reference,state):
            captured.update(model=model,score=score,state=state)
            return original(model,score,reference,state)
        trial=trials.omni_trial if method=='omni_anomaly_tsb_ad' else trials.transformer_trial
        with replace_attribute(trials,'pickle_roundtrip',watch):
            evidence,scores=trial(method,*arrays,configuration,seed,output)
        model=captured['model'];state=captured['state']
        function=lambda values:rebind_test(captured['score'],values)(model)
    else:
        raise ValueError('DAGMM requires validated archived checkpoint restoration')
    return Fitted(model,function,state),np.asarray(scores),evidence


def restore_dagmm(checkpoint, seed):
    from pathlib import Path
    import tensorflow.compat.v1 as tf
    from benchmarks.official_baselines.adapters.tab_dagmm import _load_dagmm
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    root=Path(__file__).resolve().parents[2]
    cls=_load_dagmm(root/'benchmarks/official_baselines/.external/dagmm/ts_benchmark/baselines/tods/third_party')
    np.random.seed(seed);random.seed(seed)
    model=cls(normalize=False);model.restore(str(checkpoint))
    return Fitted(model,lambda values:model.decision_function(values),get_rng())


def native_diagnostics(model, values):
    from pendulum_detector import _rolling_entity_max
    entities=np.zeros(len(values),dtype=int)
    parts=model.decision_components(values,entities)
    keys=[model.position_component_name,model.latent_dynamics_component_name,
          'raw_dynamics','state_speed','velocity','regime_distance','transition']
    reconstructed=sum(parts[k] for k in keys)
    np.testing.assert_allclose(reconstructed,parts['pre_propagation_score'],rtol=1e-10,atol=1e-10)
    propagated=_rolling_entity_max(reconstructed,entities,model.propagation_radius)
    np.testing.assert_allclose(propagated,parts['score'],rtol=1e-10,atol=1e-10)
    target=240+int(np.argmax(parts['score'][240:272]));radius=model.propagation_radius
    lower=max(0,target-radius);upper=min(len(values),target+radius+1)
    origin=lower+int(np.argmax(reconstructed[lower:upper]))
    return dict(accounting_verified=True,propagation_verified=True,target=float(parts['score'][target]),
                target_row=target,origin_row=origin,
                components={k:float(parts[k][origin]) for k in keys})
