"""Minimal-runtime workers around unchanged pinned model implementations."""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import resource
import sys
import textwrap
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOCK_ROOT = ROOT/'results/quality_cost_20261026'
PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'pandas', 'torch', 'tensorflow', 'pyod', 'threadpoolctl')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    result = {'python': sys.version, 'executable': sys.executable}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    result['installed_distributions'] = {d.metadata['Name'].lower().replace('_','-'):d.version
                                        for d in importlib.metadata.distributions() if d.metadata['Name']}
    return result


@contextmanager
def hardware_lock(kind):
    if kind not in ('controller','worker'):
        raise ValueError('Unknown hardware lock')
    LOCK_ROOT.mkdir(parents=True,exist_ok=True)
    with (LOCK_ROOT/('.extension-'+kind+'.lock')).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield handle


def restoration_metadata(method,configuration):
    return dict(cpu_compatibility=True,epochs=int(configuration['epochs']),
                revision='audited-v2' if method=='tranad' else 'released-budget-v2',
                audited=True,history=[],training_loss=[])


def random_state():
    result = {'numpy': np.random.get_state(), 'python': random.getstate()}
    if 'torch' in sys.modules:
        result['torch'] = sys.modules['torch'].get_rng_state()
    return result


def restore_random(state):
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if 'torch' in state:
        sys.modules['torch'].set_rng_state(state['torch'])


def pickle_roundtrip(model, score, reference, state):
    payload = pickle.dumps(model, protocol=4)
    restored = pickle.loads(payload)
    restore_random(state)
    actual = np.asarray(score(restored), dtype=float)
    np.testing.assert_allclose(actual[100:-100], reference[100:-100], rtol=1e-7, atol=1e-9)
    return len(payload)


def verify_reference(scores, reference, seed):
    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in sys.modules:
        sys.modules['torch'].manual_seed(seed)
    actual=np.asarray(reference(),dtype=float)
    np.testing.assert_allclose(actual[100:-100],scores[100:-100],rtol=1e-7,atol=1e-9)


def numerical_threads():
    # Minimal Anomaly Transformer runtime has no sklearn/threadpoolctl dependency.
    # Its fresh worker receives all BLAS environment limits before NumPy imports.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return nullcontext()
    return threadpool_limits(limits=1)


def split_backend(function, first_assignment):
    """Compile original adapter statements, without rewriting their expressions."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = tree.body[0].body
    starts = [i for i, node in enumerate(body) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == first_assignment for t in node.targets)]
    stops = [i for i, node in enumerate(body) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
             and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'model'
             and node.value.func.attr == 'eval']
    if len(starts) != 1 or len(stops) != 1 or starts[0] >= stops[0]:
        raise ValueError('Adapter no longer has the declared fit/scoring boundary')
    if not isinstance(body[-1], ast.Return):
        raise ValueError('Adapter must return exactly one terminal output')
    final = ast.Assign(targets=[ast.Name(id='_output', ctx=ast.Store())], value=body[-1].value)
    pieces = [body[:starts[0]], body[starts[0]:stops[0]+1], body[stops[0]+1:-1]+[final]]
    return tuple(compile(ast.fix_missing_locations(ast.Module(
        body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + part,
        type_ignores=[])), inspect.getfile(function), 'exec') for part in pieces)


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def seed_runtime(method, seed):
    random.seed(seed)
    np.random.seed(seed)
    if method != 'dagmm':
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(seed)
        if torch.get_num_threads() != 1:
            raise ValueError('CPU thread isolation failed')


def transformer_program(method):
    module = importlib.import_module('benchmarks.official_baselines.adapters.' + method)
    # Finish framework/upstream imports before timers, preserving the adapter body.
    source = module.source_directory(method)
    sys.path.insert(0, str(source))
    try:
        if method == 'tranad':
            with module._upstream_import_arguments():
                importlib.import_module('src.models')
        else:
            importlib.import_module('model.AnomalyTransformer')
    finally:
        sys.path.pop(0)
    return module, split_backend(module.official_backend, 'window' if method == 'anomaly_transformer' else 'revision')


class TransformerState:
    """Fitted network and required inference metadata; excludes training optimizer."""
    def __init__(self, method, model, configuration, train):
        self.method, self.model, self.configuration = method, model, dict(configuration)
        self.minimum = np.min(train, axis=0) if method == 'tranad' else None
        self.denominator = np.max(train, axis=0)-self.minimum+0.0001 if method == 'tranad' else None

    def score(self, test):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        module, programs = transformer_program(self.method)
        window = int(self.configuration.get('window_size', 100)) if self.method == 'anomaly_transformer' else self.model.n_window
        if self.method == 'tranad':
            test = (test-self.minimum)/self.denominator
            windows = module._target_windows(torch.as_tensor(test, dtype=torch.double), window)
        else:
            windows = module._windows(torch.as_tensor(test, dtype=torch.float32), window)
        loader = DataLoader(TensorDataset(windows), batch_size=int(self.configuration.get('batch_size',128)), shuffle=False)
        scope = dict(module.__dict__, torch=torch, nn=nn, model=self.model, test_loader=loader,
                     device=torch.device('cpu'), window=window, configuration=self.configuration,
                     **restoration_metadata(self.method,self.configuration))
        exec(programs[2], scope)
        result = scope['_output']
        from benchmarks.official_baselines.adapters.common import align_scores
        return align_scores(result, len(test))


def transformer_trial(method, train, calibration, test, config, seed, output):
    del calibration, output
    import torch
    from benchmarks.official_baselines.adapters.common import align_scores
    module, programs = transformer_program(method)
    seed_runtime(method, seed)
    scope = dict(module.__dict__, train=train, test=test, configuration=config, seed=seed)
    exec(programs[0], scope)
    before = rss()
    with numerical_threads():
        started = time.perf_counter()
        exec(programs[1], scope)
        fit_seconds = time.perf_counter()-started
        state = random_state()
        started = time.perf_counter()
        exec(programs[2], scope)
        scores = align_scores(scope['_output'], len(test))
        score_seconds = time.perf_counter()-started
        peak = rss()
        if str(scope['device']) != 'cpu':
            raise ValueError('Transformer used a non-CPU device')
        model = TransformerState(method, scope['model'], config, train)
        size = pickle_roundtrip(model, lambda x: x.score(test), scores, state)
    return dict(fit_seconds=fit_seconds, score_seconds=score_seconds, process_peak_rss_bytes=peak,
                prefit_peak_rss_bytes=before, serialized_bytes=size, serialization_verified=True,
                serialization='pickle4-network-and-inference-metadata',
                timing_preparation='adapter-order; test-window construction occurs before scoring',
                torch_version=torch.__version__), scores


def omni_trial(method, train, calibration, test, config, seed, output):
    del output
    import torch
    source = ROOT/'benchmarks/official_baselines/.external/omni_anomaly_tsb_ad'
    sys.path.insert(0, str(source))
    try:
        from TSB_AD.models.OmniAnomaly import OmniAnomaly
    finally:
        sys.path.pop(0)
    seed_runtime(method, seed)
    if int(.8*(len(train)+len(calibration))) != len(train) or min(len(train),len(calibration))<int(config['window_size']):
        raise ValueError('Reserved calibration boundary is incompatible')
    combined = np.vstack([train,calibration])
    before = rss()
    with numerical_threads():
        started = time.perf_counter()
        model = OmniAnomaly(win_size=int(config['window_size']), feats=train.shape[1],
                            batch_size=int(config.get('batch_size',128)), epochs=int(config['epochs']),
                            lr=float(config.get('lr',.002)))
        if str(model.device) != 'cpu':
            raise ValueError('OmniAnomaly used a non-CPU device')
        model.fit(combined)
        fit_seconds = time.perf_counter()-started
        state = random_state()
        started = time.perf_counter()
        scores = np.asarray(model.decision_function(test),dtype=float).reshape(-1)
        score_seconds = time.perf_counter()-started
        peak = rss()
        snapshot = copy.copy(model)
        for key in ('optimizer','scheduler','early_stopping'):
            delattr(snapshot,key)
        size = pickle_roundtrip(snapshot,lambda x:x.decision_function(test),scores,state)
    return dict(fit_seconds=fit_seconds,score_seconds=score_seconds,process_peak_rss_bytes=peak,
                prefit_peak_rss_bytes=before,serialized_bytes=size,serialization_verified=True,
                serialization='pickle4-fitted-estimator-without-training-optimizer-scheduler',
                torch_version=torch.__version__, validation_partition='reserved-normal-calibration-v1'),scores


def dagmm_trial(method, train, calibration, test, config, seed, output):
    del calibration
    import tensorflow.compat.v1 as tf
    from benchmarks.official_baselines.adapters.tab_dagmm import _load_dagmm
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    cls = _load_dagmm(ROOT/'benchmarks/official_baselines/.external/dagmm/ts_benchmark/baselines/tods/third_party')
    seed_runtime(method,seed)
    before = rss()
    with numerical_threads():
        started = time.perf_counter()
        model = cls(epoch_size=int(config['epochs']),minibatch_size=int(config.get('batch_size',128)),random_seed=seed,normalize=False)
        model.fit(train)
        fit_seconds = time.perf_counter()-started
        started = time.perf_counter()
        scores = np.asarray(model.decision_function(test),dtype=float).reshape(-1)
        score_seconds = time.perf_counter()-started
        peak = rss()
        checkpoint = Path(output)/'fitted-checkpoint'
        model.save(str(checkpoint))
        restored = cls(normalize=False)
        restored.restore(str(checkpoint))
        actual = np.asarray(restored.decision_function(test),dtype=float).reshape(-1)
        np.testing.assert_allclose(actual[100:-100],scores[100:-100],rtol=1e-7,atol=1e-9)
        size = sum(p.stat().st_size for p in checkpoint.iterdir() if p.is_file())
        restored.sess.close()
        model.sess.close()
    return dict(fit_seconds=fit_seconds,score_seconds=score_seconds,process_peak_rss_bytes=peak,
                prefit_peak_rss_bytes=before,serialized_bytes=size,serialization_verified=True,
                serialization='tensorflow-native-checkpoint-including-graph-and-optimizer-variables',
                tensorflow_version=tf.__version__),scores


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--case-id',required=True)
    parser.add_argument('--method',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    doc=json.loads(args.manifest.read_text())
    with hardware_lock('worker'):
        for name,sha in doc['sources'].items():
            if digest(ROOT/name)!=sha:
                raise ValueError('Changed frozen source: '+name)
        for name,sha in doc['upstream'][args.method]['files'].items():
            if digest(ROOT/'benchmarks/official_baselines/.external'/args.method/name)!=sha:
                raise ValueError('Changed frozen upstream source: '+name)
        if versions()!=doc['runtimes'][args.method]:
            raise ValueError('Changed frozen runtime')
        case=next(c for c in doc['cases'] if c['case_id']==args.case_id)
        inputs=Path(doc['original_root'])/'inputs'/case['case_id']
        for name,sha in case['input_sha256'].items():
            if digest(inputs/name)!=sha:
                raise ValueError('Changed original timing input')
        arrays=[np.load(inputs/(name+'.npy'),allow_pickle=False) for name in ('train','calibration','test')]
        if args.pilot:
            # Compatibility only, explicitly excluded from the sixty full-sequence trials.
            rng=np.random.RandomState(17)
            sizes=(110,100,410) if args.method=='anomaly_transformer' else (400,100,410)
            arrays=[rng.normal(size=(n,6)) for n in sizes]
        fn=dagmm_trial if args.method=='dagmm' else omni_trial if args.method=='omni_anomaly_tsb_ad' else transformer_trial
        measurement,scores=fn(args.method,*arrays,doc['configurations'][args.method],args.seed,args.output)
        if scores.shape!=(len(arrays[2]),) or not np.isfinite(scores[100:-100]).all():
            raise ValueError('Invalid complete common scoring support')
        if args.pilot:
            config=doc['configurations'][args.method]
            if args.method=='omni_anomaly_tsb_ad':
                from research.submission_recovery.run import omni_backend
                reference=lambda:omni_backend(*arrays,config,args.seed).scores
            else:
                name='tab_dagmm' if args.method=='dagmm' else args.method
                adapter=importlib.import_module('benchmarks.official_baselines.adapters.'+name)
                from benchmarks.official_baselines.adapters.common import align_scores
                reference=lambda:align_scores(adapter.official_backend(arrays[0],arrays[2],config,args.seed),len(arrays[2]))
            verify_reference(scores,reference,args.seed)
            measurement['reference_adapter_equivalence']=True
        np.save(args.output/'scores.npy',scores,allow_pickle=False)
        result=dict(measurement,method_id=args.method,case_id=args.case_id,seed=args.seed,
                    manifest_id=doc['manifest_id'],configuration=doc['configurations'][args.method],
                    status='complete',scores_sha256=digest(args.output/'scores.npy'),
                    runtime=versions(),device='cpu',threads=1,pilot=args.pilot)
        with (args.output/'run.json').open('x') as stream:
            json.dump(result,stream,indent=2,sort_keys=True)


if __name__=='__main__':
    main()
