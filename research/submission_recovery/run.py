"""Versioned DAGMM/OmniAnomaly recovery on unchanged held-out sources."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research.evaluation_protocols.campaign import ROOT, _identity, _json, digest
from research.submission_campaign import deep
from research.submission_campaign.studies import load_case

METHODS = ('dagmm', 'omni_anomaly_tsb_ad')
OUTPUT = ROOT/'results/submission_recovery_20261024'
OVERLAY = ROOT/'dependencies/dagmm-overlay'
RUNTIME = ROOT/'dependencies/runtime-evidence.json'
PINS = {'pandas':'1.5.3', 'pytz':'2024.1'}


def reserved_data(train, calibration, window):
    train,calibration=np.asarray(train),np.asarray(calibration)
    if (train.ndim!=2 or calibration.ndim!=2 or train.shape[1]!=calibration.shape[1]
            or not np.isfinite(train).all() or not np.isfinite(calibration).all()):
        raise ValueError('Invalid reserved normal validation arrays')
    if min(len(train),len(calibration))<window:
        raise ValueError('Insufficient reserved validation rows for the fixed window')
    if int(.8*(len(train)+len(calibration)))!=len(train):
        raise ValueError('Reserved validation does not reproduce the fitting boundary')
    return np.vstack([train,calibration])


def worker_environment(method, seed, overlay, environment=None):
    result=dict(os.environ if environment is None else environment)
    paths=[str(ROOT)]
    if method=='dagmm':paths.insert(0,str(overlay))
    if result.get('PYTHONPATH'):paths.append(result['PYTHONPATH'])
    result.update(PYTHONPATH=os.pathsep.join(paths),PYTHONHASHSEED=str(seed),
                  CUDA_VISIBLE_DEVICES='',DYNSPHERE_DEVICE='cpu',OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',LOKY_MAX_CPU_COUNT='1')
    return result


def overlay_files(overlay):
    overlay=Path(overlay)
    return {str(p.relative_to(overlay)):digest(p) for p in sorted(overlay.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}


@contextmanager
def run_lock(folder):
    with (Path(folder)/'.writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def verify_worker_sources(doc,manifest_id):
    if (doc['manifest_id']!=manifest_id
            or _identity({k:v for k,v in doc.items() if k!='manifest_id'})!=manifest_id):
        raise ValueError('Worker frozen manifest identity differs')
    evidence={}
    for section in (doc['base_manifest']['implementation_hashes'],doc['code_hashes'],doc['recovery_source_hashes']):
        for name,expected in section.items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts or digest(ROOT/path)!=expected:
                raise ValueError('Worker actual source differs from frozen manifest')
            evidence[name]=expected
    return evidence


def versions():
    result={'python':sys.version.split()[0]}
    for name in ('numpy','pandas','pytz','scipy','scikit-learn','tensorflow','torch','python-dateutil'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def input_hashes(case,method):
    result=deep.frozen_input_hashes(case,'catch' if method=='omni_anomaly_tsb_ad' else method)
    return result


def validate_record(record,folder,doc,entry,case):
    method,seed=record['method_id'],record['seed']
    identity=_identity({'manifest':doc['manifest_id'],'case':entry['case_id'],'seed':seed,'method':method})
    if (record.get('manifest_id')!=doc['manifest_id'] or record.get('run_identity')!=identity
            or record.get('status')!='complete' or record.get('case_id')!=entry['case_id']
            or record.get('source_family')!=entry['source_family']
            or method not in doc['methods'] or seed not in doc['seeds']):
        raise ValueError('Recovery parent identity differs from frozen manifest')
    hashes=input_hashes(case,method)
    if record.get('input_sha256')!=hashes:raise ValueError('Recovery input hashes differ')
    subdirectory=Path(record.get('worker_subdirectory','worker'))
    if subdirectory.is_absolute() or '..' in subdirectory.parts:
        raise ValueError('Invalid recovery worker path')
    scores,worker=deep.validate_worker(Path(folder)/subdirectory,method,seed,len(case['labels']),hashes)
    if worker.get('configuration')!=doc['configurations'][method]:
        raise ValueError('Recovery configuration differs')
    if worker.get('upstream_source')!=doc['upstream_sources'][method]:
        raise ValueError('Recovery upstream source differs')
    if (worker.get('runtime_evidence')!=doc['runtime_evidence']
            or worker.get('recovery_manifest_id')!=doc['manifest_id']
            or record.get('scores_sha256')!=worker['scores_sha256']):
        raise ValueError('Recovery runtime, manifest, or score evidence differs')
    if 'recovery_source_hashes' in doc:
        expected={**doc['base_manifest']['implementation_hashes'],**doc['code_hashes'],**doc['recovery_source_hashes']}
        if worker.get('verified_source_hashes')!=expected:
            raise ValueError('Recovery worker source evidence differs')
    return scores


def freeze(output,runtime,overlay):
    runtime=json.loads(Path(runtime).read_text())
    files=overlay_files(overlay)
    if not files or _identity(files)!=runtime['overlay_sha256']:
        raise ValueError('Private dependency overlay differs from installation evidence')
    original=deep.freeze_deep(output)
    doc={k:v for k,v in original.items() if k!='manifest_id'}
    doc.update(methods=list(METHODS),configurations={m:doc['configurations'][m] for m in METHODS},
               upstream_sources={m:doc['upstream_sources'][m] for m in METHODS},
               upstream_revisions={m:doc['upstream_revisions'][m] for m in METHODS},
               recovery_revision='isolated-cpu-reserved-normal-validation-v1',
               recovered_from_manifest=original['manifest_id'],runtime_evidence=runtime,
               recovery_source_hashes={str(p.relative_to(ROOT)):digest(p)
                     for p in sorted(Path(__file__).parent.glob('*.py'))})
    doc['configurations']['omni_anomaly_tsb_ad'].update(validation_partition='reserved-normal-calibration-v1')
    doc['manifest_id']=_identity(doc)
    _json(Path(output)/'manifests'/(doc['manifest_id']+'.json'),doc)
    return doc


def omni_backend(train,calibration,test,configuration,seed):
    from benchmarks.official_baselines.adapters.common import BackendOutput
    source=ROOT/'benchmarks/official_baselines/.external/omni_anomaly_tsb_ad'
    sys.path.insert(0,str(source))
    try:from TSB_AD.models.OmniAnomaly import OmniAnomaly
    finally:sys.path.pop(0)
    window=int(configuration.get('window_size',100))
    combined=reserved_data(train,calibration,window)
    model=OmniAnomaly(win_size=window,feats=train.shape[1],
          batch_size=int(configuration.get('batch_size',128)),epochs=int(configuration.get('epochs',20)),
          lr=float(configuration.get('lr',.002)))
    model.fit(combined)
    scores=np.asarray(model.decision_function(test),dtype=float).reshape(-1)
    return BackendOutput(scores=scores,metadata={'backend':'TSB-AD OmniAnomaly',
          'window_size':window,'fit_rows':len(train),'normal_validation_rows':len(calibration),
          'validation_partition':'reserved-normal-calibration-v1','device':str(model.device),
          'normalization':'upstream ReconstructDataset normalizes each partition separately'})


def worker(args):
    doc=json.loads(args.manifest.read_text())
    verified_sources=verify_worker_sources(doc,args.manifest_id)
    configuration=json.loads(args.configuration.read_text())
    source=json.loads(args.source_evidence.read_text())
    runtime=json.loads(args.runtime.read_text())
    if (configuration!=doc['configurations'][args.method] or source!=doc['upstream_sources'][args.method]
            or runtime!=doc['runtime_evidence']):
        raise ValueError('Worker configuration or runtime differs from frozen manifest')
    if versions()!=runtime['versions'][args.method]:
        raise ValueError('Worker dependency versions differ from recorded runtime')
    if args.method=='dagmm' and _identity(overlay_files(args.overlay))!=runtime['overlay_sha256']:
        raise ValueError('Worker dependency overlay content differs')
    if args.method=='omni_anomaly_tsb_ad':
        calibration=np.load(args.input_dir/'calibration.npy',allow_pickle=False)
        adapter=importlib.import_module('benchmarks.official_baselines.adapters.omni_anomaly')
        adapter.official_backend=lambda train,test,config,seed:omni_backend(train,calibration,test,config,seed)
    # The in-memory adapter replacement exists only inside this new worker.
    # No upstream file or live shared adapter is changed.
    deep.worker(args.method,args.input_dir,args.worker_output,args.seed,configuration,source)
    path=args.worker_output/'run.json';record=json.loads(path.read_text())
    if args.method=='omni_anomaly_tsb_ad':
        record['input_sha256']['calibration.npy']=digest(args.input_dir/'calibration.npy')
    record.update(runtime_evidence=runtime,recovery_manifest_id=args.manifest_id,
                  verified_source_hashes=verified_sources)
    _json(path,record)


def controller(args):
    doc=freeze(args.output,args.runtime,args.overlay)
    entries=[e for e in doc['base_manifest']['cases'] if e['source_family']!='PowerSystem'
             and (not args.families or e['source_family'] in args.families)]
    if args.pilot:
        entries=[min((e for e in entries if e['source_family']==f),key=lambda e:e['case_id'])
                 for f in sorted({e['source_family'] for e in entries})]
    jobs=[(e,s,m) for e in entries for s in args.seeds for m in args.methods]
    jobs=jobs[args.shard_index::args.shard_count]
    root=args.output/'campaigns'/doc['manifest_id'][:12]
    print(json.dumps({'manifest_id':doc['manifest_id'],'jobs':len(jobs),'output':str(root),
                      'methods':args.methods,'manuscript_ready':False}),flush=True)
    if args.dry_run:return 0
    root.mkdir(parents=True,exist_ok=True)
    outcomes=[];cached_name=None;case=None;case_error=None;busy=False
    for index,(entry,seed,method) in enumerate(jobs):
        identity=_identity({'manifest':doc['manifest_id'],'case':entry['case_id'],'seed':seed,'method':method})
        folder=root/'runs'/identity;folder.mkdir(parents=True,exist_ok=True)
        if cached_name!=entry['case_id']:
            cached_name=entry['case_id'];case_error=None
            try:case=load_case(entry)
            except ValueError as exc:case_error=str(exc)
        try:
            with run_lock(folder):
                record=execute_job(args,doc,entry,seed,method,identity,folder,root,case,case_error)
        except BlockingIOError:
            busy=True
            print(json.dumps({'method':method,'case':entry['case_id'],'seed':seed,'status':'busy-preserved'}),flush=True)
            continue
        outcomes.append(record)
        print(json.dumps({'done':index+1,'of':len(jobs),'method':method,'case':entry['case_id'],
                          'seed':seed,'status':record['status']}),flush=True)
    _json(root/f'completion-shard-{args.shard_index}.json',{'manifest_id':doc['manifest_id'],
          'outcomes':outcomes,'busy_preserved':busy,'manuscript_ready':False})
    return 1 if busy or any(r['status']=='failed' for r in outcomes) else 0


def execute_job(args,doc,entry,seed,method,identity,folder,root,case,case_error):
    record={'manifest_id':doc['manifest_id'],'run_identity':identity,'case_id':entry['case_id'],
            'source_family':entry['source_family'],'method_id':method,'seed':seed,'status':'running',
            'run_directory':str(folder.resolve()),'manuscript_ready':False}
    if case_error:
        record.update(status='incompatible',error=case_error)
    else:
        try:
                old=json.loads((folder/'run.json').read_text()) if (folder/'run.json').is_file() else None
                if old and old['status']=='complete':
                    validate_record(old,folder,doc,entry,case);record=old
                else:
                    number=1
                    while (folder/f'attempt-{number:03d}').exists():number+=1
                    attempt=folder/f'attempt-{number:03d}';attempt.mkdir()
                    record['worker_subdirectory']=attempt.name
                    inputs=root/'inputs'/_identity(entry)[:16];inputs.mkdir(parents=True,exist_ok=True)
                    arrays={'train.npy':case['train'],'test.npy':case['test']}
                    if method=='omni_anomaly_tsb_ad':arrays['calibration.npy']=case['calibration']
                    from benchmarks.official_baselines.contract import _atomic_save_array
                    for name,values in arrays.items():
                        target=inputs/name
                        if not target.exists():_atomic_save_array(target,values)
                        elif not np.array_equal(np.load(target,allow_pickle=False),values):
                            raise ValueError('Recovery cached input differs from frozen source')
                    configuration=attempt/'configuration.json';_json(configuration,doc['configurations'][method])
                    source=attempt/'source-evidence.json';_json(source,doc['upstream_sources'][method])
                    command=[str(ROOT/'benchmarks/official_baselines/.envs'/method/'bin/python'),
                         '-m','research.submission_recovery.run','--worker','--method',method,'--seed',str(seed),
                         '--input-dir',str(inputs),'--worker-output',str(attempt),'--configuration',str(configuration),
                         '--source-evidence',str(source),'--runtime',str(args.runtime),'--overlay',str(args.overlay),
                         '--manifest-id',doc['manifest_id'],'--manifest',
                         str(args.output/'manifests'/(doc['manifest_id']+'.json'))]
                    _json(folder/'run.json',record)
                    with (attempt/'stdout.log').open('w') as out,(attempt/'stderr.log').open('w') as err:
                        completed=subprocess.run(command,cwd=ROOT,env=worker_environment(method,seed,args.overlay),
                              stdout=out,stderr=err,timeout=args.timeout_seconds)
                    if completed.returncode:raise RuntimeError(f'Worker exit {completed.returncode}; see {attempt.name}/stderr.log')
                    metadata=json.loads((attempt/'run.json').read_text())
                    record.update(status='complete',input_sha256=metadata['input_sha256'],
                         scores_sha256=metadata['scores_sha256'],backend_seconds=metadata['backend_seconds'])
                    scores=validate_record(record,folder,doc,entry,case)
                    mask=deep.common_mask(len(scores));y=case['labels'][mask];s=scores[mask]
                    record['metrics']={'auprc':float(average_precision_score(y,s)),
                         'auroc':float(roc_auc_score(y,s)),'pauc_fpr_05':float(roc_auc_score(y,s,max_fpr=.05))}
                    record['evaluated_rows']=int(mask.sum())
                    _json(attempt/'parent-record.json',record)
        except Exception as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            if record.get('worker_subdirectory'):
                _json(folder/record['worker_subdirectory']/'parent-record.json',record)
    _json(folder/'run.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--seed',type=int,default=37);parser.add_argument('--manifest-id')
    for name in ('input-dir','worker-output','configuration','source-evidence','manifest'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--runtime',type=Path,default=RUNTIME);parser.add_argument('--overlay',type=Path,default=OVERLAY)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[37,38,39])
    parser.add_argument('--families',nargs='+',choices=['MSL','SMAP','SMD','BATADAL','HAI-20.07'])
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--timeout-seconds',type=int,default=14400)
    args=parser.parse_args()
    if args.worker:worker(args);return 0
    if not 0<=args.shard_index<args.shard_count:parser.error('Invalid shard')
    if len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<={37,38,39}:
        parser.error('Use distinct frozen seeds from 37, 38, 39')
    if len(set(args.methods))!=len(args.methods):parser.error('Methods must be distinct')
    return controller(args)


if __name__=='__main__':raise SystemExit(main())
