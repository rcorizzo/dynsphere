"""Isolated, source-traceable deep comparisons on the frozen held-out inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

from research.evaluation_protocols.campaign import ROOT,_identity,_json,digest
from .studies import BASE,BASE_ID,OUT,load_case


EXTERNAL = ('mtad_gat','catch','dagmm','omni_anomaly_tsb_ad','tranad','anomaly_transformer')
MODULES = {'mtad_gat':'mtad_gat','catch':'catch','dagmm':'tab_dagmm',
           'omni_anomaly_tsb_ad':'omni_anomaly','tranad':'tranad','anomaly_transformer':'anomaly_transformer'}


def upstream_snapshot(directory,expected_revision):
    directory=Path(directory)
    def git(*arguments):
        return subprocess.check_output(['git','-C',str(directory),*arguments])
    revision=git('rev-parse','HEAD').decode().strip()
    if revision!=expected_revision:
        raise ValueError('Actual upstream revision differs from registry revision')
    tracked=set(git('ls-files','-z').decode().strip('\0').split('\0'))-{''}
    untracked=set(git('ls-files','--others','--exclude-standard','-z').decode().strip('\0').split('\0'))-{''}
    source_suffixes={'.py','.pyx','.pyi','.sh','.json','.yaml','.yml','.toml','.ini','.cfg'}
    extra={p for p in untracked if Path(p).suffix in source_suffixes}
    files={name:digest(directory/name) for name in sorted(tracked|extra)}
    return {'revision':revision,'tree_sha256':_identity(files),'files':files,
            'untracked_source_files':sorted(extra)}


def declared_revisions():
    rows=json.loads((ROOT/'benchmarks/official_baselines/registry.json').read_text())['implementations']
    return {r['method_id']:r['revision'] for r in rows if r['method_id'] in EXTERNAL}


def frozen_input_hashes(case,method):
    arrays={'train.npy':case['train'],'test.npy':case['test']}
    if method=='catch':arrays['calibration.npy']=case['calibration']
    result={}
    for name,array in arrays.items():
        data=io.BytesIO();np.save(data,array,allow_pickle=False)
        result[name]=hashlib.sha256(data.getvalue()).hexdigest()
    return result


def common_mask(length):
    if length <= 200:
        raise ValueError('Insufficient common observed support')
    result=np.zeros(length,dtype=bool);result[100:-100]=True
    return result


def prime_and_seed(method,seed):
    """Upstream import side effects finish before the requested random seed."""
    if method=='tranad':
        from benchmarks.official_baselines.adapters import tranad
        source=tranad.source_directory('tranad');sys.path.insert(0,str(source))
        try:
            with tranad._upstream_import_arguments():
                importlib.import_module('src.models')
        finally:
            sys.path.pop(0)
    elif method=='catch':
        from benchmarks.official_baselines.adapters.catch import _load_corrected_catch
        _load_corrected_catch()
    random.seed(seed);np.random.seed(seed)
    if method!='dagmm':
        import torch
        torch.set_num_threads(1);torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True


def configurations():
    rows=json.loads((ROOT/'benchmarks/official_baselines/registry.json').read_text())['implementations']
    result={r['method_id']:dict(r['default_hyperparameters']) for r in rows if r['method_id'] in EXTERNAL}
    result['tranad'].update(pipeline_revision='audited-v2')
    result['anomaly_transformer'].update(training_revision='released-budget-v2',epochs=10,d_model=512,n_heads=8)
    result['omni_anomaly_tsb_ad'].update(implementation_backend='tsb-ad')
    result['catch'].update(validation_partition='reserved-normal-calibration-v1')
    return result


def catch_fit_frame(train,calibration):
    if int(.8*(len(train)+len(calibration)))!=len(train) or len(calibration)<100:
        raise ValueError('CATCH reserved validation boundary is incompatible with its fixed window')
    return pd.DataFrame(np.vstack([train,calibration]),columns=[f'feature_{i}' for i in range(train.shape[1])])


def worker(method,input_dir,output,seed,configuration,expected_source=None):
    from benchmarks.official_baselines.adapters.common import align_scores
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    actual_source=upstream_snapshot(ROOT/'benchmarks/official_baselines/.external'/method,declared_revisions()[method])
    if expected_source is not None and actual_source!=expected_source:
        raise ValueError('Upstream source differs from frozen manifest')
    files={n:Path(input_dir)/n for n in ['train.npy','test.npy']}
    reserved=(method=='catch' and configuration.get('validation_partition')=='reserved-normal-calibration-v1')
    if reserved:files['calibration.npy']=Path(input_dir)/'calibration.npy'
    hashes={n:digest(p) for n,p in files.items()}
    train,test=[np.load(files[n],allow_pickle=False) for n in ['train.npy','test.npy']]
    if (train.ndim!=2 or test.ndim!=2 or train.shape[1]!=test.shape[1]
            or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError('Invalid label-free worker inputs')
    prime_and_seed(method,seed)
    module=importlib.import_module('benchmarks.official_baselines.adapters.'+MODULES[method])
    started=time.perf_counter()
    if reserved:
        from benchmarks.official_baselines.adapters.catch import _load_corrected_catch
        from benchmarks.official_baselines.adapters.common import BackendOutput
        calibration=np.load(files['calibration.npy'],allow_pickle=False)
        if (calibration.ndim!=2 or calibration.shape[1]!=train.shape[1]
                or not np.isfinite(calibration).all()):
            raise ValueError('Invalid label-free normal validation input')
        combined=catch_fit_frame(train,calibration)
        model=_load_corrected_catch()(num_epochs=int(configuration['epochs']),
               seq_len=int(configuration['window_size']),batch_size=int(configuration['batch_size']))
        model.detect_fit(combined,combined)
        raw=np.asarray(model.detect_score(pd.DataFrame(test,columns=combined.columns))[0],dtype=float).reshape(-1)
        raw=raw[:len(test)]
        result=BackendOutput(scores=raw,tail_rows=len(test)-len(raw),metadata={
               'backend':'author CATCH; coverage-count-v1','validation_partition':'reserved-normal-calibration-v1',
               'fit_rows':len(train),'normal_validation_rows':len(calibration),'device':str(model.device)})
    else:
        result=module.official_backend(train,test,configuration,seed)
    if result.warmup_rows>100 or result.tail_rows>100:
        raise ValueError('Backend unobserved support exceeds common trim')
    scores=align_scores(result,len(test))
    np.save(output/'scores.npy',scores,allow_pickle=False)
    _json(output/'run.json',{'status':'complete','method_id':method,'seed':seed,
          'input_sha256':hashes,'scores_sha256':digest(output/'scores.npy'),
          'backend_seconds':time.perf_counter()-started,'metadata':dict(result.metadata),
          'warmup_rows':result.warmup_rows,'tail_rows':result.tail_rows,
          'configuration':configuration,'numpy_version':np.__version__,'python':sys.version,
          'upstream_source':actual_source,
          'seed_protocol':'prime-upstream-imports-then-seed-python-numpy-torch-v1'})


def validate_worker(output,method,seed,rows,input_hashes):
    output=Path(output);r=json.loads((output/'run.json').read_text())
    if (r.get('status')!='complete' or r.get('method_id')!=method or r.get('seed')!=seed
            or r.get('input_sha256')!=input_hashes
            or digest(output/'scores.npy')!=r.get('scores_sha256')):
        raise ValueError('Worker identity or artifact integrity mismatch')
    scores=np.load(output/'scores.npy',allow_pickle=False)
    if scores.shape!=(rows,) or not np.isfinite(scores).all():
        raise ValueError('Invalid or unaligned worker scores')
    return scores,r


def freeze_deep(output):
    base=json.loads((BASE/'manifests'/(BASE_ID+'.json')).read_text())
    files=[Path(__file__),Path(__file__).with_name('studies.py'),ROOT/'research/evaluation_protocols/campaign.py']
    files+=list((ROOT/'benchmarks/official_baselines/adapters').glob('*.py'))
    doc={'base_manifest':base,'methods':EXTERNAL,'seeds':[37,38,39],'configurations':configurations(),
         'code_hashes':{str(p.relative_to(ROOT)):digest(p) for p in files},
         'support':'first-and-last-100-observations-excluded-v1',
         'exclude_power':'unverified temporal ordering and normal-selection boundaries',
         'upstream_revisions':declared_revisions(),
         'upstream_sources':{m:upstream_snapshot(ROOT/'benchmarks/official_baselines/.external'/m,r)
                             for m,r in declared_revisions().items()},
         'manuscript_ready':False}
    doc['manifest_id']=_identity(doc)
    _json(Path(output)/'deep-manifests'/(doc['manifest_id']+'.json'),doc)
    return doc


def controller(args):
    doc=freeze_deep(args.output)
    selected=[e for e in doc['base_manifest']['cases'] if e['source_family']!='PowerSystem'
              and (not args.families or e['source_family'] in args.families)]
    if args.pilot:
        selected=[min((e for e in selected if e['source_family']==f),key=lambda e:e['case_id'])
                  for f in sorted({e['source_family'] for e in selected})]
    jobs=[(entry,seed,method) for entry in selected for seed in args.seeds for method in args.methods]
    jobs=jobs[args.shard_index::args.shard_count]
    summary={'manifest_id':doc['manifest_id'],'jobs':len(jobs),'shard_index':args.shard_index,
             'shard_count':args.shard_count,'methods':args.methods,'manuscript_ready':False}
    print(json.dumps(summary),flush=True)
    if args.dry_run:return 0
    root=Path(args.output)/'deep'/doc['manifest_id'][:12];root.mkdir(parents=True,exist_ok=True)
    outcomes=[];started=time.perf_counter();case_cache={}
    for index,(entry,seed,method) in enumerate(jobs):
        if args.budget_seconds and time.perf_counter()-started>args.budget_seconds:
            summary['stopped_at_budget']=True;break
        identity=_identity({'manifest':doc['manifest_id'],'case':entry['case_id'],'seed':seed,'method':method})
        folder=root/'runs'/identity;folder.mkdir(parents=True,exist_ok=True)
        record={'run_identity':identity,'manifest_id':doc['manifest_id'],'case_id':entry['case_id'],
                'source_family':entry['source_family'],'method_id':method,'seed':seed,'status':'running',
                'run_directory':str(folder.resolve()),'manuscript_ready':False}
        try:
            if entry['case_id'] not in case_cache:
                case_cache={entry['case_id']:load_case(entry)}
            case=case_cache[entry['case_id']]
            input_dir=root/'inputs'/_identity(entry)[:16];input_dir.mkdir(parents=True,exist_ok=True)
            inputs=[('train.npy',case['train']),('test.npy',case['test'])]
            if method=='catch':inputs.append(('calibration.npy',case['calibration']))
            for name,array in inputs:
                target=input_dir/name
                if not target.is_file():
                    from benchmarks.official_baselines.contract import _atomic_save_array
                    _atomic_save_array(target,array)
                elif not np.array_equal(np.load(target,allow_pickle=False),array):
                    raise ValueError('Cached model input differs from frozen source')
            hashes={n:digest(input_dir/n) for n,_ in inputs}
            worker_output=folder/'worker'
            if (folder/'run.json').is_file() and json.loads((folder/'run.json').read_text()).get('status')=='complete':
                scores,metadata=validate_worker(worker_output,method,seed,len(case['test']),hashes)
                if metadata.get('configuration')!=doc['configurations'][method]:
                    raise ValueError('Saved worker configuration differs')
                if metadata.get('upstream_source')!=doc['upstream_sources'][method]:
                    raise ValueError('Saved upstream source differs')
                record=json.loads((folder/'run.json').read_text())
                if record.get('run_identity')!=identity or record.get('manifest_id')!=doc['manifest_id']:
                    raise ValueError('Saved parent identity differs')
            else:
                config_path=folder/'configuration.json';_json(config_path,doc['configurations'][method])
                evidence_path=folder/'source-evidence.json';_json(evidence_path,doc['upstream_sources'][method])
                python=ROOT/'benchmarks/official_baselines/.envs'/method/'bin/python'
                command=[str(python),'-m','research.submission_campaign.deep','--worker',
                         '--method',method,'--seed',str(seed),'--input-dir',str(input_dir),
                         '--worker-output',str(worker_output),'--configuration',str(config_path),
                         '--source-evidence',str(evidence_path)]
                env={**os.environ,'PYTHONHASHSEED':str(seed),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                     'LOKY_MAX_CPU_COUNT':'1','MPLCONFIGDIR':str(folder/'mpl'),'XDG_CACHE_HOME':str(folder/'cache'),
                     'DGLDEFAULTDIR':str(folder/'dgl'),'DGLBACKEND':'pytorch'}
                _json(folder/'run.json',record)
                with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
                    result=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                                          timeout=args.timeout_seconds)
                if result.returncode:raise RuntimeError(f'Worker exit {result.returncode}; see stderr.log')
                scores,metadata=validate_worker(worker_output,method,seed,len(case['test']),hashes)
                if metadata.get('configuration')!=doc['configurations'][method]:
                    raise ValueError('Worker configuration differs')
                if metadata.get('upstream_source')!=doc['upstream_sources'][method]:
                    raise ValueError('Worker upstream source differs')
                mask=common_mask(len(scores));labels=case['labels'][mask];values=scores[mask]
                if set(np.unique(labels))!={0,1}:raise ValueError('Both classes required on common support')
                record.update(status='complete',metrics={'auprc':float(average_precision_score(labels,values)),
                      'auroc':float(roc_auc_score(labels,values)),
                      'pauc_fpr_05':float(roc_auc_score(labels,values,max_fpr=.05))},
                      input_sha256=hashes,scores_sha256=metadata['scores_sha256'],
                      evaluated_rows=int(mask.sum()),backend_seconds=metadata['backend_seconds'])
        except ValueError as exc:
            record.update(status='incompatible',error=str(exc))
        except subprocess.TimeoutExpired:
            record.update(status='timeout',error=f'Exceeded {args.timeout_seconds} seconds')
        except Exception as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        _json(folder/'run.json',record);outcomes.append(record)
        print(json.dumps({'done':index+1,'of':len(jobs),'case_id':entry['case_id'],
                          'method':method,'seed':seed,'status':record['status']}),flush=True)
        _json(root/f'completion-shard-{args.shard_index}.json',
              {**summary,'processed':len(outcomes),'outcomes':outcomes,'status':'running'})
    summary.update(processed=len(outcomes),status='partial' if len(outcomes)!=len(jobs) else 'complete',
                   outcomes=outcomes)
    _json(root/f'completion-shard-{args.shard_index}.json',summary)
    return 1 if any(r['status'] in ['failed','timeout'] for r in outcomes) else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--method',choices=EXTERNAL)
    parser.add_argument('--input-dir',type=Path);parser.add_argument('--worker-output',type=Path)
    parser.add_argument('--configuration',type=Path);parser.add_argument('--seed',type=int,default=37)
    parser.add_argument('--source-evidence',type=Path)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--methods',nargs='+',choices=EXTERNAL,default=['mtad_gat','catch'])
    parser.add_argument('--families',nargs='+',choices=['MSL','SMAP','SMD','BATADAL','HAI-20.07'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[37,38,39]);parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--timeout-seconds',type=int,default=14400)
    parser.add_argument('--budget-seconds',type=int,default=0)
    parser.add_argument('--shard-index',type=int,default=0);parser.add_argument('--shard-count',type=int,default=1)
    args=parser.parse_args()
    if args.worker:
        evidence=json.loads(args.source_evidence.read_text()) if args.source_evidence else None
        worker(args.method,args.input_dir,args.worker_output,args.seed,json.loads(args.configuration.read_text()),evidence)
        return 0
    if not 0<=args.shard_index<args.shard_count:parser.error('Invalid shard index/count')
    return controller(args)


if __name__=='__main__':raise SystemExit(main())
