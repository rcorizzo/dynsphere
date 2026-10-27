"""Immutable one-seed pilot and technically gated three-seed execution."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile

import numpy as np

from . import core

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/quality_cost_20261026/five-family-cpu-v1'
EXT=ROOT/'results/quality_cost_20261026/four-methods-cpu-v1'
FOCAL='dynsphere_euclidean_one_regime'
LOCAL=[FOCAL,'pca_reconstruction','kmeans_distance','linear_ridge_ar1',
       'dynsphere_euclidean','dynsphere_one_regime','dynsphere']
DEEP=['dagmm','tranad','omni_anomaly_tsb_ad','mtad_gat','catch','anomaly_transformer']
METHODS=LOCAL+DEEP
FAMILIES=['BATADAL','HAI-20.07','MSL','SMAP','SMD']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Complete the temporary inode before atomically linking an exclusive name.
    fd,temporary=tempfile.mkstemp(prefix='.record-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        os.unlink(temporary)


def verify_files(files):
    for path,sha in files.items():
        if digest(path)!=sha:
            raise ValueError('Changed frozen file: '+path)


def existing_state(directory):
    directory=Path(directory)
    if (directory/'result.json').exists():
        return json.loads((directory/'result.json').read_text())['status']
    return 'interrupted' if (directory/'started.json').exists() else None


def require_stopped(owner):
    owner=Path(owner)
    if not owner.exists():return
    pid=int(json.loads(owner.read_text())['pid'])
    try:os.kill(pid,0)
    except ProcessLookupError:return
    raise RuntimeError('Refusing recovery while prior worker PID may be live: '+str(pid))


def recover_terminal(directory,expected):
    directory=Path(directory);require_stopped(directory/'worker/worker-owner.json')
    path=directory/'worker/run.json'
    if not path.exists():return dict(expected,status='interrupted',technical_verified=False)
    record=json.loads(path.read_text())
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Changed orphan terminal identity')
    if record['status']=='complete':
        verify_files(record['event_files'])
        if len(record['event_files'])!=6:raise ValueError('Incomplete orphan event coverage')
        if digest(directory/'worker/full-scores.npy')!=record['full_scores_sha256']:raise ValueError('Changed orphan scores')
    return record


def execute(command,directory,timeout,env):
    directory=Path(directory)
    with (directory/'stdout.log').open('x') as out,(directory/'stderr.log').open('x') as err:
        process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,**env},stdout=out,stderr=err,start_new_session=True)
        try:
            write_new(directory/'worker-owner.json',dict(pid=process.pid,controller_pid=os.getpid()))
            code=process.wait(timeout=timeout)
        except BaseException as error:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
            if isinstance(error,subprocess.TimeoutExpired):raise TimeoutError('Hard whole-worker budget exceeded') from error
            raise
    if code:raise RuntimeError('Worker exit code '+str(code))


@contextmanager
def controller_signals():
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop(signum,frame):raise KeyboardInterrupt('Controller interrupted')
    for s in previous:signal.signal(s,stop)
    try:yield
    finally:
        for s,handler in previous.items():signal.signal(s,handler)


def executable(method):
    return ROOT/('benchmarks/official_baselines/.envs/'+method+'/bin/python') if method in DEEP else ROOT/'.venv/bin/python'


def environment():
    from research.quality_cost.run import cpu_environment
    return dict(cpu_environment(),TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',DGLBACKEND='pytorch')


def runtime(method):
    command=[str(executable(method)),'-c',
             'import json;from research.quality_cost_extension.trials import versions;print(json.dumps(versions()))']
    return json.loads(subprocess.check_output(command,cwd=ROOT,env=environment(),text=True))


def freeze(output):
    from research.quality_cost.run import validate_manifest
    from research.quality_cost_extension.run import validate
    old=json.loads((OLD/'manifest.json').read_text()); ext=json.loads((EXT/'manifest.json').read_text())
    validate_manifest(old);validate(ext)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Partial preparation is preserved; choose a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    bindings={str(OLD/'manifest.json'):digest(OLD/'manifest.json'),str(EXT/'manifest.json'):digest(EXT/'manifest.json')}
    for doc in (old,ext):
        bindings.update({str(ROOT/name):sha for name,sha in doc['sources'].items()})
        for m,evidence in doc['upstream'].items():
            bindings.update({str(ROOT/'benchmarks/official_baselines/.external'/m/name):sha
                             for name,sha in evidence['files'].items()})
    bindings.update({str(p):digest(p) for p in Path(__file__).parent.glob('*.py')})
    cases=[]
    for case in old['cases']:
        inputs=OLD/'inputs'/case['case_id']
        bindings.update({str(inputs/name):sha for name,sha in case['input_sha256'].items()})
        train=np.load(inputs/'train.npy',allow_pickle=False);test=np.load(inputs/'test.npy',allow_pickle=False)
        labels=np.load(inputs/'labels.npy',allow_pickle=False);entities=np.load(inputs/'test_entities.npy',allow_pickle=False)
        sd=np.std(train,axis=0)
        starts=core.select_windows(test,labels,entities,sd)
        if len(train)<256:
            raise ValueError('Insufficient training donor observations')
        refs=[np.pad(block,((128,128),(0,0)),mode='reflect') for block in (train[:256],train[-256:])]
        events=[]
        for index,start in enumerate(starts):
            seed=int(hashlib.sha256((case['case_id']+':'+str(index)).encode()).hexdigest()[:8],16)
            clean=test[start:start+512].copy()
            for kind in ('mean','variance','reversal'):
                values,truth=core.inject(clean,sd,kind,seed)
                path=output/'events'/case['case_id']/(str(index)+'-'+kind+'.npz')
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:
                    np.savez_compressed(stream,x=values,clean=clean,truth=truth,reference0=refs[0],reference1=refs[1],training_sd=sd)
                bindings[str(path)]=digest(path)
                events.append(dict(event_id=str(index)+'-'+kind,path=str(path),sha256=digest(path),
                                   start=start,kind=kind,seed=seed,edit_rms=float(np.sqrt(np.mean((values-clean)**2))),
                                   changed_observations=int(truth.any(1).sum()),changed_features=np.flatnonzero(truth.any(0)).tolist()))
        cases.append(dict(case_id=case['case_id'],source_family=case['source_family'],
                          inputs=str(inputs),events=events,features=case['features']))
    configs={**old['configurations'],**ext['configurations']}
    original_records={}
    for case in cases:
        for method in METHODS:
            for seed in (37,38,39):
                study=OLD if method in old['methods'] else EXT
                directory=study/'runs'/case['case_id']/method/('seed-'+str(seed))
                record=json.loads((directory/'result.json').read_text())
                bindings[str(directory/'result.json')]=digest(directory/'result.json')
                key=case['case_id']+'|'+method+'|'+str(seed)
                item=dict(status=record['status'])
                if record['status']=='complete':
                    score=directory/record['worker_subdirectory']/'scores.npy'
                    bindings[str(score)]=digest(score);item['scores']=str(score)
                    if method=='dagmm':
                        checkpoint=score.parent/'fitted-checkpoint';item['checkpoint']=str(checkpoint)
                        for p in checkpoint.iterdir():
                            if p.is_file():bindings[str(p)]=digest(p)
                original_records[key]=item
    doc=dict(methods=METHODS,cases=cases,seeds=[37,38,39],pilot_seed=37,
             configurations={m:configs[m] for m in METHODS},bindings=bindings,
             original_records=original_records,runtimes={m:runtime(m) for m in METHODS},
             original_manifest_id=old['manifest_id'],extension_manifest_id=ext['manifest_id'],
             fit_stage_timeout=3600,explanation_stage_timeout=3600,whole_worker_timeout=7230,
             protocol='controlled-context512-group-occlusion-v1',target='max score rows240:272',
             donors='first/last256 training rows, reflect-padded outside the128:384 intervention region',
             claims='exploratory perturbation fidelity and edit localization; no physical root-cause or human-utility claim',
             manuscript_ready=False)
    verify_files(bindings);doc['manifest_id']=identity(doc)
    write_new(output/'manifest.json',doc)
    return doc


def validate(doc,method=None):
    if identity({k:v for k,v in doc.items() if k!='manifest_id'})!=doc['manifest_id']:
        raise ValueError('Changed manifest identity')
    verify_files(doc['bindings'])
    if method is not None:
        from research.quality_cost_extension.trials import versions
        if versions()!=doc['runtimes'][method]:raise ValueError('Changed worker runtime')


@contextmanager
def deadline(seconds):
    previous=signal.getsignal(signal.SIGALRM)
    def expired(signum,frame):
        raise TimeoutError('Frozen stage budget exceeded')
    signal.signal(signal.SIGALRM,expired);signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


def worker(args):
    from research.quality_cost_extension.trials import hardware_lock
    from .models import fit_original,restore_dagmm,native_diagnostics
    doc=json.loads(args.manifest.read_text());case=next(c for c in doc['cases'] if c['case_id']==args.case_id)
    result=dict(manifest_id=doc['manifest_id'],method_id=args.method,case_id=args.case_id,
                source_family=case['source_family'],seed=args.seed,status='failed',technical_verified=False)
    phase='validation';event_files={}
    if args.controller_pid is not None and os.getppid()!=args.controller_pid:
        raise RuntimeError('Original controller no longer owns this worker')
    with hardware_lock('worker'):
        try:
            validate(doc,args.method)
            key=args.case_id+'|'+args.method+'|'+str(args.seed);old=doc['original_records'][key]
            phase='fit_fullscore_roundtrip'
            with deadline(doc['fit_stage_timeout']):
                if args.method=='dagmm':
                    fitted=restore_dagmm(old['checkpoint'],args.seed)
                    full=fitted.score(np.load(Path(case['inputs'])/'test.npy',allow_pickle=False))
                    evidence={'archived_checkpoint_reused':True,'checkpoint':old['checkpoint']}
                else:
                    fitted,full,evidence=fit_original(args.method,args.seed,case['inputs'],doc['configurations'][args.method],args.output)
                    with (args.output/'fitted-model.pkl').open('xb') as stream:pickle.dump(fitted.model,stream,protocol=4)
                if old['status']=='complete':
                    np.testing.assert_allclose(full[100:-100],np.load(old['scores'],allow_pickle=False)[100:-100],rtol=1e-7,atol=1e-9)
                if not np.isfinite(full[100:-100]).all():raise ValueError('Invalid full-sequence scores')
                np.save(args.output/'full-scores.npy',full,allow_pickle=False)
                write_new(args.output/'fit-evidence.json',dict(evidence,full_score_verified=True,
                           original_score_comparison=old['status']=='complete',runtime=doc['runtimes'][args.method]))
                with (args.output/'scoring-rng.pkl').open('xb') as stream:pickle.dump(fitted.score.state,stream,protocol=4)
            phase='explanation'
            with deadline(doc['explanation_stage_timeout']):
                for event in case['events']:
                    with np.load(event['path'],allow_pickle=False) as data:
                        explanation=core.explain_event(fitted.target,data['x'],data['clean'],data['truth'],
                            [data['reference0'],data['reference1']],data['training_sd'],event['seed'])
                        if args.method.startswith('dynsphere'):
                            explanation['native']=native_diagnostics(fitted.model.model,data['x'])
                    path=args.output/('event-'+event['event_id']+'.json')
                    write_new(path,dict(event_id=event['event_id'],input_sha256=event['sha256'],**explanation))
                    event_files[str(path)]=digest(path)
                    print(json.dumps(dict(event=event['event_id'],case=args.case_id,method=args.method,seed=args.seed)),flush=True)
            result.update(status='complete',technical_verified=True,event_files=event_files,
                          full_scores_sha256=digest(args.output/'full-scores.npy'),
                          fitted_artifacts={str(p):digest(p) for p in args.output.iterdir()
                                            if p.name in ('fitted-model.pkl','fit-evidence.json','scoring-rng.pkl')})
        except TimeoutError as error:
            result.update(status='timeout',phase=phase,error=repr(error),event_files=event_files)
        except Exception as error:
            result.update(status='failed',phase=phase,error=repr(error),event_files=event_files)
            import traceback;traceback.print_exc()
        write_new(args.output/'run.json',result)


def technical_gate(records,methods,cases):
    expected={(c['case_id'],m,37) for c in cases for m in methods}
    seen={(r['case_id'],r['method_id'],r['seed']) for r in records}
    problems=[]
    if seen!=expected or len(records)!=len(expected):problems.append('Missing or duplicate pilot outcomes')
    if any(r['status'] not in ('complete','timeout') for r in records):problems.append('Implementation failure or interruption')
    if any(r['status']=='complete' and not r.get('technical_verified') for r in records):problems.append('Unverified completed worker')
    complete=[r for r in records if r['status']=='complete' and r.get('technical_verified')]
    if {r['source_family'] for r in complete if r['method_id']==FOCAL}!={c['source_family'] for c in cases}:
        problems.append('Original E1 lacks five-family technical completion')
    if not any(r['method_id'] in ('pca_reconstruction','kmeans_distance','linear_ridge_ar1') for r in complete):
        problems.append('No verified statistical comparator')
    if not any(r['method_id'] in ('tranad','catch','mtad_gat','omni_anomaly_tsb_ad','dagmm','anomaly_transformer') for r in complete):
        problems.append('No verified deep comparator')
    return dict(passed=not problems,problems=problems,promotion_criterion='technical-only',
                recorded=len(records),timeouts=sum(r['status']=='timeout' for r in records))


def paired_metric(rows,metric,method,reference,seeds,families):
    valid={}
    for m in (method,reference):
        valid[m]={}
        for r in rows:
            if r['method_id']!=m or r.get(metric) is None:continue
            valid[m].setdefault((r['source_family'],r['case_id']),{})[r['seed']]=r[metric]
    keys=sorted(k for k in set(valid[method])&set(valid[reference])
                if set(valid[method][k])==set(seeds) and set(valid[reference][k])==set(seeds))
    common=sorted({k[0] for k in keys})
    def mean(m):
        return None if not keys else float(np.mean([np.mean([np.mean(list(valid[m][k].values()))
                    for k in keys if k[0]==f]) for f in common]))
    return dict(method=method,reference=reference,metric=metric,families=common,cases=[k[1] for k in keys],
                full_family_support=set(common)==set(families),e1=mean(method),competitor=mean(reference))


def verify_event(event):
    for part in event['resolutions'].values():
        phi=np.asarray(part['attributions']);noisy=np.asarray(part['noisy_attributions'])
        random_phi=np.asarray(part['random_attributions']);random_noisy=np.asarray(part['random_noisy_attributions'])
        masks=np.asarray(part['heldout_masks']);truth=np.asarray(part['truth'])
        expected=dict(infidelity=core.infidelity(phi,masks,part['heldout_changes']),
            second_reference_infidelity=core.infidelity(phi,masks,part['second_reference_changes']),
            random_infidelity=core.infidelity(random_phi,masks,part['heldout_changes']),
            ap=core.average_precision(np.maximum(phi,0),truth),
            precision_at_2=core.precision_at_k(np.maximum(phi,0),truth),
            random_ap=core.average_precision(np.maximum(random_phi,0),truth),
            random_precision_at_2=core.precision_at_k(np.maximum(random_phi,0),truth),
            stability=core.rank_agreement(phi,noisy),random_stability=core.rank_agreement(random_phi,random_noisy))
        for name,value in expected.items():
            actual=part[name]
            if (value is None)!=(actual is None) or (value is not None and not np.isclose(value,actual,rtol=1e-10,atol=1e-10)):
                raise ValueError('Changed stored explanation metric: '+name)


METRICS=('infidelity','second_reference_infidelity','random_infidelity','ap','precision_at_2',
         'random_ap','random_precision_at_2','stability','random_stability')


def summarize_events(events):
    row={'metric_coverage':{}}
    for axis in ('sensor','time'):
        for metric in METRICS:
            values=[e['resolutions'][axis][metric] for e in events];key=axis+'_'+metric
            row[key]=None if any(v is None for v in values) else float(np.mean(values))
            row['metric_coverage'][key]=dict(available=sum(v is not None for v in values),total=len(values))
        for metric,flag in [('ap','edit_score_increase'),('precision_at_2','edit_score_increase'),('stability','score_stable')]:
            values=[e['resolutions'][axis][metric] for e in events if e[flag]]
            suffix='score_increase_subset' if flag=='edit_score_increase' else 'stable_score_subset'
            row[axis+'_'+metric+'_'+suffix]=None if not values or any(v is None for v in values) else float(np.mean(values))
    row.update(explanation_seconds=sum(e['explanation_seconds'] for e in events),events=len(events),
        score_increase_events=sum(e['edit_score_increase'] for e in events),
        stable_score_events=sum(e['score_stable'] for e in events))
    return row


def common_metric(rows,metric,methods,seeds,families):
    supports={m:{r['case_id'] for r in rows if r['method_id']==m and r.get(metric) is not None
                 and {t['seed'] for t in rows if t['method_id']==m and t['case_id']==r['case_id']
                      and t.get(metric) is not None}==set(seeds)} for m in methods}
    common=set.intersection(*(supports[m] for m in methods));kept=[r for r in rows if r['case_id'] in common]
    fs=sorted({r['source_family'] for r in kept})
    means={m:(None if not fs else float(np.mean([np.mean([r[metric] for r in kept
                if r['method_id']==m and r['source_family']==f]) for f in fs]))) for m in methods}
    return dict(metric=metric,cases=sorted(common),families=fs,full_family_support=set(fs)==set(families),means=means)


def read_records(output,doc,seeds):
    result=[]
    for seed in seeds:
        for method in doc['methods']:
            for case in doc['cases']:
                directory=output/'runs'/case['case_id']/method/('seed-'+str(seed))
                if not (directory/'result.json').exists():continue
                r=json.loads((directory/'result.json').read_text())
                expected=dict(manifest_id=doc['manifest_id'],case_id=case['case_id'],method_id=method,seed=seed)
                if any(r[k]!=v for k,v in expected.items()):raise ValueError('Changed stored outcome identity')
                if r['status']=='complete':
                    verify_files(r['event_files'])
                    verify_files(r['fitted_artifacts'])
                    if len(r['event_files'])!=6:raise ValueError('Incomplete event coverage')
                    if digest(directory/'worker/full-scores.npy')!=r['full_scores_sha256']:raise ValueError('Changed scores')
                    for path in r['event_files']:verify_event(json.loads(Path(path).read_text()))
                result.append(r)
    return result


def aggregate(output,doc):
    records=read_records(output,doc,[37,38,39]);rows=[]
    for r in records:
        if r['status']!='complete':continue
        row={k:r[k] for k in ('method_id','case_id','source_family','seed')}
        events=[json.loads(Path(p).read_text()) for p in r['event_files']]
        row.update(summarize_events(events))
        rows.append(row)
    metrics=[axis+'_'+metric for axis in ('sensor','time') for metric in METRICS]+['explanation_seconds']
    comparisons=[paired_metric(rows,k,FOCAL,m,[37,38,39],FAMILIES) for k in metrics for m in doc['methods'] if m!=FOCAL]
    coverage={m:{f:{status:sum(r['method_id']==m and r['source_family']==f and r['status']==status for r in records)
                       for status in ('complete','timeout','failed','interrupted')} for f in FAMILIES} for m in doc['methods']}
    summary=dict(manifest_id=doc['manifest_id'],recorded=len(records),expected=195,
                 outcomes={s:sum(r['status']==s for r in records) for s in ('complete','timeout','failed','interrupted')},
                 coverage=coverage,per_case_seed=rows,paired=comparisons,
                 all_method_common=[common_metric(rows,k,doc['methods'],[37,38,39],FAMILIES) for k in metrics],
                 conditional_subsets='descriptive per-method only; selection may differ between detectors',
                 random_stability_qualification='shared rank permutation preserves correlation and is not an independent stability control',
                 manuscript_ready=False)
    path=output/('summary-'+identity(summary)[:12]+'.json')
    if not path.exists():write_new(path,summary)
    return summary


def controller(args):
    from research.quality_cost_extension.trials import hardware_lock
    output=args.output.resolve();manifest=output/'manifest.json'
    doc=json.loads(manifest.read_text()) if manifest.exists() else freeze(output)
    validate(doc)
    print(json.dumps(dict(manifest_id=doc['manifest_id'],planned=195,pilot=65,methods=doc['methods'],families=FAMILIES)),flush=True)
    if args.dry_run:return
    if args.aggregate:
        print(json.dumps(aggregate(output,doc)['outcomes']),flush=True);return
    with hardware_lock('controller'),controller_signals():
        with hardware_lock('worker'):pass
        for owner in output.glob('runs/*/*/seed-*/worker/worker-owner.json'):require_stopped(owner)
        for seed in (37,38,39):
            if seed!=37:
                records=read_records(output,doc,[37]);gate=technical_gate(records,doc['methods'],doc['cases'])
                if not gate['passed']:
                    write_new(output/'pilot-blocked.json',gate);print(json.dumps(gate),flush=True);return
                if not (output/'pilot-gate.json').exists():write_new(output/'pilot-gate.json',gate)
            for method in doc['methods']:
                for case in doc['cases']:
                    directory=output/'runs'/case['case_id']/method/('seed-'+str(seed))
                    directory.mkdir(parents=True,exist_ok=True)
                    expected=dict(manifest_id=doc['manifest_id'],method_id=method,case_id=case['case_id'],
                                  source_family=case['source_family'],seed=seed)
                    state=existing_state(directory)
                    if state and state!='interrupted':continue
                    if state=='interrupted':
                        if not (directory/'result.json').exists():write_new(directory/'result.json',recover_terminal(directory,expected))
                        continue
                    validate(doc);write_new(directory/'started.json',expected)
                    attempt=directory/'worker';attempt.mkdir()
                    env=environment();env.update(PYTHONHASHSEED=str(seed),MPLCONFIGDIR=str(attempt/'mpl'),XDG_CACHE_HOME=str(attempt/'cache'))
                    command=[str(executable(method)),'-m','research.explanation_study.run','--worker','--manifest',str(manifest),
                             '--case-id',case['case_id'],'--method',method,'--seed',str(seed),'--output',str(attempt),
                             '--controller-pid',str(os.getpid())]
                    try:
                        execute(command,attempt,doc['whole_worker_timeout'],env)
                        r=json.loads((attempt/'run.json').read_text())
                        if any(r[k]!=v for k,v in expected.items()):raise ValueError('Worker identity mismatch')
                    except TimeoutError as error:
                        r=dict(expected,status='timeout',phase='hard-process-limit',error=repr(error),technical_verified=False)
                    except KeyboardInterrupt:
                        write_new(directory/'result.json',dict(expected,status='interrupted',technical_verified=False))
                        raise
                    except Exception as error:
                        r=dict(expected,status='failed',error=repr(error),technical_verified=False)
                    write_new(directory/'result.json',r)
                    print(json.dumps({k:r.get(k) for k in ('method_id','case_id','seed','status','phase','error')}),flush=True)
            print(json.dumps(dict(seed=seed,outcomes=aggregate(output,doc)['outcomes'])),flush=True)
        summary=aggregate(output,doc);write_new(output/'completion.json',dict(manifest_id=doc['manifest_id'],recorded=summary['recorded'],outcomes=summary['outcomes']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/explanation_20261027/v1')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--manifest',type=Path)
    parser.add_argument('--method');parser.add_argument('--case-id');parser.add_argument('--seed',type=int)
    parser.add_argument('--controller-pid',type=int)
    args=parser.parse_args()
    worker(args) if args.worker else controller(args)


if __name__=='__main__':main()
