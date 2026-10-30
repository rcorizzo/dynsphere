"""Explicit Linux replication around the unchanged scientific worker."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import signal
import threading
from types import SimpleNamespace

from research.explanation_study import run as original


def derive(parent, output, sources):
    doc=copy.deepcopy(parent)
    for key in ('manifest_id','assignments','production_output','parent_manifest_path'):
        doc.pop(key,None)
    doc.update(replication_parent_id=parent['manifest_id'],execution_host='independent-linux-cpu',
        production_output=str(Path(output).resolve()),mac_reference_records=copy.deepcopy(parent['original_records']),
        mac_score_equivalence_required=False,
        equivalence='within-Linux fitted-state roundtrip and RNG-replayed repeated scores; no Mac refit equality',
        protocol='controlled-context512-group-occlusion-v1-independent-linux-replication',
        representation_protocol='context256-mean-native-layers-train-only-pca-v1',
        representation_timeout=3600)
    doc['original_records']={k:dict(v,status='separate-host') for k,v in parent['original_records'].items()}
    doc['bindings'].update(sources)
    doc['trial_ids']=[c['case_id']+'|'+m+'|'+str(s) for s in doc['seeds']
                      for m in doc['methods'] for c in doc['cases']]
    doc['manifest_id']=original.identity(doc)
    return doc


def validate(doc):
    original.validate(doc)
    if doc.get('mac_score_equivalence_required') is not False:
        raise ValueError('Separate-host protocol required')


def select_trial(doc, stage, index):
    if stage not in ('pilot','remaining'):raise ValueError('Unknown stage')
    keys=[k for k in doc['trial_ids'] if (k.endswith('|37'))==(stage=='pilot')]
    if index<1 or index>len(keys):raise ValueError('Array index out of range')
    return keys[index-1]


def reserve(doc,key,root):
    if Path(root).resolve()!=Path(doc['production_output']).resolve():raise ValueError('Noncanonical output')
    if key not in doc['trial_ids']:raise ValueError('Unassigned trial')
    case,method,seed=key.split('|')
    if any(Path(p).name!=p or p in ('.','..') for p in (case,method,seed)):
        raise ValueError('Invalid trial path')
    directory=Path(root).resolve()/'runs'/case/method/('seed-'+seed)
    directory.mkdir(parents=True,exist_ok=False)
    return directory


def validate_gate(doc,gate):
    if not gate or gate.get('manifest_id')!=doc['manifest_id'] or not gate.get('passed'):
        raise ValueError('Passed independent Linux pilot gate required')
    if not gate.get('evidence'):raise ValueError('Missing gate evidence')
    original.verify_files(gate['evidence'])
    if 'trial_ids' in doc:
        records=[]
        for key in [k for k in doc['trial_ids'] if k.endswith('|37')]:
            case,method,seed=key.split('|')
            directory=Path(doc['production_output'])/'runs'/case/method/('seed-'+seed)
            path=directory/'result.json'
            if str(path) not in gate['evidence']:raise ValueError('Missing canonical pilot evidence')
            record,files=verify_record(directory,doc)
            if any(gate['evidence'].get(p)!=sha for p,sha in files.items()):
                raise ValueError('Incomplete pilot artifact binding')
            records.append(record)
        if not original.technical_gate(records,doc['methods'],doc['cases'])['passed']:
            raise ValueError('Pilot evidence fails technical gate')


def verify_record(directory,doc):
    import numpy as np
    record=json.loads((directory/'result.json').read_text())
    case,method,seed=directory.parent.parent.name,directory.parent.name,int(directory.name.split('-')[-1])
    if any(record.get(k)!=v for k,v in dict(manifest_id=doc['manifest_id'],case_id=case,
                                            method_id=method,seed=seed).items()):
        raise ValueError('Changed terminal identity')
    evidence={str(directory/'result.json'):original.digest(directory/'result.json')}
    if record['status']=='complete':
        if not record.get('technical_verified') or len(record.get('event_files',{}))!=6:
            raise ValueError('Incomplete verified events')
        case_doc=next(c for c in doc['cases'] if c['case_id']==case)
        expected_paths={str(directory/'worker'/('event-'+e['event_id']+'.json')):e for e in case_doc['events']}
        if set(record['event_files'])!=set(expected_paths):raise ValueError('Wrong canonical event coverage')
        required={'fit-evidence.json','scoring-rng.pkl'}
        if method!='dagmm':required.add('fitted-model.pkl')
        expected_artifacts={str(directory/'worker'/n) for n in required}
        if set(record['fitted_artifacts'])!=expected_artifacts:raise ValueError('Incomplete fitted state evidence')
        original.verify_files(record['event_files']);original.verify_files(record['fitted_artifacts'])
        fit=json.loads((directory/'worker/fit-evidence.json').read_text())
        if not fit.get('full_score_verified') or fit.get('original_score_comparison') is not False or fit['runtime']!=doc['runtimes'][method]:
            raise ValueError('Wrong within-host score/runtime evidence')
        if not fit.get('within_host_repeated_score_verified'):
            raise ValueError('Missing within-host repeated score proof')
        if method!='dagmm' and not fit.get('serialization_verified'):
            raise ValueError('Missing within-host restoration/repeated score proof')
        if method=='dagmm' and not fit.get('archived_checkpoint_reused'):raise ValueError('Missing archived checkpoint proof')
        if original.digest(directory/'worker/full-scores.npy')!=record['full_scores_sha256']:
            raise ValueError('Changed full scores')
        for path in record['event_files']:
            event=json.loads(Path(path).read_text());original.verify_event(event)
            verify_event_identity(expected_paths[path],method,event)
        evidence.update(record['event_files']);evidence.update(record['fitted_artifacts'])
        evidence[str(directory/'worker/full-scores.npy')]=record['full_scores_sha256']
    return record,evidence


def verify_event_identity(expected,method,event):
    import numpy as np
    if event.get('event_id')!=expected['event_id'] or event.get('input_sha256')!=expected['sha256']:
        raise ValueError('Changed event/input provenance')
    if method.startswith('dynsphere'):
        native=event.get('native',{})
        if not native.get('accounting_verified') or not native.get('propagation_verified'):
            raise ValueError('Unverified native accounting')
        if not np.isclose(native.get('target',np.nan),event['target'],rtol=1e-10,atol=1e-10):
            raise ValueError('Changed native target')
        if not np.isclose(sum(native.get('components',{}).values()),event['target'],rtol=1e-10,atol=1e-10):
            raise ValueError('Changed native score decomposition')
        if not 240<=native['target_row']<272 or abs(native['target_row']-native['origin_row'])>8:
            raise ValueError('Invalid propagation provenance')


def start_guard(output,seconds):
    if os.getpid()!=os.getpgrp():raise ValueError('Hard guard requires an owned worker process group')
    def expired():
        try:original.write_new(Path(output)/'hard-timeout.json',dict(status='timeout',phase='hard-process-limit'))
        finally:os.killpg(os.getpgrp(),signal.SIGKILL)
    guard=threading.Timer(seconds,expired);guard.daemon=True;guard.start()
    return guard


def repeated_scores(fitted,inputs,reference=None):
    import numpy as np
    values=np.load(Path(inputs)/'test.npy',allow_pickle=False)
    first=fitted.score(values) if reference is None else np.asarray(reference)
    second=fitted.score(values)
    if first.shape!=(len(values),) or second.shape!=first.shape:raise ValueError('Wrong full-score shape')
    if not np.isfinite(first[100:-100]).all():raise ValueError('Nonfinite full scores')
    np.testing.assert_allclose(second[100:-100],first[100:-100],rtol=1e-7,atol=1e-9)
    return first


def make_gate(doc,output):
    output=Path(output).resolve()
    records=[];evidence={}
    for key in [k for k in doc['trial_ids'] if k.endswith('|37')]:
        case,method,seed=key.split('|')
        directory=output/'runs'/case/method/('seed-'+seed)
        record,files=verify_record(directory,doc);records.append(record);evidence.update(files)
    checked=original.technical_gate(records,doc['methods'],doc['cases'])
    gate=dict(checked,manifest_id=doc['manifest_id'],evidence=evidence,
              scope='independent Linux replication only; representation metrics excluded from promotion')
    original.write_new(output/'pilot-gate.json',gate)
    return gate


def child(args,doc):
    import numpy as np
    from research.explanation_study import models
    from research.quality_cost_extension import trials
    owner=json.loads((args.output/'worker-owner.json').read_text())
    if owner!={'pid':os.getpid(),'controller_pid':os.getppid()}:raise ValueError('Unowned worker')
    dispatch=json.loads((args.output.parent/'dispatch.json').read_text())
    if dispatch['manifest_id']!=doc['manifest_id']:raise ValueError('Changed dispatch identity')
    case,method,seed=dispatch['trial'].split('|');seed=int(seed)
    if dispatch['trial'] not in doc['trial_ids']:raise ValueError('Unassigned worker')
    if seed!=37:validate_gate(doc,json.loads((Path(doc['production_output'])/'pilot-gate.json').read_text()))
    trials.LOCK_ROOT=args.output/'reservation-locks'
    captured={}
    def checked_fit(*values,**kwargs):
        fitted,full,evidence=fit(*values,**kwargs)
        repeated_scores(fitted,values[2],full)
        captured['fitted']=fitted
        return fitted,full,dict(evidence,within_host_repeated_score_verified=True)
    def checked_restore(*values,**kwargs):
        fitted=restore(*values,**kwargs)
        repeated_scores(fitted,next(c for c in doc['cases'] if c['case_id']==case)['inputs'])
        captured['fitted']=fitted;return fitted
    write=original.write_new
    def evidenced_write(path,value):
        if Path(path).name=='fit-evidence.json' and 'fitted' in captured:
            value=dict(value,within_host_repeated_score_verified=True)
        return write(path,value)
    fit,restore=models.fit_original,models.restore_dagmm
    # Independent wall guard retains the original explanation hard cap even
    # though the parent also allows a subsequent representation stage.
    guard=start_guard(args.output,doc['whole_worker_timeout'])
    try:
        with models.replace_attribute(models,'fit_original',checked_fit),models.replace_attribute(models,'restore_dagmm',checked_restore),models.replace_attribute(original,'write_new',evidenced_write):
            original.worker(SimpleNamespace(manifest=args.manifest,case_id=case,method=method,seed=seed,
                                            output=args.output,controller_pid=os.getppid()))
    finally:guard.cancel()
    # The original scientific worker is terminal before new analysis starts.
    record=json.loads((args.output/'run.json').read_text())
    representation=dict(status='unavailable',reason='No verified fitted state',method_id=method)
    if 'fitted' in captured and (args.output/'fit-evidence.json').exists():
        try:
            from .representations import study
            with original.deadline(doc['representation_timeout']):
                representation=study(captured['fitted'],method,next(c for c in doc['cases'] if c['case_id']==case),args.output)
        except TimeoutError as error:representation=dict(status='timeout',error=repr(error))
        except Exception as error:
            representation=dict(status='unavailable',error=repr(error))
            import traceback;traceback.print_exc()
    original.write_new(args.output/'representation.json',dict(representation,manifest_id=doc['manifest_id'],
        case_id=case,method_id=method,seed=seed,explanation_status=record['status']))


def trial(args,doc):
    key=select_trial(doc,args.stage,args.index)
    if not key.endswith('|37'):validate_gate(doc,json.loads((args.output/'pilot-gate.json').read_text()))
    directory=reserve(doc,key,args.output);worker=directory/'worker';worker.mkdir()
    case,method,seed=key.split('|');seed=int(seed)
    expected=dict(manifest_id=doc['manifest_id'],case_id=case,method_id=method,seed=seed,
        source_family=next(c['source_family'] for c in doc['cases'] if c['case_id']==case))
    original.write_new(directory/'dispatch.json',dict(trial=key,manifest_id=doc['manifest_id']))
    original.write_new(directory/'started.json',expected)
    env=original.environment();env.update(CUDA_VISIBLE_DEVICES='',DYNSPHERE_DEVICE='cpu',PYTHONHASHSEED=str(seed),
        MPLCONFIGDIR=str(worker/'mpl'),XDG_CACHE_HOME=str(worker/'cache'))
    command=[doc['runtimes'][method]['executable'],'-m','research.explanation_replication.run',
             '--manifest',str(args.manifest),'--output',str(worker),'--child']
    # Original fit/explanation deadlines remain3600/3600; diagnostic budget is separate.
    try:
        original.execute(command,worker,doc['whole_worker_timeout']+doc['representation_timeout']+30,env)
        result=json.loads((worker/'run.json').read_text())
        if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('Changed worker identity')
    except TimeoutError as error:
        if (worker/'run.json').exists():result=json.loads((worker/'run.json').read_text())
        else:result=dict(expected,status='timeout',technical_verified=False,error=repr(error),phase='hard-process-limit')
    except Exception as error:
        if (worker/'run.json').exists():result=json.loads((worker/'run.json').read_text())
        else:result=dict(expected,status='timeout' if (worker/'hard-timeout.json').exists() else 'failed',
                         technical_verified=False,error=repr(error),phase='hard-process-limit' if (worker/'hard-timeout.json').exists() else 'worker')
    result.update(execution_host=platform.node(),execution_profile='independent-linux-cpu',
                  mac_refit_equivalence_claimed=False,replication_parent_id=doc['replication_parent_id'])
    original.write_new(directory/'result.json',result)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare',type=Path)
    p.add_argument('--child',action='store_true');p.add_argument('--gate',action='store_true')
    p.add_argument('--stage',choices=['pilot','remaining']);p.add_argument('--index',type=int)
    a=p.parse_args();a.output=a.output.resolve();a.manifest=a.manifest.resolve()
    if a.prepare:
        from .continuation import validate_continuation
        parent=json.loads(a.prepare.read_text());validate_continuation(parent)
        sources={str(path):original.digest(path) for path in Path(__file__).parent.glob('*.py')}
        sources[str(a.prepare.resolve())]=original.digest(a.prepare)
        doc=derive(parent,a.output,sources);validate(doc)
        original.write_new(a.manifest,doc);print(doc['manifest_id']);return
    doc=json.loads(a.manifest.read_text());validate(doc)
    if a.child:child(a,doc)
    elif a.gate:
        gate=make_gate(doc,a.output);print(json.dumps({k:v for k,v in gate.items() if k!='evidence'}))
        if not gate['passed']:raise SystemExit(1)
    elif a.stage and a.index:trial(a,doc)
    else:p.error('Choose prepare, child, gate or stage/index')


if __name__=='__main__':main()
