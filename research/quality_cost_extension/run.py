"""Read-only extension of the frozen nine-method matched CPU study."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

from research.quality_cost.run import (common_support,cpu_environment,digest,summarize,
                                      validate_manifest as validate_original,validate_worker)
from research.evaluation_protocols.campaign import ROOT,_identity
from research.submission_campaign.deep import configurations,declared_revisions,upstream_snapshot
from .trials import hardware_lock

ORIGINAL=ROOT/'results/quality_cost_20261026/five-family-cpu-v1'
METHODS=('dagmm','tranad','omni_anomaly_tsb_ad','anomaly_transformer')


def write_new(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True)
        stream.write('\n')


def runtime(method):
    python=ROOT/'benchmarks/official_baselines/.envs'/method/'bin/python'
    command=[str(python),'-c','import json; from research.quality_cost_extension.trials import versions; print(json.dumps(versions()))']
    return json.loads(subprocess.check_output(command,cwd=ROOT,env=cpu_environment(),text=True))


def new_manifest():
    old=json.loads((ORIGINAL/'manifest.json').read_text())
    validate_original(old)
    cases=old['cases']
    for case in cases:
        for name,sha in case['input_sha256'].items():
            if digest(ORIGINAL/'inputs'/case['case_id']/name)!=sha:
                raise ValueError('Original input changed')
    configs=configurations()
    configs['omni_anomaly_tsb_ad']['validation_partition']='reserved-normal-calibration-v1'
    sources=dict(old['sources'])
    sources.update({str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.glob('*.py')})
    upstream={m:upstream_snapshot(ROOT/'benchmarks/official_baselines/.external'/m,declared_revisions()[m]) for m in METHODS}
    doc=dict(original_root=str(ORIGINAL),original_manifest_id=old['manifest_id'],
             original_manifest_sha256=digest(ORIGINAL/'manifest.json'),cases=cases,
             methods=METHODS,seeds=[37,38,39],configurations={m:configs[m] for m in METHODS},
             sources=sources,upstream=upstream,runtimes={m:runtime(m) for m in METHODS},
             hardware=old['hardware'],environment=cpu_environment({}),worker_timeout_seconds=3600.,
             timing=old['timing'],memory=old['memory'],support=old['support'],
             transformer_preparation='unchanged adapter order; full test windows materialized during fitting',
             serialization='pickle4 fitted scoring state; DAGMM native TF checkpoint, format disclosed separately',
             manuscript_ready=False)
    doc['manifest_id']=_identity(doc)
    return doc


def input_directory(doc,case):
    return Path(doc['original_root'])/'inputs'/case['case_id']


def jobs(doc):
    return ((case,method,seed) for method in doc['methods'] for case in doc['cases'] for seed in doc['seeds'])


def validate(doc):
    if _identity({k:v for k,v in doc.items() if k!='manifest_id'})!=doc['manifest_id']:
        raise ValueError('Changed manifest')
    if digest(Path(doc['original_root'])/'manifest.json')!=doc['original_manifest_sha256']:
        raise ValueError('Changed original study')
    for name,sha in doc['sources'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('Changed frozen source: '+name)
    for case in doc['cases']:
        for name,sha in case['input_sha256'].items():
            if digest(input_directory(doc,case)/name)!=sha:
                raise ValueError('Changed original timing input')
    for method,evidence in doc['upstream'].items():
        if upstream_snapshot(ROOT/'benchmarks/official_baselines/.external'/method,evidence['revision'])!=evidence:
            raise ValueError('Changed pinned upstream')
        if runtime(method)!=doc['runtimes'][method]:
            raise ValueError('Changed pinned local runtime')


def execute(command,directory,timeout,environment):
    directory=Path(directory)
    with (directory/'stdout.log').open('x') as stdout,(directory/'stderr.log').open('x') as stderr:
        process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,**environment},stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            code=process.wait(timeout=timeout)
        except BaseException as error:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if isinstance(error,subprocess.TimeoutExpired):
                raise TimeoutError('Whole-worker 3600-second budget exceeded') from error
            raise
    if code:
        raise RuntimeError('Worker exit code '+str(code))


def recover_attempt(directory,expected,rows):
    attempts=sorted(Path(directory).glob('attempt-*'))
    complete=[p for p in attempts if (p/'run.json').is_file()]
    if len(complete)>1:
        raise ValueError('Multiple completed attempts require explicit reconciliation')
    if complete:
        validate_worker(complete[0],expected,rows)
        return complete[0],True
    if attempts:
        raise InterruptedError('Prior unfinished attempt retained; no automatic retry')
    return Path(directory)/'attempt-0001',False


def aggregate(output,doc):
    records=[]
    for case,method,seed in jobs(doc):
        directory=output/'runs'/case['case_id']/method/('seed-'+str(seed))
        path=directory/'result.json'
        if not path.exists():
            continue
        record=json.loads(path.read_text())
        expected=dict(case_id=case['case_id'],method_id=method,seed=seed,manifest_id=doc['manifest_id'],configuration=doc['configurations'][method])
        if any(record.get(k)!=v for k,v in expected.items()):
            raise ValueError('Changed parent record')
        if record['status']=='complete':
            _,scores=validate_worker(directory/record['worker_subdirectory'],expected,case['test_rows'])
            if record['runtime']!=doc['runtimes'][method] or record['device']!='cpu' or record['threads']!=1:
                raise ValueError('Changed recorded runtime/device/thread evidence')
            labels=np.load(input_directory(doc,case)/'labels.npy',allow_pickle=False)
            mask=common_support(scores)
            actual=[roc_auc_score(labels[mask],scores[mask],max_fpr=.05),average_precision_score(labels[mask],scores[mask]),roc_auc_score(labels[mask],scores[mask])]
            np.testing.assert_allclose(actual,[record[k] for k in ('pauc_fpr_05','auprc','auroc')],rtol=1e-11,atol=1e-11)
        records.append(record)
    pd.DataFrame(records).to_csv(output/'per_run.csv',index=False)
    summarize(records).to_csv(output/'per_case.csv',index=False)
    result=dict(expected=60,observed=len(records),complete=sum(r['status']=='complete' for r in records),
                outcomes=pd.Series([r['status'] for r in records]).value_counts().to_dict(),
                missing=60-len(records),manifest_id=doc['manifest_id'],manuscript_ready=False)
    if len(records)==60 and not (output/'completion.json').exists():
        write_new(output/'completion.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--aggregate',action='store_true')
    parser.add_argument('--pilot',choices=METHODS)
    args=parser.parse_args()
    output=args.output.resolve()
    if args.resume or args.aggregate or args.pilot:
        doc=json.loads((output/'manifest.json').read_text())
        validate(doc)
    else:
        doc=new_manifest()
        output.mkdir(parents=True,exist_ok=False)
        write_new(output/'manifest.json',doc)
    print(json.dumps(dict(manifest_id=doc['manifest_id'],planned_runs=60)),flush=True)
    if args.dry_run:
        return
    with hardware_lock('controller'):
        # Check for an orphaned worker before recovering or launching anything.
        with hardware_lock('worker'):
            pass
        if args.aggregate:
            print(json.dumps(aggregate(output,doc)),flush=True)
            return
        selected=[(doc['cases'][0],args.pilot,37)] if args.pilot else jobs(doc)
        for case,method,seed in selected:
            directory=output/('pilots' if args.pilot else 'runs')/case['case_id']/method/('seed-'+str(seed))
            directory.mkdir(parents=True,exist_ok=True)
            expected=dict(case_id=case['case_id'],method_id=method,seed=seed,manifest_id=doc['manifest_id'],configuration=doc['configurations'][method])
            if (directory/'result.json').exists():
                record=json.loads((directory/'result.json').read_text())
                if any(record.get(k)!=v for k,v in expected.items()):
                    raise ValueError('Changed stored outcome')
                continue
            record=dict(expected,source_family=case['source_family'],status='failed',pilot=bool(args.pilot))
            env=cpu_environment()
            env.update(PYTHONHASHSEED=str(seed),TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',
                       DGLBACKEND='pytorch')
            try:
                attempt,recovered=recover_attempt(directory,expected,410 if args.pilot else case['test_rows'])
                record['worker_subdirectory']=attempt.name
                if not recovered:
                    attempt.mkdir()
                    python=ROOT/'benchmarks/official_baselines/.envs'/method/'bin/python'
                    command=[str(python),'-m','research.quality_cost_extension.trials','--manifest',str(output/'manifest.json'),
                             '--case-id',case['case_id'],'--method',method,'--seed',str(seed),'--output',str(attempt)]
                    if args.pilot:
                        command.append('--pilot')
                    env.update(MPLCONFIGDIR=str(attempt/'mpl'),XDG_CACHE_HOME=str(attempt/'cache'))
                    execute(command,attempt,doc['worker_timeout_seconds'],env)
                measurement,scores=validate_worker(attempt,expected,410 if args.pilot else case['test_rows'])
                record.update(measurement)
                if not args.pilot:
                    labels=np.load(input_directory(doc,case)/'labels.npy',allow_pickle=False)
                    mask=common_support(scores)
                    record.update(pauc_fpr_05=float(roc_auc_score(labels[mask],scores[mask],max_fpr=.05)),
                                  auprc=float(average_precision_score(labels[mask],scores[mask])),
                                  auroc=float(roc_auc_score(labels[mask],scores[mask])),evaluated_rows=int(mask.sum()))
            except TimeoutError as error:
                record.update(status='timeout',error=repr(error),timeout_seconds=doc['worker_timeout_seconds'])
            except InterruptedError as error:
                record.update(status='interrupted',error=repr(error))
            except Exception as error:
                record.update(status='failed',error=repr(error))
            write_new(directory/'result.json',record)
            print(json.dumps({k:record.get(k) for k in ('method_id','case_id','seed','status','fit_seconds','score_seconds','error')}),flush=True)
        if not args.pilot:
            print(json.dumps(aggregate(output,doc)),flush=True)


if __name__=='__main__':
    main()
