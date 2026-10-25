"""Explicitly combine original GPU and corrected CPU results on frozen support."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from research.evaluation_protocols.campaign import METHODS as LOCAL_METHODS,_identity,_json,digest
from research.submission_campaign.aggregate import validate_deep_record
from research.submission_campaign.studies import BASE,BASE_ID,load_case
from .run import METHODS,deep,validate_record

GPU_METHODS=tuple(m for m in deep.EXTERNAL if m not in METHODS)


def select_records(original,repaired):
    if any(r['method_id'] not in METHODS for r in repaired):
        raise ValueError('Unexpected method in recovery campaign')
    rows=[(r,'original') for r in original if r['method_id'] in GPU_METHODS]
    rows += [(r,'recovery') for r in repaired]
    keys=[(r['case_id'],r['seed'],r['method_id']) for r,_ in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate method/case/seed records')
    return rows


def validate_manifests(original,repaired):
    for doc in (original,repaired):
        if _identity({k:v for k,v in doc.items() if k!='manifest_id'})!=doc['manifest_id']:
            raise ValueError('Manifest content digest differs')
    if (repaired['recovered_from_manifest']!=original['manifest_id']
            or repaired['base_manifest']!=original['base_manifest']
            or repaired['seeds']!=original['seeds'] or repaired['methods']!=list(METHODS)):
        raise ValueError('Recovery and original manifest inputs differ')
    for method in METHODS:
        expected=copy.deepcopy(original['configurations'][method])
        if method=='omni_anomaly_tsb_ad':expected['validation_partition']='reserved-normal-calibration-v1'
        if expected!=repaired['configurations'][method]:
            raise ValueError('Unapproved recovery configuration change')
        if original['upstream_sources'][method]!=repaired['upstream_sources'][method]:
            raise ValueError('Upstream source differs across campaigns')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',required=True,type=Path)
    parser.add_argument('--original-manifest',required=True,type=Path)
    parser.add_argument('--recovery-root',required=True,type=Path)
    parser.add_argument('--recovery-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    original=json.loads(args.original_manifest.read_text());repaired=json.loads(args.recovery_manifest.read_text())
    validate_manifests(original,repaired)
    base=json.loads((BASE/'manifests'/(BASE_ID+'.json')).read_text())
    if original['base_manifest']!=base:raise ValueError('Frozen base manifest differs')
    entries={e['case_id']:e for e in base['cases'] if e['source_family']!='PowerSystem'}
    records=select_records([json.loads(p.read_text()) for p in (args.original_root/'runs').glob('*/run.json')],
                           [json.loads(p.read_text()) for p in (args.recovery_root/'runs').glob('*/run.json')])
    local=[]
    for path in (BASE/'runs').glob('*/run.json'):
        record=json.loads(path.read_text())
        if record.get('manifest_id')==BASE_ID and record.get('case_id') in entries and record['status']=='complete':
            local.append((record,path.parent))
    rows=[];noncomplete=[]
    for name,entry in entries.items():
        selected=[(r,origin) for r,origin in records if r['case_id']==name]
        noncomplete += [r for r,_ in selected if r['status']!='complete']
        completed=[(r,o) for r,o in selected if r['status']=='complete']
        local_case=[(r,p) for r,p in local if r['case_id']==name]
        if not completed and not local_case:continue
        case=load_case(entry);mask=deep.common_mask(len(case['labels']));labels=case['labels'][mask]
        artifacts=[]
        for record,path in local_case:
            method=record['method_id']
            if method not in LOCAL_METHODS:raise ValueError('Unexpected local method')
            identity=_identity({'manifest':BASE_ID,'case':name,'seed':record['seed'],'method':method})
            if record['run_identity']!=identity or digest(path/'scores.npz')!=record['artifact_sha256']:
                raise ValueError('Local score identity or artifact differs')
            with np.load(path/'scores.npz',allow_pickle=False) as arrays:
                if not np.array_equal(arrays['labels'],case['labels']):raise ValueError('Local label alignment differs')
                artifacts.append((record,arrays['scores'].copy(),'local'))
        for record,origin in completed:
            doc,root=(original,args.original_root) if origin=='original' else (repaired,args.recovery_root)
            path=root/'runs'/record['run_identity']
            validator=validate_deep_record if origin=='original' else validate_record
            artifacts.append((record,validator(record,path,doc,entry,case),origin))
        for record,scores,origin in artifacts:
            if scores.shape!=case['labels'].shape or not np.isfinite(scores).all():
                raise ValueError('Invalid score array')
            scores=scores[mask]
            rows.append({k:record[k] for k in ('case_id','source_family','method_id','seed','run_identity')} |
                        {'campaign':origin,'auprc':float(average_precision_score(labels,scores)),
                         'auroc':float(roc_auc_score(labels,scores)),
                         'pauc_fpr_05':float(roc_auc_score(labels,scores,max_fpr=.05))})
    columns=['case_id','source_family','method_id','seed','run_identity','campaign','auprc','auroc','pauc_fpr_05']
    frame=pd.DataFrame(rows,columns=columns)
    if frame.duplicated(['case_id','seed','method_id']).any():raise ValueError('Duplicate aggregate records')
    expected=len(LOCAL_METHODS)+len(deep.EXTERNAL)
    counts=frame.groupby(['case_id','seed']).method_id.nunique()
    support=counts[counts==expected].reset_index()[['case_id','seed']]
    matched=frame.merge(support,on=['case_id','seed'],validate='many_to_one')
    args.output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(args.output/'per_run_available.csv',index=False)
    matched.to_csv(args.output/'per_run_matched.csv',index=False)
    metrics=['auprc','auroc','pauc_fpr_05']
    cases=matched.groupby(['source_family','case_id','method_id'],as_index=False)[metrics].mean()
    cases.to_csv(args.output/'per_case_matched.csv',index=False)
    cases.groupby(['source_family','method_id'],as_index=False)[metrics].mean().to_csv(args.output/'per_family_matched.csv',index=False)
    _json(args.output/'completion.json',{'original_manifest':original['manifest_id'],
          'recovery_manifest':repaired['manifest_id'],'noncomplete':noncomplete,
          'matched_case_seed_pairs':len(support),'matched_cases':cases.case_id.nunique(),
          'planned_deep_records':len(entries)*len(deep.EXTERNAL)*len(original['seeds']),
          'observed_deep_records':len(records),'complete_deep_records':sum(r['status']=='complete' for r,_ in records),
          'support':original['support'],'manuscript_ready':False,
          'selection':{'original':GPU_METHODS,'recovery':METHODS},
          'scope':'Partial unless all declared executions have terminal records; incompatibilities remain explicit.'})


if __name__=='__main__':main()
