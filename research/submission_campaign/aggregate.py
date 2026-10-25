"""Compare completed deep outputs with frozen local scores on identical observed rows."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from .studies import BASE,BASE_ID,OUT,load_case
from .deep import EXTERNAL,common_mask,validate_worker,frozen_input_hashes
from research.evaluation_protocols.campaign import METHODS,_json,_identity,digest


def validate_deep_record(record,path,doc,entry,case):
    method=record['method_id'];seed=record['seed']
    identity=_identity({'manifest':doc['manifest_id'],'case':entry['case_id'],'seed':seed,'method':method})
    if (record.get('manifest_id')!=doc['manifest_id'] or record.get('run_identity')!=identity
            or record.get('case_id')!=entry['case_id'] or record.get('source_family')!=entry['source_family']
            or method not in doc['methods'] or seed not in doc['seeds']):
        raise ValueError('Deep parent identity differs from frozen manifest')
    hashes=frozen_input_hashes(case,method)
    if record.get('input_sha256')!=hashes:
        raise ValueError('Deep input hashes differ from independently reconstructed frozen inputs')
    scores,worker=validate_worker(Path(path)/'worker',method,seed,len(case['labels']),hashes)
    if worker.get('configuration')!=doc['configurations'][method]:
        raise ValueError('Deep worker configuration differs from frozen manifest')
    if worker.get('upstream_source')!=doc['upstream_sources'][method]:
        raise ValueError('Deep upstream content differs from frozen manifest')
    if record.get('scores_sha256')!=worker['scores_sha256']:
        raise ValueError('Deep parent score hash differs from worker')
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep-root',type=Path,required=True)
    parser.add_argument('--methods',nargs='+',choices=EXTERNAL,default=list(EXTERNAL))
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=args.output or args.deep_root/'aggregate';output.mkdir(parents=True,exist_ok=True)
    base=json.loads((BASE/'manifests'/(BASE_ID+'.json')).read_text())
    manifests=list((args.deep_root.parent.parent/'deep-manifests').glob(args.deep_root.name+'*.json'))
    if args.manifest:manifests=[args.manifest]
    if len(manifests)!=1:raise ValueError('Exactly one frozen deep manifest is required')
    doc=json.loads(manifests[0].read_text())
    if (_identity({k:v for k,v in doc.items() if k!='manifest_id'})!=doc['manifest_id']
            or not doc['manifest_id'].startswith(args.deep_root.name)
            or doc['base_manifest']!=base or 'upstream_sources' not in doc):
        raise ValueError('Frozen deep manifest identity or base inputs differ')
    entries={e['case_id']:e for e in base['cases'] if e['source_family']!='PowerSystem'}
    local=[json.loads(p.read_text()) for p in (BASE/'runs').glob('*/run.json')]
    local=[r for r in local if r['manifest_id']==BASE_ID and r['source_family']!='PowerSystem' and r['status']=='complete']
    deep=[json.loads(p.read_text()) for p in (args.deep_root/'runs').glob('*/run.json')]
    deep=[r for r in deep if r['method_id'] in args.methods]
    rows=[];failures=[r for r in deep if r['status']!='complete'];cache={}
    for record in local+[r for r in deep if r['status']=='complete']:
        name=record['case_id']
        if name not in cache:cache={name:load_case(entries[name])}
        case=cache[name];mask=common_mask(len(case['labels']))
        path=Path(record['run_directory'])
        if record['method_id'] in METHODS:
            if digest(path/'scores.npz')!=record['artifact_sha256']:raise ValueError('Local score integrity mismatch')
            with np.load(path/'scores.npz',allow_pickle=False) as arrays:
                if not np.array_equal(arrays['labels'],case['labels']):raise ValueError('Local label alignment differs')
                scores=arrays['scores'].copy()
        else:
            if not path.is_dir():path=args.deep_root/'runs'/record['run_identity']
            scores=validate_deep_record(record,path,doc,entries[name],case)
        labels=case['labels'][mask];scores=scores[mask]
        rows.append({'case_id':name,'source_family':record['source_family'],'method_id':record['method_id'],
                     'seed':record['seed'],'run_identity':record['run_identity'],
                     'auprc':float(average_precision_score(labels,scores)),
                     'auroc':float(roc_auc_score(labels,scores)),
                     'pauc_fpr_05':float(roc_auc_score(labels,scores,max_fpr=.05))})
    frame=pd.DataFrame(rows)
    if frame.duplicated(['case_id','seed','method_id']).any():raise ValueError('Duplicate method/case/seed records')
    frame.to_csv(output/'per_run_available.csv',index=False)
    expected=len(METHODS)+len(args.methods)
    counts=frame.groupby(['case_id','seed']).method_id.nunique()
    support=counts[counts==expected].reset_index()[['case_id','seed']]
    matched=frame.merge(support,on=['case_id','seed'],validate='many_to_one')
    matched.to_csv(output/'per_run_matched.csv',index=False)
    metrics=['auprc','auroc','pauc_fpr_05']
    case=matched.groupby(['source_family','case_id','method_id'],as_index=False)[metrics].mean()
    case.to_csv(output/'per_case_matched.csv',index=False)
    case.groupby(['source_family','method_id'],as_index=False)[metrics].mean().to_csv(output/'per_family_matched.csv',index=False)
    _json(output/'completion.json',{'deep_methods':args.methods,'deep_records_observed':len(deep),
          'deep_complete':sum(r['status']=='complete' for r in deep),'noncomplete':failures,
          'matched_case_seed_pairs':len(support),'matched_cases':case.case_id.nunique(),
          'support':'first-and-last-100-observations-excluded','manuscript_ready':False,
          'scope':'partial unless all declared source/method/seed executions have terminal records'})


if __name__=='__main__':main()
