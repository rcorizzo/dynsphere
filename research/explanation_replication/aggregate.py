"""Host-qualified verification and complete-outcome exploratory summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.explanation_study import run as original
from .run import validate,verify_record
from .diagnostics import budget_curve,cross_seed
from .representations import compactness,input_probe


def same_metrics(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same_metrics(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same_metrics(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and bool(np.isclose(a,b,rtol=1e-9,atol=1e-11))
    return a==b


def aggregate(output,doc):
    output=Path(output).resolve();records=[];representations=[];curves=[];seed_groups={}
    for key in doc['trial_ids']:
        case,method,seed=key.split('|');directory=output/'runs'/case/method/('seed-'+seed)
        if not (directory/'result.json').exists():raise ValueError('Missing terminal outcome: '+key)
        record,_=verify_record(directory,doc);records.append(record)
        rep_path=directory/'worker/representation.json'
        if rep_path.exists():
            rep=json.loads(rep_path.read_text())
            if any(rep.get(k)!=v for k,v in dict(manifest_id=doc['manifest_id'],case_id=case,method_id=method,seed=int(seed)).items()):
                raise ValueError('Changed representation identity')
            if rep['status']=='complete':
                original.verify_files({rep['representation_path']:rep['representation_sha256']})
                with np.load(rep['representation_path'],allow_pickle=False) as data:
                    metrics=compactness(data['train'],data['test'])
                    probe=input_probe(data['train'],data['train_raw_mean'],data['test'],data['test_raw_mean'])
                if not same_metrics(metrics,rep['metrics']):raise ValueError('Changed compactness metric')
                if (probe is None)!=(rep['input_mean_probe_r2'] is None) or (probe is not None and not np.isclose(probe,rep['input_mean_probe_r2'])):
                    raise ValueError('Changed reconstruction probe')
            representations.append(rep)
        else:representations.append(dict(case_id=case,method_id=method,seed=int(seed),status='unavailable',reason='No terminal diagnostic record'))
        if record['status']!='complete':continue
        for path in record['event_files']:
            event=json.loads(Path(path).read_text())
            for axis,part in event['resolutions'].items():
                curves.append(dict(case_id=case,method_id=method,seed=int(seed),event_id=event['event_id'],
                                   resolution=axis,infidelity_by_budget=budget_curve(part)))
                seed_groups.setdefault((case,method,axis),{}).setdefault(int(seed),{})[event['event_id']]=part['attributions']
    primary=original.aggregate(output,doc)
    agreement=[dict(case_id=k[0],method_id=k[1],resolution=k[2],agreement=cross_seed(v),
                    completed_seeds=sorted(v)) for k,v in seed_groups.items()]
    summary=dict(manifest_id=doc['manifest_id'],recorded=len(records),primary_outcomes=primary['outcomes'],
        primary_summary_identity=original.identity(primary),representations=representations,
        attribution_budget_curves=curves,cross_seed_attribution_agreement=agreement,
        host_pooling=False,manuscript_ready=False,
        qualification='Exploratory compactness/input retention and saved-coalition diagnostics; no physical-root-cause or human-usability claim')
    path=output/('extended-summary-'+original.identity(summary)[:12]+'.json')
    if not path.exists():original.write_new(path,summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();doc=json.loads((a.output/'manifest.json').read_text());validate(doc)
    summary=aggregate(a.output,doc)
    print(json.dumps(dict(recorded=summary['recorded'],outcomes=summary['primary_outcomes'])))


if __name__=='__main__':main()
