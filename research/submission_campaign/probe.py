"""Fresh-process seed replay checks using small synthetic inputs, not accuracy experiments."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from research.evaluation_protocols.campaign import ROOT,_json,digest
from .deep import configurations,validate_worker
from .studies import OUT


def main():
    root=OUT/'seed-replay';inputs=root/'input';inputs.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(910)
    train=rng.normal(size=(400,4));test=rng.normal(size=(400,4))
    np.save(inputs/'train.npy',train);np.save(inputs/'test.npy',test)
    np.save(inputs/'calibration.npy',rng.normal(size=(100,4)))
    hashes={n:digest(inputs/n) for n in ['train.npy','test.npy']}
    results=[]
    for method in ('tranad','catch','mtad_gat'):
        hashes={n:digest(inputs/n) for n in ['train.npy','test.npy']+(['calibration.npy'] if method=='catch' else [])}
        config={**configurations()[method],'epochs':1}
        config_path=root/(method+'-configuration.json');_json(config_path,config)
        vectors=[];errors=[]
        for index,seed in enumerate([37,37,38]):
            folder=root/method/str(index);folder.mkdir(parents=True,exist_ok=True)
            python=ROOT/'benchmarks/official_baselines/.envs'/method/'bin/python'
            command=[str(python),'-m','research.submission_campaign.deep','--worker','--method',method,
                     '--seed',str(seed),'--input-dir',str(inputs),'--worker-output',str(folder),
                     '--configuration',str(config_path)]
            env={**os.environ,'DYNSPHERE_DEVICE':'cpu','PYTHONHASHSEED':str(seed),'OMP_NUM_THREADS':'1',
                 'OPENBLAS_NUM_THREADS':'1','LOKY_MAX_CPU_COUNT':'1','DGLBACKEND':'pytorch',
                 'DGLDEFAULTDIR':str(folder/'dgl'),'MPLCONFIGDIR':str(folder/'mpl'),
                 'XDG_CACHE_HOME':str(folder/'cache')}
            try:
                with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
                    result=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,timeout=180)
                if result.returncode:raise RuntimeError(f'worker exit {result.returncode}')
                values,_=validate_worker(folder,method,seed,len(test),hashes)
                vectors.append(values)
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
        replay=(len(vectors)==3 and np.array_equal(vectors[0],vectors[1]))
        changed=(len(vectors)==3 and not np.array_equal(vectors[0],vectors[2]))
        results.append({'method':method,'same_seed_exact_replay':replay,
                        'different_seed_changes_scores':changed,'errors':errors})
        _json(root/'verification.json',{'results':results,'status':'running','synthetic_probe_only':True})
        print(json.dumps(results[-1]),flush=True)
    passed=all(r['same_seed_exact_replay'] and r['different_seed_changes_scores'] for r in results)
    _json(root/'verification.json',{'results':results,'status':'verified' if passed else 'failed',
                                  'synthetic_probe_only':True,'device':'cpu','epochs':1})
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
