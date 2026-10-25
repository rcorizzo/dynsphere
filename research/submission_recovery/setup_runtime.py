"""Install two binary dependencies privately, without modifying shared environments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

from .run import ROOT,OVERLAY,RUNTIME,PINS,METHODS,_identity,_json,overlay_files,worker_environment


def install_command(uv,python,overlay):
    return [str(uv),'pip','install','--python',str(python),'--target',str(overlay),
            '--only-binary',':all:','--no-deps','--link-mode','copy',
            *[f'{name}=={version}' for name,version in PINS.items()]]


def inspect_runtime(method,overlay,imports=False):
    code='''import importlib.metadata as m,json,sys
names=('numpy','pandas','pytz','scipy','scikit-learn','tensorflow','torch','python-dateutil')
result={'python':sys.version.split()[0]}
for name in names:
    try:result[name]=m.version(name)
    except m.PackageNotFoundError:result[name]=None
print(json.dumps(result))
'''
    if imports:
        code+='import numpy,pandas,pytz,sklearn,tensorflow\nprint(json.dumps({"pandas":pandas.__file__,"pytz":pytz.__file__}))\n'
    executable=ROOT/'benchmarks/official_baselines/.envs'/method/'bin/python'
    result=subprocess.check_output([str(executable),'-c',code],cwd=ROOT,
            env=worker_environment(method,37,overlay),text=True)
    return [json.loads(line) for line in result.splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay',type=Path,default=OVERLAY)
    parser.add_argument('--runtime',type=Path,default=RUNTIME)
    args=parser.parse_args()
    if args.runtime.exists():
        record=json.loads(args.runtime.read_text())
        if _identity(overlay_files(args.overlay))!=record['overlay_sha256']:
            raise ValueError('Existing private dependency evidence differs')
        print(json.dumps({'status':'already-prepared','runtime':str(args.runtime)}));return
    if args.overlay.exists() and any(args.overlay.iterdir()):
        raise ValueError('Refusing to overwrite an existing private dependency directory')
    # An empty target is excluded from the baseline version inspection.
    before=inspect_runtime('dagmm',args.overlay)[0]
    args.overlay.parent.mkdir(parents=True,exist_ok=True)
    command=install_command(ROOT/'.venv/bin/uv',ROOT/'benchmarks/official_baselines/.envs/dagmm/bin/python',args.overlay)
    subprocess.run(command,check=True,cwd=ROOT)
    after,origins=inspect_runtime('dagmm',args.overlay,imports=True)
    for package in ('numpy','tensorflow','scikit-learn','scipy','python-dateutil'):
        if after[package]!=before[package]:raise ValueError(f'Shared dependency version changed: {package}')
    for name,version in PINS.items():
        if after[name]!=version or not Path(origins[name]).resolve().is_relative_to(args.overlay.resolve()):
            raise ValueError('Pinned dependency was not imported from private overlay')
    runtime={'pins':PINS,'overlay_sha256':_identity(overlay_files(args.overlay)),
             'versions':{'dagmm':after,'omni_anomaly_tsb_ad':inspect_runtime('omni_anomaly_tsb_ad',args.overlay)[0]}}
    _json(args.overlay.parent/'installation-evidence.json',{'command':command,'before':before,
          'after':after,'import_origins':origins,'overlay_files':overlay_files(args.overlay)})
    _json(args.runtime,runtime)
    print(json.dumps({'status':'prepared','runtime':str(args.runtime),'evidence':runtime},indent=2))


if __name__=='__main__':main()
