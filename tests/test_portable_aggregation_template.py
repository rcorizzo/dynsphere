"""Aggregation templates consume user paths instead of a developer's host."""
import os
from pathlib import Path
import subprocess


def test_cpu_aggregation_template_forwards_explicit_artifact_paths(tmp_path):
    python=tmp_path/'.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    names={
        'DYNSPHERE_ORIGINAL_ROOT':'original-runs',
        'DYNSPHERE_ORIGINAL_MANIFEST':'original.json',
        'DYNSPHERE_RECOVERY_ROOT':'recovery-runs',
        'DYNSPHERE_RECOVERY_MANIFEST':'recovery.json',
        'DYNSPHERE_AGGREGATION_OUTPUT':'summary',
    }
    env=dict(os.environ,DYNSPHERE_AGGREGATION_REPO=str(tmp_path))
    env.update({key:str(tmp_path/value) for key,value in names.items()})
    script=Path(__file__).resolve().parents[1]/'research/submission_recovery/au_l40_cpu_aggregate.bsub'
    result=subprocess.run(['bash','-c','module() { :; }; export -f module; exec bash "$1"',
        'test',str(script)],env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    args=result.stdout.splitlines()
    for flag,key in [('--original-root','DYNSPHERE_ORIGINAL_ROOT'),
            ('--original-manifest','DYNSPHERE_ORIGINAL_MANIFEST'),
            ('--recovery-root','DYNSPHERE_RECOVERY_ROOT'),
            ('--recovery-manifest','DYNSPHERE_RECOVERY_MANIFEST'),
            ('--output','DYNSPHERE_AGGREGATION_OUTPUT')]:
        assert args[args.index(flag)+1]==env[key]
