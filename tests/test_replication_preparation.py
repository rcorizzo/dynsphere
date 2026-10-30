"""The published Linux preparation entry point must need no private package."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

import pytest


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def continuation(tmp_path):
    bound=tmp_path/'input.npy';bound.write_bytes(b'controlled test input')
    parent=dict(methods=['dynsphere_euclidean_one_regime'],
        cases=[dict(case_id='case',source_family='BATADAL')],seeds=[37,38,39],
        original_records={},bindings={str(bound):hashlib.sha256(bound.read_bytes()).hexdigest()},
        runtimes={},configurations={'dynsphere_euclidean_one_regime':{'q':5}})
    parent['manifest_id']=identity(parent)
    path=tmp_path/'parent.json';path.write_text(json.dumps(parent))
    child={k:v for k,v in parent.items() if k not in ('manifest_id','runtimes')}
    child.update(parent_manifest_id=parent['manifest_id'],parent_manifest_path=str(path),runtimes={},
                 path_map={'parent_root':str(tmp_path),'execution_root':str(tmp_path)},assignments={})
    child['manifest_id']=identity(child)
    return child,bound


def test_prepare_linux_replication_without_unpublished_handoff_package(tmp_path):
    doc,_=continuation(tmp_path)
    source=tmp_path/'continuation.json';source.write_text(json.dumps(doc))
    output=tmp_path/'linux';manifest=output/'manifest.json'
    result=subprocess.run([sys.executable,'-m','research.explanation_replication.run',
        '--manifest',str(manifest),'--output',str(output),'--prepare',str(source)],
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1])),
        text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stderr
    prepared=json.loads(manifest.read_text())
    assert prepared['mac_score_equivalence_required'] is False
    assert prepared['trial_ids']==['case|dynsphere_euclidean_one_regime|37',
        'case|dynsphere_euclidean_one_regime|38','case|dynsphere_euclidean_one_regime|39']
    assert not (output/'runs').exists()
    assert prepared['configurations']==doc['configurations']


def test_continuation_rejects_changed_bound_input(tmp_path):
    from research.explanation_replication.continuation import validate_continuation
    doc,bound=continuation(tmp_path)
    validate_continuation(doc)
    bound.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed binding'):
        validate_continuation(doc)


def test_continuation_rejects_rehashed_scientific_changes(tmp_path):
    from research.explanation_replication.continuation import validate_continuation
    doc,_=continuation(tmp_path)
    doc['seeds']=[37]
    doc['manifest_id']=identity({k:v for k,v in doc.items() if k!='manifest_id'})
    with pytest.raises(ValueError,match='Changed inherited scientific field'):
        validate_continuation(doc)


def test_relocation_rejects_unmapped_and_traversal_paths(tmp_path):
    from research.explanation_replication.continuation import relocate
    assert relocate({'path':'/old/root/input'},'/old/root','/new/root')=={'path':'/new/root/input'}
    with pytest.raises(ValueError,match='Unmapped'):
        relocate('/outside/input','/old/root','/new/root')
    with pytest.raises(ValueError,match='Path traversal'):
        relocate('/old/root/../input','/old/root','/new/root')


def test_documented_replication_commands_prepare_and_aggregate(tmp_path):
    root=Path(__file__).resolve().parents[1]
    doc,_=continuation(tmp_path)
    source=tmp_path/'continuation.json';source.write_text(json.dumps(doc))
    output=tmp_path/'linux'
    blocks=re.findall(r'```bash\n(.*?)```',
        (root/'research/explanation_replication/README.md').read_text(),re.S)
    def execute(block):
        command=block.replace('.venv/bin/python',str(Path(sys.executable).resolve()))
        command=command.replace('/path/to/continuation.json',str(source))
        command=command.replace('results/linux-explanations',str(output))
        return subprocess.run(['bash','-c',command],cwd=root,
            env=dict(os.environ,PYTHONPATH=str(root)),text=True,capture_output=True,timeout=30)
    prepared=execute(next(b for b in blocks if '--prepare' in b))
    assert prepared.returncode==0,prepared.stderr
    assert (output/'manifest.json').exists()
    assert not (output/'runs').exists()
    aggregated=execute(next(b for b in blocks if 'replication.aggregate' in b))
    # Preparation deliberately has no fitted outcomes: aggregation must reach
    # its missing-outcome check, not fail command-line parsing.
    assert aggregated.returncode!=0
    assert 'Missing terminal outcome:' in aggregated.stderr,aggregated.stderr
