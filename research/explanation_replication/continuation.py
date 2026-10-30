"""Validate relocated scientific manifests before independent replication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relocate(value, old_root, new_root):
    if isinstance(value, dict):
        return {relocate(k, old_root, new_root): relocate(v, old_root, new_root)
                for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, old_root, new_root) for v in value]
    if isinstance(value, str) and value.startswith('/'):
        try:
            relative = Path(value).relative_to(old_root)
        except ValueError:
            raise ValueError('Unmapped absolute path: ' + value)
        if '..' in relative.parts:
            raise ValueError('Path traversal')
        return str(Path(new_root) / relative)
    return value


def validate_continuation(doc):
    if identity({k: v for k, v in doc.items() if k != 'manifest_id'}) != doc['manifest_id']:
        raise ValueError('Changed continuation')
    if 'parent_manifest_path' in doc:
        parent = json.loads(Path(doc['parent_manifest_path']).read_text())
        if identity({k: v for k, v in parent.items() if k != 'manifest_id'}) != doc['parent_manifest_id']:
            raise ValueError('Changed immutable parent')
        prior = dict(parent)
        prior.pop('runtimes')
        prior.pop('manifest_id')
        expected = relocate(prior, doc['path_map']['parent_root'], doc['path_map']['execution_root'])
        for key, value in expected.items():
            if key != 'bindings' and doc.get(key) != value:
                raise ValueError('Changed inherited scientific field: ' + key)
        if any(doc['bindings'].get(key) != value for key, value in expected['bindings'].items()):
            raise ValueError('Changed parent binding')
    for path, sha in doc['bindings'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('Changed binding: ' + path)
