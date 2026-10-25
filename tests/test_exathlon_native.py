"""Frozen native-source preparation contracts, with hand-checked fixtures."""
import hashlib
import json
import zipfile
import subprocess

import numpy as np
import pandas as pd
import pytest


def api():
    from research.exathlon_native import data, protocol
    return data, protocol


def test_selects_all_eligible_groups_and_nested_archives():
    _, p = api()
    rows = []
    for name in ('1_0_100000_15', '1_0_100000_16', '1_2_100000_68',
                 '4_0_100000_27', '4_0_100000_28', '4_1_100000_61',
                 '7_0_100000_53', '8_1_100000_73'):
        app = name.split('_')[0]
        rows.append({'path': f'data/raw/app{app}/{name}/{name}.zip',
                     'sha': 'a'*40, 'size': 10, 'type': 'blob'})
    cohort, inventory = p.select_cohort({'tree': rows, 'truncated': False})
    assert [g['application'] for g in cohort] == [1, 4]
    assert cohort[1]['tests'][0]['trace'] == '4_1_100000_61'
    assert len(inventory) == 4
    with pytest.raises(ValueError, match='truncated'):
        p.select_cohort({'tree': rows, 'truncated': True})


def test_duplicate_snapshots_are_averaged_before_within_gap_differences():
    d, _ = api()
    required = d.IDENTITY+d.DIFFERENCE+[f'{i}_{f}' for f in d.EXECUTOR for i in range(1, 6)]
    frame = pd.DataFrame({c: [1., 3., 7., 20., 22.] for c in required})
    frame['t'] = [10, 10, 11, 20, 21]
    x, t, entities, audit = d.transform_frame(frame)
    assert t.tolist() == [10, 11, 20, 21]
    assert entities.tolist() == [0, 0, 1, 1]
    assert x[:, 0].tolist() == [2., 7., 20., 22.]
    assert x[:, 3].tolist() == [0., 5., 0., 2.]
    assert audit['duplicate_timestamp_rows_averaged'] == 1
    with pytest.raises(ValueError, match='nondecreasing'):
        d.transform_frame(frame.iloc[::-1])


def test_missing_published_feature_fails_explicitly():
    d, _ = api()
    with pytest.raises(ValueError, match='feature'):
        d.transform_frame(pd.DataFrame({'t': [1, 2]}))


def test_train_only_scaling_and_fixed_support():
    d, _ = api()
    train = np.array([[0., 2.], [2., 4.]])
    cal = np.array([[100., 102.]])
    test = np.array([[200., 202.]])
    a, b, c, info = d.preprocess(train, cal, test)
    assert a.tolist() == [[-1., -1.], [1., 1.]]
    assert b.tolist() == [[99., 99.]]
    assert c.tolist() == [[199., 199.]]
    mask = d.support(np.r_[np.zeros(201), np.ones(200)])
    assert np.flatnonzero(mask).tolist() == [100]


def test_source_bytes_and_immutable_document_are_verified(tmp_path):
    _, p = api()
    blob = tmp_path/'source'
    blob.write_bytes(b'hello')
    expected = hashlib.sha1(b'blob 5\0hello').hexdigest()
    p.verify_blob(blob, expected)
    with pytest.raises(ValueError, match='blob'):
        p.verify_blob(blob, '0'*40)
    path = tmp_path/'frozen.json'
    p.immutable_json(path, {'value': 1})
    p.immutable_json(path, {'value': 1})
    with pytest.raises(ValueError, match='immutable'):
        p.immutable_json(path, {'value': 2})
    assert json.loads(path.read_text()) == {'value': 1}


def test_split_release_archive_is_joined_without_modifying_source_parts(tmp_path):
    d, _ = api()
    raw = tmp_path/'trace.csv'
    content = np.random.default_rng(3).bytes(200000)
    raw.write_bytes(content)
    archive = tmp_path/'trace.zip'
    subprocess.run(['zip', '-q', '-s', '64k', str(archive), raw.name], cwd=tmp_path, check=True)
    parts = sorted(tmp_path.glob('trace.z[0-9][0-9]'))
    assert len(parts) >= 2
    old = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [archive, *parts]}
    joined = d.join_archive(archive, tmp_path/'joined'/'trace.zip')
    with zipfile.ZipFile(joined) as z:
        assert z.read('trace.csv') == content
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in old} == old
