"""Verified raw release inputs and training-only preparation."""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from research.evaluation_protocols.campaign import ROOT, boundary_mask, digest, preprocess
from .protocol import EXPECTED, OUTPUT, REVISION, immutable_json, select_cohort, verify_blob

IDENTITY = [f'driver_StreamingMetrics_streaming_lastCompletedBatch_{x}_value'
            for x in ('processingDelay', 'schedulingDelay', 'totalDelay')]
DIFFERENCE = [f'driver_StreamingMetrics_streaming_{x}_value' for x in (
    'totalCompletedBatches', 'totalProcessedRecords', 'totalReceivedRecords', 'lastReceivedBatch_records')]
DIFFERENCE += ['driver_BlockManager_memory_memUsed_MB_value', 'driver_jvm_heap_used_value']
DIFFERENCE += [f'node{i}_CPU_ALL_Idle%' for i in range(5, 9)]
EXECUTOR = ['executor_filesystem_hdfs_write_ops_value', 'executor_cpuTime_count',
            'executor_runTime_count', 'executor_shuffleRecordsRead_count',
            'executor_shuffleRecordsWritten_count', 'jvm_heap_used_value']


def support(entities):
    return boundary_mask(entities, 100, 100)


def clean_name(column):
    return column.replace('_'.join(column.split('_')[1:10])+'_', '') if 'StreamingMetrics' in column else column


def required_columns():
    return IDENTITY+DIFFERENCE+[f'{i}_{f}' for f in EXECUTOR for i in range(1, 6)]


def transform_frame(frame):
    required = required_columns()
    if 't' not in frame or any(c not in frame for c in required) or not frame.columns.is_unique:
        raise ValueError('Published feature bundle is missing or ambiguous')
    raw = frame['t'].to_numpy(dtype=float)
    if not len(raw) or not np.isfinite(raw).all() or not (np.diff(raw) >= 0).all():
        raise ValueError('Timestamps must be finite and nondecreasing')
    duplicates = len(frame)-frame['t'].nunique()
    frame = frame[['t', *required]].groupby('t', sort=True, as_index=False).mean()
    timestamps = frame.pop('t').to_numpy(dtype=float)
    entities = np.cumsum(np.r_[0, np.diff(timestamps) > 1]).astype(int)
    frame = frame.fillna(-1)
    for executor in range(1, 6):
        counter = frame[f'{executor}_executor_runTime_count'].to_numpy()
        edges = np.diff(np.r_[False, counter == -1, False].astype(int))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if (start > 0 and stop < len(frame) and entities[start-1] == entities[stop]
                    and counter[stop] >= counter[start-1]):
                columns = [c for c in frame if c.startswith(f'{executor}_')]
                frame.loc[start:stop-1, columns] = frame.loc[start-1, columns].to_numpy()
    os_columns = [c for c in frame if c.startswith('node')]
    for entity in np.unique(entities):
        ids = np.flatnonzero(entities == entity)
        frame.loc[ids, os_columns] = frame.loc[ids, os_columns].replace(-1, np.nan).ffill().bfill().fillna(-1)
    transformed = frame[IDENTITY].copy()
    for name in DIFFERENCE:
        transformed['diff_'+name] = frame.groupby(entities)[name].diff().fillna(0)
    for name in EXECUTOR:
        avg = frame[[f'{i}_{name}' for i in range(1, 6)]].replace(-1, np.nan).mean(axis=1).fillna(-1)
        transformed['diff_avg_'+name] = avg.groupby(entities).diff().fillna(0)
    return transformed.to_numpy(dtype=float), timestamps, entities, {
        'features': list(transformed.columns), 'duplicate_timestamp_rows_averaged': int(duplicates),
        'rows': len(frame), 'entities': int(entities[-1])+1,
        'timestamp_start': float(timestamps[0]), 'timestamp_end': float(timestamps[-1])}


def read_trace(path, events):
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
        if len(members) != 1:
            raise ValueError('Expected exactly one trace CSV')
        with archive.open(members[0]) as stream:
            header = pd.read_csv(stream, nrows=0).columns.tolist()
        mapping = {c: clean_name(c) for c in header}
        selected = [c for c in header if c == 't' or mapping[c] in required_columns()]
        with archive.open(members[0]) as stream:
            frame = pd.read_csv(stream, usecols=selected).rename(columns=mapping)
    values, times, entities, audit = transform_frame(frame)
    name = Path(path).stem
    selected_events = events[events.trace_name == name]
    if name.split('_')[1] == '0' and len(selected_events):
        raise ValueError('Normal execution has event annotations')
    labels = np.zeros(len(values), dtype=np.int8)
    retained_events = 0
    mask = support(entities)
    for event in selected_events.itertuples():
        end = event.root_cause_end if pd.isna(event.extended_effect_end) else event.extended_effect_end
        affected = (times >= event.root_cause_start) & (times <= end)
        labels[affected] = 1
        retained_events += int((affected & mask).any())
    audit.update(trace=name, annotated_events=len(selected_events), retained_events=retained_events,
                 evaluated_rows=int(mask.sum()), positive_rows=int(labels[mask].sum()),
                 negative_rows=int((labels[mask] == 0).sum()), retained_fraction=float(mask.mean()),
                 evaluable=set(np.unique(labels[mask])) == {0, 1})
    return {'values': values, 'labels': labels, 'entities': entities, 'timestamps': times,
            'mask': mask, 'audit': audit}


def fetch(entry, output):
    path = Path(output)/'source'/entry['path']
    pilot = ROOT/'results/exathlon_native_pilot_20261025/source'/entry['path']
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if pilot.exists():
            verify_blob(pilot, entry['blob'])
            shutil.copyfile(pilot, path)
        else:
            url = f'https://raw.githubusercontent.com/exathlonbenchmark/exathlon/{REVISION}/{entry["path"]}'
            temporary = path.with_suffix('.download')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            verify_blob(temporary, entry['blob'])
            temporary.replace(path)
    verify_blob(path, entry['blob'])
    return path


def join_archive(archive, target):
    """Join non-ZIP64 volumes, updating only ZIP directory offsets.

    Layout follows PKWARE APPNOTE sections 4.3.12 and 4.3.16:
    https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
    Every decompressed member is CRC checked before accepting the derivative.
    """
    archive, target = Path(archive), Path(target)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.stem+'.building.zip')
        parts = sorted(archive.parent.glob(archive.stem+'.z[0-9][0-9]'))+[archive]
        offsets = np.r_[0, np.cumsum([part.stat().st_size for part in parts])].tolist()
        if offsets[-1] >= 0xffffffff:
            raise ValueError('ZIP64 split archives are outside this loader')
        with archive.open('rb') as stream:
            stream.seek(max(0, archive.stat().st_size-65557))
            tail = stream.read()
        at = tail.rfind(b'PK\x05\x06')
        if at < 0:
            raise ValueError('Missing ZIP end directory')
        eocd = bytearray(tail[at:at+22])
        _, disk, cd_disk, count_disk, count, cd_size, cd_offset, comment = struct.unpack('<4s4H2LH', eocd)
        if disk != len(parts)-1 or cd_disk >= len(parts) or count == 65535:
            raise ValueError('Missing or unsupported split ZIP volumes')
        central = offsets[cd_disk]+cd_offset
        with temporary.open('wb') as stream:
            for part in parts:
                with part.open('rb') as source:
                    shutil.copyfileobj(source, stream)
        with temporary.open('r+b') as stream:
            position = central
            for _ in range(count):
                stream.seek(position)
                header = bytearray(stream.read(46))
                if header[:4] != b'PK\x01\x02':
                    raise ValueError('Invalid split ZIP central directory')
                volume = struct.unpack_from('<H', header, 34)[0]
                local = struct.unpack_from('<L', header, 42)[0]
                if volume >= len(parts) or local == 0xffffffff:
                    raise ValueError('Unsupported ZIP64 or volume offset')
                struct.pack_into('<H', header, 34, 0)
                struct.pack_into('<L', header, 42, offsets[volume]+local)
                stream.seek(position)
                stream.write(header)
                name, extra, note = struct.unpack_from('<3H', header, 28)
                position += 46+name+extra+note
            struct.pack_into('<4H', eocd, 4, 0, 0, count, count)
            struct.pack_into('<L', eocd, 16, central)
            stream.seek(offsets[-2]+archive.stat().st_size-len(tail)+at)
            stream.write(eocd)
        with zipfile.ZipFile(temporary) as merged:
            if merged.testzip() is not None:
                raise ValueError('Assembled archive CRC differs')
        temporary.replace(target)
    return target


def fetch_trace(entry, tree, output):
    archive = fetch(entry, output)
    stem = Path(entry['path']).with_suffix('')
    parts = [r for r in tree['tree'] if Path(r['path']).with_suffix('') == stem
             and len(Path(r['path']).suffix) == 4 and Path(r['path']).suffix[1:2] == 'z'
             and Path(r['path']).suffix[2:].isdigit()]
    if not parts:
        return archive
    for row in sorted(parts, key=lambda r: r['path']):
        fetch({'path': row['path'], 'blob': row['sha']}, output)
    target = Path(output)/'assembled'/archive.name
    merged = join_archive(archive, target)
    immutable_json(target.with_suffix('.json'), {'source_parts': {r['path']: r['sha'] for r in parts},
                   'source_zip': entry['blob'], 'assembled_sha256': digest(merged)})
    return merged


def save_input(path, trace, values):
    path = Path(path)
    arrays = {k: trace[k] for k in ('labels', 'entities', 'timestamps', 'mask')}
    arrays['values'] = values
    if not np.isfinite(values).all():
        raise ValueError('Prepared features must be finite')
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                raise ValueError('Existing immutable input differs')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        path.with_suffix('.tmp').replace(path)


def prepare(tree, output=OUTPUT):
    output = Path(output)
    cohort, inventory = select_cohort(tree)
    if {g['application']: [t['number'] for t in g['tests']] for g in cohort} != EXPECTED:
        raise ValueError('Release cohort differs from approved scope')
    immutable_json(output/'release_inventory.json', tree)
    immutable_json(output/'eligibility.json', inventory)
    by_path = {r['path']: r for r in tree['tree']}
    shared = {}
    for role, relative in {'labels': 'data/raw/ground_truth.zip', 'license': 'LICENSE',
                           'features': 'src/features/spark_alteration.py'}.items():
        row = by_path[relative]
        shared[role] = fetch({'path': relative, 'blob': row['sha']}, output)
    with zipfile.ZipFile(shared['labels']) as archive:
        member = [n for n in archive.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
        if len(member) != 1:
            raise ValueError('Ambiguous event archive')
        with archive.open(member[0]) as stream:
            events = pd.read_csv(stream)
    audits = []
    for group in cohort:
        fit = read_trace(fetch_trace(group['fit'], tree, output), events)
        cal = read_trace(fetch_trace(group['calibration'], tree, output), events)
        if fit['labels'].any() or cal['labels'].any():
            raise ValueError('Normal fitting/calibration data are contaminated')
        tests = [read_trace(fetch_trace(t, tree, output), events) for t in group['tests']]
        train, calibration, _, prep = preprocess(fit['values'], cal['values'], tests[0]['values'])
        directory = output/'inputs'/f'app{group["application"]}'
        save_input(directory/'fit.npz', fit, train)
        save_input(directory/'calibration.npz', cal, calibration)
        entries = []
        for source, test in zip(group['tests'], tests):
            _, _, values, same = preprocess(fit['values'], cal['values'], test['values'])
            if same != prep:
                raise ValueError('Training-only preparation differs between test executions')
            save_input(directory/(source['trace']+'.npz'), test, values)
            entries.append({**test['audit'], 'test_after_both_normals':
                            test['audit']['timestamp_start'] > max(fit['audit']['timestamp_end'], cal['audit']['timestamp_end'])})
        immutable_json(directory/'preprocessing.json', prep)
        audits.append({'application': group['application'], 'fit': fit['audit'],
                       'calibration': cal['audit'], 'tests': entries})
        print(json.dumps({'prepared_application': group['application'], 'test_support': entries}), flush=True)
    sources = {r['path']: digest(output/'source'/r['path']) for g in cohort
               for r in [g['fit'], g['calibration'], *g['tests']]}
    sources.update({str(p.relative_to(output/'source')): digest(p) for p in shared.values()})
    for path in (output/'source').rglob('*'):
        if path.is_file() and path.suffix[1:2] == 'z' and path.suffix[2:].isdigit():
            sources[str(path.relative_to(output/'source'))] = digest(path)
    return immutable_json(output/'prepared.json', {'cohort': cohort, 'audits': audits,
                          'source_sha256': sources, 'release_revision': REVISION})


def load_input(output, application, name):
    path = Path(output)/'inputs'/f'app{application}'/(name+'.npz')
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
