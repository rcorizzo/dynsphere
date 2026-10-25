"""Matched E1 component study on the frozen extension inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from benchmarks.official_baselines.cached_extension_metrics import load_scores
from benchmarks.official_baselines.extension_campaign import point_metrics, source_digest
from benchmarks.official_baselines.local_controls import DYNSPHERE_CONFIGURATION
from pendulum_detector import MultiConceptEuclideanDetector, _rolling_entity_max


COMPONENTS = ('position', 'euclidean_dynamics', 'raw_dynamics', 'state_speed',
              'velocity', 'regime_distance', 'transition')


def score_configurations(model, test):
    if model.n_concepts_ != 1:
        raise ValueError('Component reuse requires a single fitted regime')
    entities = np.zeros(len(test), dtype=int)
    components = model.decision_components(test, entities)
    result = {'full': components['score']}
    for removed in COMPONENTS[:-1]:
        value = sum(components[name] for name in COMPONENTS if name != removed)
        result[f'without_{removed}'] = _rolling_entity_max(value, entities, model.propagation_radius)
    for radius in (0, 1, 4, 16):
        result[f'radius_{radius}'] = _rolling_entity_max(components['pre_propagation_score'], entities, radius)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.input_csv)
    frame = frame[frame.method_id == 'dynsphere_euclidean_one_regime']
    if frame.empty or frame.run_identity.duplicated().any():
        raise ValueError('Unique frozen E1 reference runs are required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    provenance = {'reference_csv_sha256': hashlib.sha256(args.input_csv.read_bytes()).hexdigest(),
                  'source_sha256': source_digest(), 'manuscript_ready': False,
                  'reference_runs': frame.run_identity.tolist(),
                  'protocol_sha256': hashlib.sha256(Path(__file__).with_name(
                      'COMPONENT_STUDY_PROTOCOL_20261022.md').read_bytes()).hexdigest()}
    (args.output_dir/'protocol.json').write_text(json.dumps(provenance, indent=2))
    for index, row in enumerate(frame.to_dict('records')):
        _, frozen = load_scores(row, args.runs_root)
        run = Path(args.runs_root) / row['run_directory'].rsplit('/runs/', 1)[1]
        train = np.load(run/'case/model_input/train.npy', allow_pickle=False)
        test = np.load(run/'case/model_input/test.npy', allow_pickle=False)
        labels = np.load(run/'case/private/labels.npy', allow_pickle=False)
        started = time.perf_counter()
        model = MultiConceptEuclideanDetector(**{
            **DYNSPHERE_CONFIGURATION, 'n_concepts': 1, 'random_state': int(row['seed'])})
        model.fit(train, np.zeros(len(train), dtype=int))
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = score_configurations(model, test)
        scoring_seconds = time.perf_counter() - started
        if not np.allclose(scores['full'][1:], frozen, rtol=1e-10, atol=1e-10):
            raise ValueError(f'Reconstructed full score differs: {row["run_identity"]}')
        score_path = args.output_dir / f'{row["run_identity"]}.npz'
        np.savez_compressed(score_path, **scores)
        for configuration, values in scores.items():
            rows.append({**{k: row[k] for k in ('case_id', 'trajectory_id', 'source_family', 'seed')},
                         'reference_run_identity': row['run_identity'],
                         'configuration': configuration, **point_metrics(labels, values),
                         'fit_seconds': fit_seconds, 'all_configuration_scoring_seconds': scoring_seconds,
                         'score_archive': str(score_path), 'manuscript_ready': False})
        print(json.dumps({'completed_reference_runs': index + 1, 'of': len(frame),
                          'source_family': row['source_family']}), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir/'per_run.csv', index=False)
    result.groupby(['source_family', 'configuration', 'trajectory_id'])[
        ['auprc', 'auroc', 'pauc_fpr_05']].mean().groupby(
            ['source_family', 'configuration']).mean().to_csv(args.output_dir/'per_family.csv')
    summary = {**provenance, 'status': 'complete', 'completed_reference_runs': len(frame),
               'metric_records': len(rows), 'per_run_csv': str(args.output_dir/'per_run.csv')}
    (args.output_dir/'completion.json').write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    main()
