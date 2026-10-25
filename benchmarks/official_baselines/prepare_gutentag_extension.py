"""Generate a fixed multivariate study using official GutenTAG 1.5.0."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys


GENERATOR_VERSION = '1.5.0'


def generation_config(length=8000):
    if length < 4000:
        raise ValueError('Use at least 4000 samples for this fixed mechanism study')
    mechanisms = [
        {'kind': 'frequency', 'frequency_factor': 2.0},
        {'kind': 'pattern-shift', 'shift_by': 20, 'transition_window': 25},
        {'kind': 'mode-correlation'},
        {'kind': 'amplitude', 'amplitude_factor': 1.5},
        {'kind': 'mean', 'offset': 0.75},
        {'kind': 'variance', 'variance': 0.15},
    ]
    cases = []
    for mechanism in mechanisms:
        if mechanism['kind'] == 'mode-correlation':
            oscillations = [{'kind': 'random-mode-jump', 'frequency': 20,
                             'random-seed': 7, 'channel-diff': 0, 'channel-offset': 1,
                             'variance': 0.02} for _ in range(3)]
        else:
            oscillations = [{'kind': 'sine', 'frequency': frequency, 'amplitude': 1.0,
                             'variance': 0.02} for frequency in (1.0, 1.3, 0.7)]
        cases.append({
            'name': mechanism['kind'], 'length': length,
            'semi-supervised': True, 'supervised': False,
            'base-oscillations': oscillations,
            'anomalies': [{'exact-position': length // 2, 'length': length // 20,
                           'channel': 0, 'kinds': [mechanism]}],
        })
    return {'timeseries': cases}


def generate(output: Path, seeds=(101, 102, 103), length=8000):
    from gutenTAG import GutenTAG
    import pandas as pd

    if version('timeeval-gutenTAG') != GENERATOR_VERSION:
        raise ValueError(f'Requires timeeval-gutenTAG=={GENERATOR_VERSION}')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Generation seeds must be nonempty and unique')
    config = generation_config(length)
    # Refuse replacement of an earlier generation, including partial output.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'generation_config.json').write_text(json.dumps(config, indent=2) + '\n')
    cases = []
    for seed in seeds:
        directory = output / f'generation-{seed}'
        directory.mkdir()
        GutenTAG(seed=seed, n_jobs=1).load_config_dict(config).generate(output_folder=directory)
        for specification in config['timeseries']:
            name = specification['name']
            train = directory / name / 'train_no_anomaly.csv'
            test = directory / name / 'test.csv'
            training, testing = pd.read_csv(train), pd.read_csv(test)
            if training['is_anomaly'].any() or not testing['is_anomaly'].any():
                raise ValueError(f'Invalid normal training or missing injection: {seed}/{name}')
            cases.append({
                'family': 'GutenTAG', 'case_name': f'{name}-g{seed}',
                'train': str(train.relative_to(output)), 'test': str(test.relative_to(output)),
                'mechanism': name, 'generation_seed': seed,
                'train_sha256': hashlib.sha256(train.read_bytes()).hexdigest(),
                'test_sha256': hashlib.sha256(test.read_bytes()).hexdigest(),
            })
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
                            capture_output=True, text=True).stdout
    (output / 'generator_environment.txt').write_text(freeze)
    manifest = {'generator': 'official TimeEval/GutenTAG', 'version': GENERATOR_VERSION,
                'repository': 'https://github.com/TimeEval/GutenTAG',
                'generation_seeds': list(seeds), 'length': length, 'cases': cases,
                'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                'limitations': 'One fixed severity and base family per mechanism; synthetic evidence only.'}
    target = output / 'native_manifest.json'
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 102, 103])
    parser.add_argument('--length', type=int, default=8000)
    args = parser.parse_args()
    print(generate(args.output_dir, args.seeds, args.length))


if __name__ == '__main__':
    main()
