import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('group,methods,device', [
    ('gpu', ['catch', 'mtad_gat', 'anomaly_transformer', 'tranad'], 'cuda'),
    ('cpu', ['dagmm', 'omni_anomaly_tsb_ad'], 'cpu'),
    ('predictor', ['predictive_euclidean_single'], 'cpu'),
])
def test_extension_runner_selects_methods_device_and_zero_based_shard(tmp_path, group, methods, device):
    executable = tmp_path/'python'
    executable.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$DYNSPHERE_DEVICE" "$@"\n')
    executable.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT/'benchmarks/official_baselines/hpc/run_extension_baselines.sh'),
                             '--dry-run'], env={**os.environ, 'DYNSPHERE_REPO': str(ROOT),
                                                'DYNSPHERE_PYTHON': str(executable),
                                                'DYNSPHERE_EXTENSION_GROUP': group, 'LSB_JOBINDEX': '7'},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == device
    start, stop = lines.index('--methods')+1, lines.index('--seeds')
    assert lines[start:stop] == methods
    assert lines[lines.index('--shard-index')+1] == '6'
    assert lines[lines.index('--baseline-profile')+1] == 'audited-v2'
