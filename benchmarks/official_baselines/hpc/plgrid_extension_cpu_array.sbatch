#!/bin/bash -l
#SBATCH --job-name=dynsphere-extension
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-15%4
#SBATCH --output=logs/plgrid-extension-%A-%a.out
#SBATCH --error=logs/plgrid-extension-%A-%a.err
set -euo pipefail
: "${DYNSPHERE_REPO:?Set the code repository path}"
if [[ -n "${DYNSPHERE_PYTHON_MODULE:-}" ]]; then
  module load "$DYNSPHERE_PYTHON_MODULE"
fi
exec bash "$DYNSPHERE_REPO/benchmarks/official_baselines/hpc/run_extension_cpu.sh"
