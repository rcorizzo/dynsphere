#!/usr/bin/env bash
#SBATCH --job-name=dynsphere-extension-vus
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/plgrid-extension-vus-%A-%a.out
#SBATCH --error=logs/plgrid-extension-vus-%A-%a.err
set -euo pipefail
: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the code checkout}"
exec bash "$DYNSPHERE_REPO/benchmarks/official_baselines/hpc/run_extension_metrics.sh"
