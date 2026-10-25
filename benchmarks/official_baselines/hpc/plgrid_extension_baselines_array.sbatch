#!/usr/bin/env bash
#SBATCH --job-name=dynsphere-extension-baselines
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/plgrid-extension-baselines-%A-%a.out
#SBATCH --error=logs/plgrid-extension-baselines-%A-%a.err
# Supply allocation, partition, and --gres=gpu:1 when using group=gpu.
set -euo pipefail
exec bash "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO}/benchmarks/official_baselines/hpc/run_extension_baselines.sh"
