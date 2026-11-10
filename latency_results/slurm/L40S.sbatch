#!/bin/bash
#SBATCH --job-name=carla_latency_L40S
#SBATCH --partition=free-gpu32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=04:00:00
#SBATCH -o /dfs6b/pub/asaha6/agent/PersonaDrive/latency_results/slurm/%x_%j.log

set -euo pipefail
export CARLA_ROOT=/dfs6b/pub/asaha6/agent/PersonaDrive/software/carla0915
BASE_PORT=$((21000 + SLURM_JOB_ID % 200 * 100))
/dfs6b/pub/asaha6/envs/latency_venv/bin/python -u /dfs6b/pub/asaha6/agent/PersonaDrive/scripts/analysis/carla_latency_bench.py run --label L40S --clients 1,2,4 --topologies shared,paired --base-port "$BASE_PORT" --town Town12 --npcs 30 --ticks 600 --warmup 100 --width 1024 --height 512 --inference-ms 0.0 --out /dfs6b/pub/asaha6/agent/PersonaDrive/latency_results
