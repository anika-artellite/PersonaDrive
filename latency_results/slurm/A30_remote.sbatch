#!/bin/bash
#SBATCH --job-name=carla_latency_A30_remote
#SBATCH --partition=free-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:1
#SBATCH --time=03:00:00
#SBATCH -o /dfs6b/pub/asaha6/agent/PersonaDrive/latency_results/slurm/%x_%j.log

set -euo pipefail
export CARLA_ROOT=/dfs6b/pub/asaha6/agent/PersonaDrive/software/carla0915
BASE_PORT=$((21000 + SLURM_JOB_ID % 200 * 100))
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" /dfs6b/pub/asaha6/envs/latency_venv/bin/python -u /dfs6b/pub/asaha6/agent/PersonaDrive/scripts/analysis/carla_latency_bench.py serve \
    --base-port "$BASE_PORT" --log /dfs6b/pub/asaha6/agent/PersonaDrive/latency_results/slurm/carla_$SLURM_JOB_ID.log &
SERVER_STEP=$!
trap 'kill $SERVER_STEP 2>/dev/null || true' EXIT
for _ in $(seq 1 900); do
    timeout 1 bash -c "</dev/tcp/${NODES[0]}/$BASE_PORT" 2>/dev/null && break
    sleep 1
done
srun --nodes=1 --ntasks=1 --nodelist="${NODES[1]}" /dfs6b/pub/asaha6/envs/latency_venv/bin/python -u /dfs6b/pub/asaha6/agent/PersonaDrive/scripts/analysis/carla_latency_bench.py run \
    --label A30_remote --external-host "${NODES[0]}" --topologies shared \
    --clients 1,2,4 --base-port "$BASE_PORT" --town Town12 --npcs 30 --ticks 600 --warmup 100 --width 1024 --height 512 --inference-ms 0.0 --out /dfs6b/pub/asaha6/agent/PersonaDrive/latency_results
