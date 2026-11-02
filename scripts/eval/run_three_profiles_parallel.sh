#!/bin/bash
# Run multiple ICL profiles IN PARALLEL - one GPU per profile.
# Use this when you have 2+ GPUs allocated (e.g. SLURM `--gres=gpu:3`).
# Wallclock = max(per-profile time) instead of sum, so a 30-min sweep
# becomes 10 min on 3 GPUs.
#
# Usage:
#   bash scripts/eval/run_three_profiles_parallel.sh                     # all 3 in parallel
#   bash scripts/eval/run_three_profiles_parallel.sh slow                # just slow (1 GPU)
#   bash scripts/eval/run_three_profiles_parallel.sh normal fast         # 2 in parallel
#
# Selection precedence:
#   1. positional args   2. PROFILES env var   3. default: normal fast slow
#
# Each run gets:
#   - its own GPU (CUDA_VISIBLE_DEVICES + CARLA --gpu-rank)
#   - its own random CARLA RPC port
#   - its own output dir (suffixed with _<profile>)
#   - its own per-profile log file under eval_results/single_route/
#
# After all parallel jobs finish, prints a summary + run-dir paths.
set -uo pipefail

REPO="${PERSONADRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# Resolve which profiles to run.
if [ $# -gt 0 ]; then
    PROFILES="$*"
else
    PROFILES="${PROFILES:-normal fast slow}"
fi
for p in ${PROFILES}; do
    case "${p}" in
        normal|fast|slow) ;;
        *) echo "ERROR: unknown profile '${p}'  (valid: normal, fast, slow)" >&2; exit 2 ;;
    esac
done

# How many GPUs are visible?
N_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
N_GPUS=${N_GPUS:-1}
N_PROF=$(echo ${PROFILES} | wc -w)

echo "=== parallel ICL-profile sweep ==="
echo "  profiles : ${PROFILES}  (count = ${N_PROF})"
echo "  GPUs seen: ${N_GPUS}"

if [ "${N_PROF}" -gt "${N_GPUS}" ]; then
    echo
    echo "WARNING: ${N_PROF} profiles but only ${N_GPUS} GPU(s)."
    echo "  Multiple profiles will share a GPU — risk of OOM and slowdown."
    echo "  Either reduce profiles, or use the sequential version:"
    echo "    bash scripts/eval/run_three_profiles.sh ${PROFILES}"
    echo "  Continuing in 5s ..."
    sleep 5
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR=${REPO}/eval_results/single_route/_sweep_${STAMP}
mkdir -p "${SWEEP_DIR}"
SUMMARY=${SWEEP_DIR}/summary.log
echo "  sweep logs: ${SWEEP_DIR}"
echo
echo "  per-profile logs:"

# Make sure no stale CARLA / agent processes hold GPU memory
pkill -9 -f "leaderboard_evaluator" 2>/dev/null || true
pkill -9 -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
pkill -9 -f "CarlaUE4.sh"            2>/dev/null || true
sleep 5

# Launch all profiles in parallel, one per GPU
declare -a PIDS=()
declare -a LOGS=()
declare -a NAMES=()
i=0
for prof in ${PROFILES}; do
    GPU_IDX=$(( i % N_GPUS ))                       # round-robin GPU assignment
    LOG=${SWEEP_DIR}/${prof}.log
    LOGS+=("${LOG}")
    NAMES+=("${prof}")
    echo "    [GPU ${GPU_IDX}]  ${prof}  →  ${LOG}"

    # Environment for this child:
    #  CUDA_VISIBLE_DEVICES=N  -> PyTorch sees only this GPU as cuda:0
    #  GPU_RANK=N              -> CARLA's Unreal Engine renders on this GPU
    #  PERSONADRIVE_ICL_PROFILE    -> ICL profile to load
    #  OUT_SUFFIX              -> dir suffix so profiles don't collide
    #  PORT/TM_PORT unset      -> launcher picks fresh random ports
    (
        export CUDA_VISIBLE_DEVICES=${GPU_IDX}
        export GPU_RANK=${GPU_IDX}
        export PERSONADRIVE_ICL_PROFILE=${prof}
        export OUT_SUFFIX="_${prof}"
        unset PORT TM_PORT
        bash scripts/eval/test_one_lb1_route.sh
    ) > "${LOG}" 2>&1 &
    PIDS+=($!)

    i=$((i + 1))
    sleep 3   # brief stagger so CARLA boots don't all hit ImportAssets at once
done

echo
echo "  launched ${#PIDS[@]} parallel jobs  (PIDs: ${PIDS[*]})"
echo "  monitor any one with:  tail -f ${SWEEP_DIR}/<profile>.log"
echo "  monitor all GPUs  :    watch -n 5 nvidia-smi"
echo
echo "  waiting for all to finish ..."

START=$(date +%s)

# Wait for all background jobs; capture each exit code
declare -a EXITS=()
for idx in "${!PIDS[@]}"; do
    wait "${PIDS[$idx]}"
    EXITS+=($?)
done

DURATION=$(( $(date +%s) - START ))

# Final cleanup
pkill -9 -f "leaderboard_evaluator|CarlaUE4" 2>/dev/null || true
sleep 3

# Summary
{
    echo "=== sweep finished in ${DURATION}s ==="
    for idx in "${!NAMES[@]}"; do
        echo "  ${NAMES[$idx]:>8s}  exit=${EXITS[$idx]}  log=${LOGS[$idx]}"
    done
    echo
    echo "Output dirs:"
    ls -td "${REPO}/eval_results/single_route/${STAMP%_*}"*_normal \
           "${REPO}/eval_results/single_route/${STAMP%_*}"*_fast   \
           "${REPO}/eval_results/single_route/${STAMP%_*}"*_slow   \
           2>/dev/null | head -10
    echo
    echo "Open viz_app at http://localhost:7860/ to view all of them."
} | tee "${SUMMARY}"
