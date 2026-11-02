#!/bin/bash
# Sweep multiple routes x multiple ICL profiles.
#
# For each route, runs all requested profiles IN PARALLEL (one per GPU). After
# all profiles for that route finish, moves to the next route. Output is
# laid out as:
#
#   eval_results/all_routes/<sweep_ts>/
#       <route_basename>/
#           normal/   <- full scripts/eval/test_one_lb1_route.sh output (log/, res/, viz/, drive.mp4)
#           fast/
#           slow/
#       <next_route>/
#           ...
#       sweep.log     <- combined per-job exit codes + timings
#
# Inputs (positional args):
#   bash scripts/eval/run_all_routes_profiles.sh                          # default glob
#   bash scripts/eval/run_all_routes_profiles.sh route_28035_*.xml        # explicit files
#   bash scripts/eval/run_all_routes_profiles.sh /abs/path/route_*.xml    # glob from any dir
#
# Default route glob: Bench2Drive/leaderboard/data/route_*.xml
# (use extract_one_route.py first if you want individual XMLs from bench2drive220)
#
# Env knobs:
#   PROFILES  space-separated profile list (default: normal fast slow)
#   N_GPUS    override GPU count (default: nvidia-smi reports)
#   AGENT_CKPT  forwarded to scripts/eval/test_one_lb1_route.sh
set -uo pipefail

REPO="${PERSONADRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# Resolve route list
if [ $# -gt 0 ]; then
    # Positional args may be globs (already expanded by shell) or explicit paths
    ROUTES=("$@")
else
    # Default: every per-route XML in the B2D data dir
    shopt -s nullglob
    ROUTES=(${REPO}/Bench2Drive/leaderboard/data/route_*.xml)
    shopt -u nullglob
fi
if [ ${#ROUTES[@]} -eq 0 ]; then
    echo "ERROR: no route XMLs found." >&2
    echo "       extract some first:" >&2
    echo "         python extract_one_route.py --master Bench2Drive/leaderboard/data/bench2drive220.xml --index 0 --out Bench2Drive/leaderboard/data/route_0.xml" >&2
    exit 2
fi
# Validate every route XML exists before kicking off a multi-hour sweep
for r in "${ROUTES[@]}"; do
    [ -f "${r}" ] || { echo "ERROR: route XML missing: ${r}" >&2; exit 2; }
done

# Resolve profile list
PROFILES="${PROFILES:-normal fast slow}"
for p in ${PROFILES}; do
    case "${p}" in
        normal|fast|slow) ;;
        *) echo "ERROR: unknown profile '${p}' (valid: normal, fast, slow)" >&2; exit 2 ;;
    esac
done
N_PROF=$(echo ${PROFILES} | wc -w)

# GPU count
if [ -z "${N_GPUS:-}" ]; then
    N_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
    N_GPUS=${N_GPUS:-1}
fi

# Sweep dir + summary log
SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR=${REPO}/eval_results/all_routes/${SWEEP_TS}
mkdir -p "${SWEEP_DIR}"
SUMMARY=${SWEEP_DIR}/sweep.log

{
    echo "=== route × profile sweep ==="
    echo "  sweep dir : ${SWEEP_DIR}"
    echo "  routes    : ${#ROUTES[@]}"
    for r in "${ROUTES[@]}"; do echo "              $(basename ${r})"; done
    echo "  profiles  : ${PROFILES}  (${N_PROF})"
    echo "  GPUs seen : ${N_GPUS}"
    if [ "${N_PROF}" -gt "${N_GPUS}" ]; then
        echo "  WARNING: more profiles than GPUs — profiles will share a GPU per route"
    fi
    echo
} | tee "${SUMMARY}"

# Pre-cleanup
pkill -9 -f "leaderboard_evaluator|CarlaUE4" 2>/dev/null || true
sleep 5

SWEEP_START=$(date +%s)

# Outer loop: routes (sequential)
ROUTE_IDX=0
for ROUTE_PATH in "${ROUTES[@]}"; do
    ROUTE_IDX=$((ROUTE_IDX + 1))
    ROUTE_NAME=$(basename "${ROUTE_PATH}" .xml)
    ROUTE_DIR=${SWEEP_DIR}/${ROUTE_NAME}
    mkdir -p "${ROUTE_DIR}"

    {
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo "▶ Route ${ROUTE_IDX}/${#ROUTES[@]}: ${ROUTE_NAME}    started $(date)"
        echo "  path: ${ROUTE_PATH}"
        echo "════════════════════════════════════════════════════════════════════════════════"
    } | tee -a "${SUMMARY}"

    ROUTE_START=$(date +%s)
    declare -a PIDS=() PROF_NAMES=() PROF_LOGS=()

    # Inner loop: profiles (parallel within this route)
    i=0
    for prof in ${PROFILES}; do
        GPU_IDX=$(( i % N_GPUS ))
        OUT_DIR_THIS=${ROUTE_DIR}/${prof}
        mkdir -p "${OUT_DIR_THIS}"
        LOG=${OUT_DIR_THIS}/launcher.log

        echo "  [GPU ${GPU_IDX}]  ${prof}  →  ${OUT_DIR_THIS}" | tee -a "${SUMMARY}"

        (
            export CUDA_VISIBLE_DEVICES=${GPU_IDX}
            export GPU_RANK=${GPU_IDX}
            export PERSONADRIVE_ICL_PROFILE=${prof}
            export OUT_DIR=${OUT_DIR_THIS}
            export ROUTE_XML=${ROUTE_PATH}
            unset PORT TM_PORT OUT_SUFFIX
            bash scripts/eval/test_one_lb1_route.sh
        ) > "${LOG}" 2>&1 &

        PIDS+=($!)
        PROF_NAMES+=("${prof}")
        PROF_LOGS+=("${LOG}")
        i=$((i + 1))
        sleep 3   # stagger CARLA boots
    done

    # Wait for this route's profile parallel batch to finish
    declare -a EXITS=()
    for idx in "${!PIDS[@]}"; do
        wait "${PIDS[$idx]}"
        EXITS+=($?)
    done

    # Per-route cleanup of any straggler CARLA / agent processes
    pkill -9 -f "leaderboard_evaluator|CarlaUE4" 2>/dev/null || true
    sleep 3

    ROUTE_DUR=$(( $(date +%s) - ROUTE_START ))
    {
        echo "  Finished ${ROUTE_NAME} in ${ROUTE_DUR}s"
        for j in "${!PROF_NAMES[@]}"; do
            echo "    ${PROF_NAMES[$j]:>8s}  exit=${EXITS[$j]}  ${PROF_LOGS[$j]}"
        done
        echo
    } | tee -a "${SUMMARY}"

    unset PIDS PROF_NAMES PROF_LOGS EXITS
done

# Final cleanup + summary
pkill -9 -f "leaderboard_evaluator|CarlaUE4" 2>/dev/null || true
sleep 3

SWEEP_DUR=$(( $(date +%s) - SWEEP_START ))
{
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "SWEEP DONE in ${SWEEP_DUR}s   (${#ROUTES[@]} routes × ${N_PROF} profiles)"
    echo "Output tree:"
    find "${SWEEP_DIR}" -maxdepth 3 -type d | head -40
    echo
    echo "Summary log: ${SUMMARY}"
    echo "Open viz_app to browse:"
    echo "  http://localhost:7860/"
} | tee -a "${SUMMARY}"