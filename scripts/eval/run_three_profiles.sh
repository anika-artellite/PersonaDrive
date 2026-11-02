#!/bin/bash
# Run the same Bench2Drive route N TIMES - once per ICL profile
# (default / normal / fast / slow). Each run lands in its own timestamped+
# tagged output dir so the viz_app at http://localhost:7860/ shows them side
# by side.
#
# Usage:
#   bash scripts/eval/run_three_profiles.sh                     # all 4 profiles
#   bash scripts/eval/run_three_profiles.sh slow                # just slow
#   bash scripts/eval/run_three_profiles.sh default fast        # subset
#   ROUTE_XML=/path/to/route.xml bash scripts/eval/run_three_profiles.sh slow
#
# Selection precedence (highest wins):
#   1. positional args:   bash scripts/eval/run_three_profiles.sh slow
#   2. PROFILES env var:  PROFILES="slow" bash scripts/eval/run_three_profiles.sh
#   3. default:           default normal fast slow
#
# Other env knobs forwarded to scripts/eval/test_one_lb1_route.sh:
#   ROUTE_XML     route XML to drive (default: route_28035 town01 t-junction)
#   AGENT_CKPT    driving model checkpoint
set -uo pipefail   # NOT -e: don't abort the whole sweep on one route's failure

REPO="${PERSONADRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# Resolve which profiles to run. Positional args > env var > default.
if [ $# -gt 0 ]; then
    PROFILES="$*"
else
    PROFILES="${PROFILES:-default normal fast slow}"
fi

# Validate before kicking anything off - fail loud, not silently mid-sweep.
for p in ${PROFILES}; do
    case "${p}" in
        default|normal|fast|slow) ;;
        *) echo "ERROR: unknown profile '${p}'  (valid: default, normal, fast, slow)" >&2; exit 2 ;;
    esac
done
SUMMARY_LOG=${REPO}/eval_results/single_route/_sweep_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "${SUMMARY_LOG}")"

echo "=== multi-profile sweep ==="                            | tee    "${SUMMARY_LOG}"
echo "  route: ${ROUTE_XML:-<launcher default>}"              | tee -a "${SUMMARY_LOG}"
echo "  profiles: ${PROFILES}"                                | tee -a "${SUMMARY_LOG}"
echo "  summary log: ${SUMMARY_LOG}"                          | tee -a "${SUMMARY_LOG}"
echo                                                          | tee -a "${SUMMARY_LOG}"

START_ALL=$(date +%s)

for prof in ${PROFILES}; do
    echo "════════════════════════════════════════════════════════════════════════════════"            | tee -a "${SUMMARY_LOG}"
    echo "▶ Profile: ${prof}    starting $(date)"             | tee -a "${SUMMARY_LOG}"
    echo "════════════════════════════════════════════════════════════════════════════════"            | tee -a "${SUMMARY_LOG}"

    # Make sure no stragglers from a previous profile
    pkill -9 -f "leaderboard_evaluator" 2>/dev/null || true
    pkill -9 -f "CarlaUE4-Linux-Shipping" 2>/dev/null || true
    pkill -9 -f "CarlaUE4.sh"           2>/dev/null || true
    sleep 5

    PROF_START=$(date +%s)

    # Per-run env: profile + dir suffix. Don't preset PORT - let the launcher
    # pick a fresh random one so killed processes don't collide on TIME_WAIT.
    PERSONADRIVE_ICL_PROFILE=${prof} \
    OUT_SUFFIX="_${prof}" \
        bash scripts/eval/test_one_lb1_route.sh 2>&1 | tee -a "${SUMMARY_LOG}"

    EXIT=${PIPESTATUS[0]}
    PROF_DUR=$(( $(date +%s) - PROF_START ))
    echo                                                      | tee -a "${SUMMARY_LOG}"
    echo "▶ Profile ${prof} finished after ${PROF_DUR}s (exit ${EXIT})"  | tee -a "${SUMMARY_LOG}"
    echo                                                      | tee -a "${SUMMARY_LOG}"
done

# Final cleanup - don't leave CARLA holding the GPU
pkill -9 -f "leaderboard_evaluator|CarlaUE4" 2>/dev/null || true
sleep 3

TOTAL=$(( $(date +%s) - START_ALL ))
echo "════════════════════════════════════════════════════════════════════════════════"            | tee -a "${SUMMARY_LOG}"
echo "ALL PROFILES DONE in ${TOTAL}s"                         | tee -a "${SUMMARY_LOG}"
echo                                                          | tee -a "${SUMMARY_LOG}"
echo "Outputs (open viz_app and look for the suffixed runs):" | tee -a "${SUMMARY_LOG}"
# Layout: eval_results/single_route/<route_name>_<profile>/<STAMP>/
# Show the most recent stamp under each <route>_<profile> parent.
for prof in default normal fast slow; do
    for parent in "${REPO}/eval_results/single_route/"*_${prof}; do
        [ -d "${parent}" ] || continue
        latest=$(ls -td "${parent}"/*/ 2>/dev/null | head -1)
        [ -n "${latest}" ] && echo "${latest}" | tee -a "${SUMMARY_LOG}"
    done
done
echo                                                          | tee -a "${SUMMARY_LOG}"
echo "Summary log: ${SUMMARY_LOG}"                            | tee -a "${SUMMARY_LOG}"
