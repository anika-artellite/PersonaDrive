#!/bin/bash
# Run the agent on a SINGLE LB1-style Bench2Drive route, with video recording.
#
# What this does:
#   1. Boots CARLA on a unique port
#   2. Runs Bench2Drive's leaderboard_evaluator on ONE route XML
#   3. Agent dumps a PNG per ~5 ticks (DEBUG=1) with waypoints + language
#      overlaid; ICL is on if the training run had `use_icl: True`
#   4. After the route finishes, ffmpegs the PNG sequence into an MP4
#
# Required (set via env or edit defaults below):
#   CARLA_ROOT      path to a CARLA 0.9.15 install (CarlaUE4.sh inside)
#   ROUTE_XML       a single B2D-style route XML
#   AGENT_CKPT      path to the driving model (DeepSpeed dir is fine)
#
# Optional ICL knobs (defaults: the retrieval artefacts under persona_drive_rag/):
#   PERSONADRIVE_ICL_PROFILE   default | fast | slow | normal   (default: normal)
#                              (neutral | aggressive | conservative | style-agnostic)
#   PERSONADRIVE_ICL_ROOT      <icl_data>             (default: persona_drive_rag)
#   PERSONADRIVE_ICL_CKPT      <icl_embed_model.pt>   (default: <root>/checkpoints/best_model.pt)
#
# Usage:
#   bash scripts/eval/test_one_lb1_route.sh
#   ROUTE_XML=/abs/path/to/your/route.xml bash scripts/eval/test_one_lb1_route.sh
set -euo pipefail

REPO="${PERSONADRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# 1. Resolve required paths
: "${CARLA_ROOT:=${REPO}/software/carla0915}"   # persistent in-repo install
: "${ROUTE_XML:=${REPO}/Bench2Drive/leaderboard/data/route_28035_town01_tjunction.xml}"   # single Town01 route, ~25 waypoints, T-junction scenario
: "${AGENT_CKPT:=${PERSONADRIVE_CKPT:-${REPO}/pretrained/personadrive_icl_sft/checkpoints/epoch=000.ckpt/pytorch_model.pt}}"
: "${PERSONADRIVE_ICL_PROFILE:=normal}"

if [ ! -d "${CARLA_ROOT}" ] || [ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]; then
    echo "ERROR: CARLA not found at ${CARLA_ROOT}"
    echo "  install CARLA 0.9.15 from https://github.com/carla-simulator/carla/releases/tag/0.9.15"
    echo "  then set CARLA_ROOT=<path>"
    exit 1
fi
if [ ! -f "${ROUTE_XML}" ]; then
    echo "ERROR: route XML not found: ${ROUTE_XML}"
    echo "  for B2D, download bench2drive220.xml from"
    echo "  https://github.com/Thinklab-SJTU/Bench2Drive  ('Eval Tools' section)"
    exit 1
fi
if [ ! -e "${AGENT_CKPT}" ]; then
    echo "ERROR: agent checkpoint missing: ${AGENT_CKPT}"
    exit 1
fi

# ── 2. CARLA + traffic-manager ports.
# If PORT is already exported (because you booted CARLA via scripts/eval/start_carla.sh
# in another terminal), reuse it and skip the boot step. Otherwise allocate
# fresh random ports and launch CARLA inline.
EXTERNAL_CARLA=0
if [ -n "${PORT:-}" ] && python -c "import socket; s=socket.socket(); s.settimeout(0.3); s.connect(('127.0.0.1',${PORT}))" 2>/dev/null; then
    EXTERNAL_CARLA=1
    : "${TM_PORT:=$((PORT + 10000))}"
    echo ">>> using already-running CARLA on port ${PORT} (TM port ${TM_PORT})"
else
    PORT=$((20000 + RANDOM % 1000))
    TM_PORT=$((30000 + RANDOM % 1000))
fi

# 3. Output dirs
# OUT_DIR env (if set) wins fully - sweep wrappers use it for nested layouts
# like .../all_routes/<sweep>/<route>/<profile>/.
# Otherwise build:
#   eval_results/single_route/<route_name><OUT_SUFFIX>/<STAMP>/
# where <route_name> is the route XML basename without .xml. This groups all
# runs of the same route+profile under a stable parent and lets the timestamp
# distinguish individual invocations.
STAMP=$(date +%Y%m%d_%H%M%S)
ROUTE_NAME=$(basename "${ROUTE_XML}" .xml)
if [ -n "${OUT_DIR:-}" ]; then
    OUT_BASE=${OUT_DIR}
else
    OUT_BASE=${REPO}/eval_results/single_route/${ROUTE_NAME}${OUT_SUFFIX:-}/${STAMP}
fi
mkdir -p "${OUT_BASE}/run" "${OUT_BASE}/res" "${OUT_BASE}/log"
RESULT_FILE=${OUT_BASE}/res/result.json
# Trailing slash is REQUIRED - agent_simlingo.py:263 does naive string
# concat:  self.save_path = os.environ.get('SAVE_PATH') + self.save_path_root
# Without the slash, frames land in `<OUT_BASE>/viz<save_path_root>/` (a
# sibling of `viz/`) instead of `<OUT_BASE>/viz/<save_path_root>/`.
SAVE_PATH=${OUT_BASE}/viz/
mkdir -p "${SAVE_PATH}"

# 4. Environment
# CUDA needed for both the driving model AND the FAISS retriever's SigLIP encode
source /usr/share/Modules/init/bash 2>/dev/null || true
module load cuda/12.2.0 2>/dev/null || true

# The agent's DEBUG flag is now driven by env - flip it on so PNGs get saved.
export PERSONADRIVE_AGENT_DEBUG=1
export PERSONADRIVE_ICL_PROFILE
[ -n "${PERSONADRIVE_ICL_ROOT:-}" ] && export PERSONADRIVE_ICL_ROOT
[ -n "${PERSONADRIVE_ICL_CKPT:-}" ] && export PERSONADRIVE_ICL_CKPT

# Standard B2D launcher env vars.
# Notes on CARLA python:
#   - The shipped carla-0.9.15-py3.7 egg in PythonAPI/carla/dist/ is for
#     Python 3.7 and is INCOMPATIBLE with the personadrive env (Python 3.10) - adding it
#     to PYTHONPATH causes a silent hang on import. Skip it; use the
#     `carla==0.9.15` wheel that's already pip-installed in the env.
#   - We still need PythonAPI/carla on the path for `agents.*` helper modules.
export CARLA_ROOT
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
# PYTHONPATH order matters: ${REPO} MUST come before Bench2Drive/leaderboard.
# Both ${REPO}/team_code/ and ${REPO}/Bench2Drive/leaderboard/team_code/ are
# Python packages (have __init__.py). If Bench2Drive/leaderboard is searched
# first, Python loads the WRONG team_code (Bench2Drive's variant, which lacks
# config_simlingo.py) and the agent's `from team_code.config_simlingo` fails.
export PYTHONPATH=${REPO}:${REPO}/team_code:${REPO}/Bench2Drive/scenario_runner:${REPO}/Bench2Drive/leaderboard:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}
export SCENARIO_RUNNER_ROOT=${REPO}/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=${REPO}/Bench2Drive/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=${PORT}
export TM_PORT=${TM_PORT}
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=False
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=only_traj
export GPU_RANK=${GPU_RANK:-0}     # which physical GPU CARLA renders on
export ROUTES=${ROUTE_XML}
export TEAM_AGENT=${REPO}/team_code/agent_simlingo.py
export TEAM_CONFIG=${AGENT_CKPT}
export CHECKPOINT_ENDPOINT=${RESULT_FILE}
export SAVE_PATH

# ── 5. CARLA is launched by Bench2Drive's leaderboard_evaluator itself (see
# Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py:209 - it always
# subprocess.Popens its own CarlaUE4.sh on --port, ignoring any pre-running
# one). So we DON'T pre-launch here; that would just waste ~5 GB of GPU mem
# on a duplicate instance.
#
# To debug CARLA in isolation, use scripts/eval/start_carla.sh in another terminal - but
# kill it before running this script so B2D can claim the port cleanly.
if [ "${EXTERNAL_CARLA}" = "1" ]; then
    echo ">>> WARNING: an existing CARLA is on port ${PORT}. Bench2Drive's"
    echo ">>>          evaluator will try to start another one on the SAME"
    echo ">>>          port and fail. Kill the existing CARLA first:"
    echo ">>>            pkill -9 -f CarlaUE4"
    exit 6
fi
echo ">>> CARLA will be launched by leaderboard_evaluator itself"
echo ">>>   world port  = ${PORT}"
echo ">>>   TM port     = ${TM_PORT}"

# When this script exits, kill any CARLA child the evaluator left dangling.
trap 'pkill -9 -f "carla-rpc-port=${PORT}" 2>/dev/null || true' EXIT

# 6. Run the leaderboard evaluator
echo ">>> launching agent on route: ${ROUTE_XML}"
echo ">>> output dir: ${OUT_BASE}"

# Diagnostic - dump the exact env the python child will inherit
echo ">>> PYTHONPATH being passed to python:"
echo "${PYTHONPATH}" | tr ':' '\n' | sed 's/^/    /'
echo ">>> CWD: $(pwd)"

# Use the simlingo conda env (the one that has all training deps)
PYTHON="${PERSONADRIVE_CONDA_ENV:-${CONDA_PREFIX:-/path/to/conda/envs/personadrive}}/bin/python"

CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u \
    ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --routes="${ROUTES}" \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint="${CHECKPOINT_ENDPOINT}" \
    --agent="${TEAM_AGENT}" \
    --agent-config="${TEAM_CONFIG}" \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME} \
    --port=${PORT} \
    --traffic-manager-port=${TM_PORT} \
    --gpu-rank=${GPU_RANK} \
    2>&1 | tee "${OUT_BASE}/log/agent.log"

# 7. Make the video from the agent's PNG dumps
# Agent saves under: SAVE_PATH/<route_index>/debug_viz/<session>/iter_<N>/<route_type>/<route_number>_<ts>/images/
echo ">>> looking for PNG frames under ${OUT_BASE} ..."
# Search the WHOLE run dir, not just SAVE_PATH - handles the agent's quirky
# path-concat regardless of trailing slash. Picks the dir with most PNGs.
FRAME_DIR=$(find "${OUT_BASE}" -type d -name images 2>/dev/null \
              | while read d; do echo "$(ls "$d"/*.png 2>/dev/null | wc -l) $d"; done \
              | sort -nr | head -1 | awk '{print $2}')
if [ -z "${FRAME_DIR}" ] || [ -z "$(ls "${FRAME_DIR}"/*.png 2>/dev/null | head -1)" ]; then
    echo "WARNING: no PNG frames found — agent may have crashed before any tick fired."
    echo "         check ${OUT_BASE}/log/agent.log for the failure"
    exit 0
fi
echo ">>> frames in ${FRAME_DIR}"
N_FRAMES=$(ls "${FRAME_DIR}"/*.png | wc -l)
echo ">>> ${N_FRAMES} frames; building MP4 ..."

VIDEO_OUT=${OUT_BASE}/drive.mp4
# Frames are named <step>.png with non-contiguous step indices (DEBUG saves
# every 5 ticks). Build a sorted glob via a temp listfile so ffmpeg sees them
# in numeric order.
LISTFILE=${OUT_BASE}/frames.txt
ls -1 "${FRAME_DIR}"/*.png \
    | awk -F/ '{n=$NF; sub(/\.png$/, "", n); print n"\t"$0}' \
    | sort -n \
    | awk '{print "file \x27"$2"\x27"}' \
    > "${LISTFILE}"

# System ffmpeg if it has libx264, otherwise the static binary bundled with the
# env's imageio-ffmpeg package (has libx264; many clusters ship ffmpeg without it).
FFMPEG_BIN="$(command -v ffmpeg 2>/dev/null || true)"
if [ -z "${FFMPEG_BIN}" ] || ! "${FFMPEG_BIN}" -hide_banner -encoders 2>/dev/null | grep libx264 >/dev/null; then
    FFMPEG_BIN="$("${PYTHON}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
fi
if [ -n "${FFMPEG_BIN}" ] && "${FFMPEG_BIN}" -y -loglevel error -r 4 -f concat -safe 0 -i "${LISTFILE}" \
        -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        "${VIDEO_OUT}"; then
    echo ">>> wrote ${VIDEO_OUT}"
else
    echo "WARNING: no ffmpeg with libx264 found (PATH or imageio-ffmpeg); PNG sequence is in ${FRAME_DIR}"
    echo "         to make the video later: ffmpeg -r 4 -f concat -safe 0 -i ${LISTFILE} -c:v libx264 -pix_fmt yuv420p ${VIDEO_OUT}"
fi

echo ">>> done. results: ${RESULT_FILE}"
echo ">>> video:        ${VIDEO_OUT}"
