#!/bin/bash
# run_scenario.sh [--day] <ScenarioType> [route_xml_basename]
#
# Drive ONE route of a given scenario type, for screenshots, demos and smoke tests.
#
# Unlike collect_session.sh this does not touch the driven-route cache, does not loop
# over personas, and does not transfer anything - so it leaves the collected dataset
# completely untouched. Output goes to a scratch directory ($SCRATCH_ROOT).
#
#   bash run_scenario.sh SignalizedJunctionLeftTurn
#   bash run_scenario.sh HazardAtSideLane 1055.xml
#   bash run_scenario.sh --day HighwayExit          # force clear daylight weather
#   bash run_scenario.sh --list                     # show available scenario types
#
# The wheel works here exactly as in a collection run. Keyboard fallback, no wheel needed:
#   W / UP      throttle          S / DOWN   brake
#   A / LEFT    steer left        D / RIGHT  steer right
#   SPACE       handbrake         Q          toggle reverse
#   P           save a window screenshot (does not pause driving)
#
# Every frame's front camera is also saved as a jpg under $SAVE_PATH/*/rgb/, so the run
# produces a folder of frames in addition to the live window.
#
# Configuration is by environment variable - source your env.sh first. See README.md.

set -uo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

require_var() {
    if [ -z "${!1:-}" ]; then
        echo "ERROR: $1 is not set. Source your env.sh first (see env.example.sh)."
        exit 1
    fi
}
# Nothing is strictly required: every tree is bundled, and CARLA_ROOT is unused at runtime.

# The leaderboard, scenario_runner, SimLingo and CARLA PythonAPI trees are all bundled in
# this package, so nothing external is needed but the simulator itself. Set any of these
# to override the bundled copy with your own checkout.
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${PKG_ROOT}/leaderboard}"
export SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-${PKG_ROOT}/scenario_runner}"
export SIMLINGO_ROOT="${SIMLINGO_ROOT:-${PKG_ROOT}/simlingo}"

# The CARLA client package. Its version must match the running server exactly. Follows an
# overridden LEADERBOARD_ROOT if that checkout carries a PythonAPI, else uses the bundled
# one. Note the bundled copy omits carla/dist/ - install the client with pip instead.
if [ -z "${CARLA_PYTHONAPI:-}" ]; then
    if [ -d "${LEADERBOARD_ROOT}/../PythonAPI/carla" ]; then
        export CARLA_PYTHONAPI="${LEADERBOARD_ROOT}/../PythonAPI/carla"
    else
        export CARLA_PYTHONAPI="${PKG_ROOT}/PythonAPI/carla"
    fi
fi
export CARLA_ROOT

PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${CARLA_PYTHONAPI}"

if [ -z "${ROUTES_DIRS:-}" ]; then
    echo "ERROR: ROUTES_DIRS is not set. The route sets are not bundled; point it at"
    echo "       SimLingo's routes_training directories. See env.example.sh."
    exit 1
fi
read -r -a ROUTE_DIRS <<< "$ROUTES_DIRS"

OUT_ROOT="${SCRATCH_ROOT:-${PKG_ROOT}/scratch}"

if [ "${1:-}" = "--list" ]; then
    for d in "${ROUTE_DIRS[@]}"; do
        find "$d" -name '*.xml' ! -name '*:*' -exec grep -ho 'type="[A-Za-z0-9_]*"' {} +
    done 2>/dev/null | sed 's/type="//;s/"//' | sort -u
    exit 0
fi
if [ $# -lt 1 ]; then
    echo "Usage: $0 [--day] <ScenarioType> [route_xml_basename]"
    echo "       $0 --list"
    exit 1
fi

FORCE_DAY=0
if [ "${1:-}" = "--day" ]; then FORCE_DAY=1; shift; fi

SCENARIO="$1"
ROUTE_XML=""
if [ -n "${2:-}" ]; then
    for d in "${ROUTE_DIRS[@]}"; do
        if [ -f "$d/$2" ]; then ROUTE_XML="$d/$2"; break; fi
    done
else
    for d in "${ROUTE_DIRS[@]}"; do
        ROUTE_XML=$(grep -rl "type=\"${SCENARIO}\"" "$d" --include='*.xml' 2>/dev/null | head -1)
        [ -n "$ROUTE_XML" ] && break
    done
fi
if [ -z "${ROUTE_XML:-}" ] || [ ! -f "$ROUTE_XML" ]; then
    echo "ERROR: no route found for scenario '$SCENARIO' in: ${ROUTE_DIRS[*]}"
    echo "       Run '$0 --list' to see available types."
    exit 1
fi

ROUTE_NAME=$(basename "$ROUTE_XML" .xml)
TOWN=$(grep -o 'town="[A-Za-z0-9]*"' "$ROUTE_XML" | head -1 | sed 's/town="//;s/"//')
[ -z "$TOWN" ] && TOWN=Town12
export TOWN

# The shipped route sets are heavily night-biased, and the brightest routes often carry
# heavy fog or rain. --day rewrites the weather block in a COPY, leaving the original
# route untouched; geometry, waypoints and scenarios are unchanged.
if [ "$FORCE_DAY" = "1" ]; then
    mkdir -p "${OUT_ROOT}/_day_routes"
    DAY_XML="${OUT_ROOT}/_day_routes/${ROUTE_NAME}_day.xml"
    python3 - "$ROUTE_XML" "$DAY_XML" <<'PYEOF'
import sys, xml.etree.ElementTree as ET
src, dst = sys.argv[1], sys.argv[2]
CLEAR = {"cloudiness":"5.0", "precipitation":"0.0", "precipitation_deposits":"0.0",
         "wetness":"0.0", "wind_intensity":"5.0", "sun_azimuth_angle":"90.0",
         "sun_altitude_angle":"65.0", "fog_density":"2.0"}
t = ET.parse(src); r = t.getroot()
n = 0
for w in r.iter('weather'):
    for k, v in CLEAR.items():
        w.set(k, v)
    n += 1
t.write(dst, xml_declaration=True, encoding='utf-8')
print(f"  weather: forced clear daylight on {n} entr{'y' if n==1 else 'ies'} (sun 65 deg, no rain/fog)")
PYEOF
    ROUTE_XML="$DAY_XML"
fi

export ROUTES="$ROUTE_XML"
export TEAM_AGENT="${TEAM_AGENT:-${PKG_ROOT}/human_agent.py}"
export DATAGEN=1                 # saves rgb/lidar/boxes - the rgb frames are the screenshots
export REPETITIONS=1
export REPETITION=0
export DEBUG_CHALLENGE=1
export DISPLAY="${DISPLAY:-:0}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"

export SAVE_PATH="${OUT_ROOT}/${SCENARIO}/${ROUTE_NAME}"
# Window capture: the full pygame frame including the speed/command overlay, which the
# sensor jpgs under rgb/ do not have.
# CARLA_SCREENSHOT_EVERY=N saves every Nth frame; 0 or unset means P-key-only capture.
export CARLA_SCREENSHOT_DIR="${SAVE_PATH}/window"
export CARLA_SCREENSHOT_EVERY="${CARLA_SCREENSHOT_EVERY:-0}"
export CHECKPOINT_ENDPOINT="${SAVE_PATH}/result.json"
mkdir -p "$SAVE_PATH"

echo "========================================================"
echo "  Scenario : $SCENARIO"
echo "  Route    : $ROUTE_NAME   (town $TOWN)"
echo "  Output   : $SAVE_PATH"
echo ""
echo "  Keyboard: W/S throttle+brake, A/D steer, SPACE handbrake, Q reverse"
echo "  Press P anytime for a window screenshot (does not pause driving)"
echo "========================================================"
echo ""

python3 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
    --host="${CARLA_HOST:-127.0.0.1}" \
    --port="${CARLA_PORT:-2000}" \
    --routes="${ROUTES}" \
    --track=MAP \
    --repetitions="${REPETITIONS}" \
    --agent="${TEAM_AGENT}" \
    --agent-config="${ROUTE_XML}" \
    --debug="${DEBUG_CHALLENGE}" \
    --checkpoint="${CHECKPOINT_ENDPOINT}" \
    --resume=0

echo ""
echo "Saved frames:"
find "$SAVE_PATH" -name '*.jpg' 2>/dev/null | wc -l | sed 's/^/  /'
echo "  -> $SAVE_PATH"
