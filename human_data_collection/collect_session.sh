#!/bin/bash
# collect_session.sh  <name>  <num_routes>
#
# Drive routes in CARLA under human control and record a PersonaDrive session.
#
# <name>        : Participant name. Data is nested under persona directories:
#                   $DATASET_ROOT/slow/<name>/<scenario_type>/<route>/
#                   $DATASET_ROOT/default/<name>/<scenario_type>/<route>/
#                   $DATASET_ROOT/fast/<name>/<scenario_type>/<route>/
# <num_routes>  : Total number of routes to drive this session.
#                 Routes are chosen from $ROUTES_DIRS using a weighting system that
#                 prioritises scenario types that are globally undersampled across all
#                 participants, so coverage converges to even over time.
#                 Routes already driven by ANY participant are never repeated.
#                 Pass 0 to run the single devtest route instead.
#                 Pass "long" to drive ALL undriven routes in $LONG_ROUTES_DIR in
#                 sequence (scenario_type stored as "long_route").
#
# Each selected route is driven 3 times in sequence: slow -> default -> fast. The driver
# is shown the persona instruction in the terminal before each run.
# Example: bash collect_session.sh person1 5
#   5 routes x 3 personas = 15 total drives.
#
# Configuration is entirely by environment variable. Copy env.example.sh to env.sh,
# edit the paths, and `source env.sh` before running. See README.md.
#
# Duplicate avoidance:
#   Startup rebuilds cache/driven_routes.json from every dataset root, recording
#   which personas of which routes already have a result.json. Both this script
#   and select_routes.py read that one cache, so they cannot disagree about what
#   is finished. Completion is tracked per persona, so a session interrupted
#   after "slow" resumes at "default" instead of redriving or abandoning the route.
#   Set INCLUDE_PARTIAL=0 to leave partially-driven routes out of the selection.
#   The cache is derived data - delete cache/ and it rebuilds on the next run.
#
#   Routes collected outside these dataset roots (an earlier dataset, another rig)
#   have no result.json to find, so they are listed in markdown manifests instead.
#   Any *.md in a dataset root's logs/ directory is parsed and merged on every
#   rebuild, and those routes are excluded outright. Drop a new .md in logs/ and it
#   is honoured on the next run.
#
# Usage examples:
#   bash collect_session.sh person1 5    # drive 5 routes (15 total drives)
#   bash collect_session.sh person2 3    # drive 3 routes, weighted toward gaps left by person1
#   bash collect_session.sh debug 0      # run devtest route only
#   bash collect_session.sh person1 long # drive all undriven long routes

# -u: error on unbound variables  -o pipefail: error if any pipe command fails
# Intentionally omitting -e: the leaderboard evaluator exits non-zero on route crashes,
# which would kill the whole session under set -e. Errors are handled explicitly instead.
set -uo pipefail

# ---------- args ----------
if [ $# -lt 2 ]; then
    echo "Usage: $0 <name> <num_routes|long>"
    exit 1
fi
SESSION_NAME="$1"
NUM_ROUTES="$2"

# Validate: must be a non-negative integer or the literal "long"
if [[ "$NUM_ROUTES" != "long" ]] && ! [[ "$NUM_ROUTES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: <num_routes> must be a non-negative integer or \"long\""
    exit 1
fi

# ---------- configuration ----------
PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

require_var() {
    if [ -z "${!1:-}" ]; then
        echo "ERROR: $1 is not set. Source your env.sh first (see env.example.sh)."
        exit 1
    fi
}
# Only the output root cannot be bundled or defaulted. CARLA_ROOT is optional: nothing in
# the runtime path reads it, and on a split setup (server on Windows, client in WSL2) there
# is no CarlaUE4.sh on this side at all. Set it only if your own tooling wants it.
for _v in DATASET_ROOT; do
    require_var "$_v"
done

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

if [ ! -d "${SIMLINGO_ROOT}/team_code" ]; then
    echo "ERROR: no team_code/ under SIMLINGO_ROOT=${SIMLINGO_ROOT}"
    exit 1
fi
export CARLA_ROOT

PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH}:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${CARLA_PYTHONAPI}"

# The agent shipped in this package. leaderboard_evaluator.py puts the directory
# holding the agent on sys.path, so wheel_hidraw.py is picked up as a sibling.
export TEAM_AGENT="${TEAM_AGENT:-${PKG_ROOT}/human_agent.py}"

export DATAGEN=1            # DataAgent records rgb / rgb_augmented / lidar / boxes
export REPETITIONS=1
export REPETITION=0
export DEBUG_CHALLENGE=1    # lets the agent read the privileged route for its HUD

# Pygame needs a real display. Under WSL2 this is the WSLg X server.
export DISPLAY="${DISPLAY:-:0}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-x11}"

# Route pool for the weighted selection: a space-separated list of directories, each
# searched recursively for *.xml. The route sets are SimLingo's released route XMLs
# (~13k files) and are NOT bundled here, so this has no usable default - set it in
# env.sh. The devtest route below ships with the leaderboard and needs none of them.
# The route XML sets are bundled under simlingo/data/, so collection works out of the
# box. Override ROUTES_DIRS to draw from your own route pool instead.
_BUNDLED_ROUTES="${SIMLINGO_ROOT}/data/simlingo/training_1_scenario/routes_training ${SIMLINGO_ROOT}/data/simlingo/training_3_scenarios/routes_training"
ROUTES_DIRS="${ROUTES_DIRS:-${_BUNDLED_ROUTES}}"
read -r -a BASE_ROUTES_DIRS <<< "$ROUTES_DIRS"

LONG_ROUTES_DIR="${LONG_ROUTES_DIR:-${SIMLINGO_ROOT}/data/simlingo/long_routes}"
DEVTEST_XML="${DEVTEST_XML:-${LEADERBOARD_ROOT}/data/routes_devtest.xml}"

# Counts how many LiDAR points fall inside each recorded bounding box.
# Export COMPUTE_SCRIPT= (empty) to skip the step.
COMPUTE_SCRIPT="${COMPUTE_SCRIPT-${PKG_ROOT}/compute_bbox_points.py}"

# The three persona instructions, driven in this order. These are also the directory
# names in the dataset, so they must match the released data: the middle persona is
# "default". Changing this list means changing the PERSONAS tuple in select_routes.py
# and build_driven_cache.py too.
PERSONAS=("slow" "default" "fast")

# Cache of everything already driven. Rebuilt at startup and updated after each
# persona finishes, so both this script and select_routes.py skip the same work.
# Derived data only - safe to delete, it rebuilds from the dataset on next run.
#
# Deliberately OUTSIDE this directory. The cache records the dataset paths it found,
# which embed participant names, so writing it into a checkout risks publishing them.
CACHE_DIR="${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/personadrive}"
DRIVEN_CACHE="${CACHE_DIR}/driven_routes.json"
INDEX_CACHE="${CACHE_DIR}/route_index.json"
BUILD_CACHE="${PKG_ROOT}/build_driven_cache.py"

# Dataset root - all participants write here.
mkdir -p "$DATASET_ROOT"
mkdir -p "$DATASET_ROOT/logs"

# Optional second root, for rigs that record onto a fast local disk and keep the
# dataset somewhere larger. Finished routes are transferred there at the end of the
# session. Leave ARCHIVE_ROOT unset and everything simply stays in DATASET_ROOT.
ARCHIVE_ROOT="${ARCHIVE_ROOT:-}"
if [ -n "$ARCHIVE_ROOT" ] && ! mkdir -p "$ARCHIVE_ROOT" 2>/dev/null; then
    echo "WARNING: could not create ARCHIVE_ROOT: $ARCHIVE_ROOT"
    echo "         Is it mounted? Data will stay in $DATASET_ROOT."
    ARCHIVE_ROOT=""
fi

echo "========================================================"
echo "  Participant : $SESSION_NAME"
echo "  Dataset dir : $DATASET_ROOT"
echo "  Archive dir : ${ARCHIVE_ROOT:-(none — data stays in the dataset dir)}"
if [ "$NUM_ROUTES" = "long" ]; then
    echo "  Mode        : long routes (all undriven routes in $LONG_ROUTES_DIR)"
elif [ "$NUM_ROUTES" -eq 0 ]; then
    echo "  Mode        : devtest ($(basename "$DEVTEST_XML"))"
else
    echo "  Mode        : $NUM_ROUTES route(s) x 3 personas = $((NUM_ROUTES * 3)) total drives"
fi
echo "========================================================"
echo ""

# ---------- driven-route cache ----------
# A route counts as done only if result.json exists inside the route directory,
# meaning the leaderboard evaluator actually completed and wrote its output.
# A bare directory (created by mkdir before the run) does NOT count.
#
# The cache collapses what would otherwise be a `find` per candidate route into one
# scan at startup plus O(1) lookups, and it records completion per persona rather
# than per route - so a route interrupted after "slow" resumes at "default" instead
# of being written off as done.

declare -A ROUTE_COMPLETE   # route_name           -> 1 when all personas are done
declare -A ROUTE_PERSONA    # "route_name|persona" -> 1 when that drive is done

dataset_dirs_args() {
    printf '%s\n' "$DATASET_ROOT"
    [ -n "$ARCHIVE_ROOT" ] && printf '%s\n' "$ARCHIVE_ROOT"
}

# Markdown manifests listing routes collected outside these roots - an earlier
# dataset, another rig. Picked up from the logs dir of every dataset root, so
# dropping a new .md there is enough to have it honoured on the next run.
collected_md_args() {
    local base
    while read -r base; do
        [ -d "$base/logs" ] || continue
        find "$base/logs" -maxdepth 1 -name '*.md' ! -name '*:*' 2>/dev/null
    done < <(dataset_dirs_args)
}

refresh_driven_cache() {
    if [ ! -f "$BUILD_CACHE" ]; then
        echo "WARNING: $BUILD_CACHE not found — duplicate detection is degraded."
        return 1
    fi
    mapfile -t _ds < <(dataset_dirs_args)
    mapfile -t _md < <(collected_md_args)
    python3 "$BUILD_CACHE" --out "$DRIVEN_CACHE" \
        --dataset-dirs "${_ds[@]}" \
        --collected-md "${_md[@]}" || return 1
}

load_driven_cache() {
    ROUTE_COMPLETE=()
    ROUTE_PERSONA=()
    [ -f "$DRIVEN_CACHE" ] || return 0
    while IFS=$'\t' read -r _route _persona _complete; do
        [ -n "$_route" ] || continue
        ROUTE_PERSONA["${_route}|${_persona}"]=1
        [ "$_complete" = "1" ] && ROUTE_COMPLETE["$_route"]=1
    done < <(python3 "$BUILD_CACHE" --out "$DRIVEN_CACHE" --dump-tsv 2>/dev/null)
    return 0
}

# Record one finished drive immediately, so a crash mid-session doesn't lose it.
mark_driven() {
    local route_name="$1" persona="$2" scenario_type="$3" save_path="$4"
    ROUTE_PERSONA["${route_name}|${persona}"]=1
    local all_done=1
    for p in "${PERSONAS[@]}"; do
        [ -n "${ROUTE_PERSONA[${route_name}|${p}]:-}" ] || all_done=0
    done
    [ "$all_done" = "1" ] && ROUTE_COMPLETE["$route_name"]=1
    [ -f "$BUILD_CACHE" ] || return 0
    python3 "$BUILD_CACHE" --out "$DRIVEN_CACHE" \
        --mark "$route_name" --mark-persona "$persona" \
        --mark-scenario "$scenario_type" --mark-path "$save_path" || true
}

route_already_done() {
    [ -n "${ROUTE_COMPLETE[$1]:-}" ]
}

persona_already_done() {
    [ -n "${ROUTE_PERSONA[${1}|${2}]:-}" ]
}

# ---------- cleanup: remove incomplete route directories ----------
# Runs at startup. Any route directory missing result.json was created by mkdir
# but never completed (the script was killed mid-run). Safe to remove.
cleanup_incomplete_routes() {
    local cleaned=0
    for base_dir in "$DATASET_ROOT" "${ARCHIVE_ROOT:-}"; do
        [ -d "$base_dir" ] || continue
        for persona in "${PERSONAS[@]}"; do
            local persona_dir="${base_dir}/${persona}"
            [ -d "$persona_dir" ] || continue
            for person_dir in "$persona_dir"/*/; do
                [ -d "$person_dir" ] || continue
                for scenario_dir in "$person_dir"*/; do
                    [ -d "$scenario_dir" ] || continue
                    for route_dir in "$scenario_dir"*/; do
                        [ -d "$route_dir" ] || continue
                        if [ ! -f "${route_dir}/result.json" ]; then
                            echo "  [cleanup] Removing incomplete: $route_dir"
                            rm -rf "$route_dir"
                            cleaned=$((cleaned + 1))
                        fi
                    done
                done
            done
        done
    done
    if [ "$cleaned" -gt 0 ]; then
        echo "  [cleanup] Removed $cleaned incomplete route director(ies)."
    fi
}

# ---------- log: cumulative across sessions for this participant ----------
LOG_FILE="${DATASET_ROOT}/logs/${SESSION_NAME}.txt"
ROUTES_DRIVEN=()  # scenario_type/route_name pairs driven this session

write_log() {
    local all_routes=()
    for persona in "${PERSONAS[@]}"; do
        local base="${DATASET_ROOT}/${persona}/${SESSION_NAME}"
        [ -d "$base" ] || continue
        for scenario_dir in "$base"/*/; do
            [ -d "$scenario_dir" ] || continue
            local sname
            sname=$(basename "$scenario_dir")
            for route_dir in "$scenario_dir"*/; do
                [ -d "$route_dir" ] || continue
                all_routes+=("${sname}/$(basename "$route_dir")")
            done
        done
    done
    all_routes+=("${ROUTES_DRIVEN[@]}")
    mapfile -t unique_routes < <(printf '%s\n' "${all_routes[@]}" | sort -u)

    {
        echo "Participant  : $SESSION_NAME"
        echo "Routes driven: ${#unique_routes[@]}"
        echo ""
        echo "Routes:"
        for route in "${unique_routes[@]}"; do
            echo "  - $route"
        done
    } > "$LOG_FILE"
}

# ---------- helper: run one route across all three personas ----------
run_route() {
    local route_xml="$1"
    local scenario_type="$2"

    # SimLingo's DataAgent names its output folder after $TOWN, so it has to be set.
    local TOWN
    TOWN=$(echo "$(basename "$route_xml")" | grep -o 'Town[0-9]*')
    [ -z "$TOWN" ] && TOWN="Town12"
    export TOWN=$TOWN

    local ROUTE_NAME
    ROUTE_NAME=$(basename "$route_xml" .xml)
    export ROUTES="$route_xml"

    # Guard: skip if every persona is already recorded for this route.
    if route_already_done "$ROUTE_NAME"; then
        echo "  [SKIP] $ROUTE_NAME already driven for all personas — skipping."
        return
    fi

    # Which personas still need driving? A route can be partially done if an
    # earlier session was interrupted part-way through.
    local todo=()
    for persona in "${PERSONAS[@]}"; do
        if persona_already_done "$ROUTE_NAME" "$persona"; then
            echo "  [SKIP] ${persona} already recorded for ${ROUTE_NAME}."
        else
            todo+=("$persona")
        fi
    done
    if [ ${#todo[@]} -eq 0 ]; then
        echo "  [SKIP] nothing left to drive for ${ROUTE_NAME}."
        return
    fi
    if [ ${#todo[@]} -lt ${#PERSONAS[@]} ]; then
        echo "  [RESUME] ${ROUTE_NAME}: driving only ${todo[*]}"
    fi

    # Show a bird's-eye view of the route before driving. Continues on Enter.
    python3 "${PKG_ROOT}/preview_route.py" --route "$route_xml" --town "$TOWN" || true

    for persona in "${todo[@]}"; do
        while true; do
            echo ""
            echo "========================================================"
            echo "  PERSONA : ${persona^^}"
            case "$persona" in
                slow)    echo "  Drive slowly and cautiously. Prioritize safety." ;;
                default) echo "  Drive in a normal, everyday manner." ;;
                fast)    echo "  Drive fast and aggressively. Push the pace." ;;
            esac
            echo "  Scenario: $scenario_type"
            echo "  Route   : $ROUTE_NAME"
            echo "========================================================"
            echo ""

            export SAVE_PATH="${DATASET_ROOT}/${persona}/${SESSION_NAME}/${scenario_type}/${ROUTE_NAME}"
            export CHECKPOINT_ENDPOINT="${SAVE_PATH}/result.json"
            mkdir -p "$SAVE_PATH"

            echo "  -> Running [${persona}]: ${scenario_type}/${ROUTE_NAME}"

            python3 "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
                --host="${CARLA_HOST:-127.0.0.1}" \
                --port="${CARLA_PORT:-2000}" \
                --routes="${ROUTES}" \
                --track=MAP \
                --repetitions="${REPETITIONS}" \
                --agent="${TEAM_AGENT}" \
                --agent-config="${route_xml}" \
                --debug="${DEBUG_CHALLENGE}" \
                --checkpoint="${CHECKPOINT_ENDPOINT}" \
                --resume=1

            echo "  -> Finished [${persona}]: ${scenario_type}/${ROUTE_NAME}"

            if [ -n "$COMPUTE_SCRIPT" ]; then
                echo "  -> Computing bounding box LiDAR hit counts..."
                if [ -f "$COMPUTE_SCRIPT" ]; then
                    python3 "$COMPUTE_SCRIPT" --dataset_root="$SAVE_PATH" \
                        && echo "  -> Bounding box computation complete." \
                        || echo "  -> WARNING: compute_bbox_points.py errored for ${ROUTE_NAME}."
                else
                    echo "  -> WARNING: $COMPUTE_SCRIPT not found — skipping."
                fi
            fi

            echo ""
            echo "[PAUSED after ${persona^^}] Press R to retry this persona, any other key to continue..."
            stty sane 2>/dev/null
            read -r -s -n 1 _key < /dev/tty

            if [ "${_key,,}" = "r" ]; then
                echo "  -> Retrying [${persona^^}]..."
                rm -rf "$SAVE_PATH"
            else
                # Record it only if the evaluator actually wrote its output -
                # otherwise a crashed run would be cached as done and never redriven.
                if [ -f "$CHECKPOINT_ENDPOINT" ]; then
                    mark_driven "$ROUTE_NAME" "$persona" "$scenario_type" "$SAVE_PATH"
                else
                    echo "  -> WARNING: no result.json for [${persona}] — not recorded as driven."
                fi
                sleep 1
                break
            fi
        done
    done

    ROUTES_DRIVEN+=("${scenario_type}/${ROUTE_NAME}")
    write_log

    echo ""
    if route_already_done "$ROUTE_NAME"; then
        echo "[ROUTE COMPLETE] All ${#PERSONAS[@]} personas driven for ${scenario_type}/${ROUTE_NAME}"
    else
        local still=()
        for persona in "${PERSONAS[@]}"; do
            persona_already_done "$ROUTE_NAME" "$persona" || still+=("$persona")
        done
        echo "[ROUTE PARTIAL] ${scenario_type}/${ROUTE_NAME} still missing: ${still[*]}"
    fi
}

# ---------- prepare the driven-route cache ----------
# Order matters: drop half-finished directories first so they are never cached
# as driven, then rebuild from every dataset root and load into memory. Every
# later duplicate check is an array lookup against what this produces.
summarize_cache() {
    local -A _seen=()
    local _key
    for _key in "${!ROUTE_PERSONA[@]}"; do
        _seen["${_key%%|*}"]=1
    done
    local _total=${#_seen[@]}
    local _complete=${#ROUTE_COMPLETE[@]}
    echo "  ${_total} route(s) touched — ${_complete} complete, $((_total - _complete)) partial"
}

echo "Scanning for incomplete route directories..."
cleanup_incomplete_routes
echo ""

mkdir -p "$CACHE_DIR"
echo "Building driven-route cache..."
refresh_driven_cache || true
load_driven_cache
summarize_cache
echo "  -> $DRIVEN_CACHE"
echo "  -> ${DRIVEN_CACHE%.json}.txt  (readable summary)"
echo ""

# ---------- transfer finished routes to the archive root ----------
# rsync, not mv: the earlier mv-based version only warned on failure and carried on,
# which left orphaned and truncated route directories behind. -rlt rather than -a,
# because -pog cannot set owner/group/permissions on most mounted Windows or network
# filesystems and makes rsync fail there. --remove-source-files deletes only what
# rsync confirms it transferred, so a partial transfer leaves the source intact.
transfer_to_archive() {
    [ -n "$ARCHIVE_ROOT" ] || return 0
    [ -d "$DATASET_ROOT" ] || return 0

    echo ""
    echo "========================================================"
    echo "Transferring collected data to $ARCHIVE_ROOT"
    echo "========================================================"

    if ! command -v rsync >/dev/null 2>&1; then
        echo "  rsync not installed — leaving the data in $DATASET_ROOT."
        return 0
    fi

    local failed=0 moved=0
    for persona in "${PERSONAS[@]}"; do
        local src="${DATASET_ROOT}/${persona}/${SESSION_NAME}"
        [ -d "$src" ] || continue
        local dst="${ARCHIVE_ROOT}/${persona}/${SESSION_NAME}"
        mkdir -p "$dst"
        echo "  -> ${persona}/${SESSION_NAME}"
        if rsync -rlt --remove-source-files "$src/" "$dst/"; then
            moved=$((moved + 1))
        else
            echo "     WARNING: rsync failed. Data remains at $src."
            failed=$((failed + 1))
        fi
    done

    if [ -f "$LOG_FILE" ]; then
        mkdir -p "${ARCHIVE_ROOT}/logs"
        rsync -lt --remove-source-files "$LOG_FILE" "${ARCHIVE_ROOT}/logs/" \
            && echo "  -> logs/${SESSION_NAME}.txt" \
            || echo "     WARNING: log transfer failed. Log remains at $LOG_FILE."
    fi

    echo ""
    echo "Transfer complete: $moved persona tree(s) moved, $failed failed."

    # rsync leaves the directory skeleton behind; only prune what is now empty.
    find "$DATASET_ROOT" -mindepth 1 -type d -empty -delete 2>/dev/null || true
    if [ -n "$(find "$DATASET_ROOT" -mindepth 1 -type f -print -quit 2>/dev/null)" ]; then
        echo "Note: some files remain in $DATASET_ROOT (failed transfers are preserved)."
    fi

    # The transfer invalidated the paths recorded during the session, so rebuild the
    # cache against where the data actually lives now.
    echo ""
    echo "Rebuilding driven-route cache after transfer..."
    refresh_driven_cache || true
    load_driven_cache
    summarize_cache
}

# ---------- long routes mode ----------
if [ "$NUM_ROUTES" = "long" ]; then
    if [ -z "$LONG_ROUTES_DIR" ]; then
        echo "ERROR: LONG_ROUTES_DIR is not set. Point it at SimLingo's long_routes"
        echo "       directory — the route sets are not bundled. See env.example.sh."
        exit 1
    fi
    if [ ! -d "$LONG_ROUTES_DIR" ]; then
        echo "ERROR: long routes directory not found: $LONG_ROUTES_DIR"
        exit 1
    fi

    # Collect all XML files, excluding the sidecar files Windows leaves on
    # downloaded archives (their names contain a colon).
    mapfile -t LONG_XMLS < <(find "$LONG_ROUTES_DIR" -maxdepth 1 -name "*.xml" \
        ! -name "*:*" | sort)

    if [ ${#LONG_XMLS[@]} -eq 0 ]; then
        echo "ERROR: No XML files found in $LONG_ROUTES_DIR"
        exit 1
    fi

    # Filter out already-driven routes
    PENDING_XMLS=()
    for xml in "${LONG_XMLS[@]}"; do
        route_name=$(basename "$xml" .xml)
        if route_already_done "$route_name"; then
            echo "  [SKIP] $route_name already driven — skipping."
        else
            PENDING_XMLS+=("$xml")
        fi
    done

    if [ ${#PENDING_XMLS[@]} -eq 0 ]; then
        echo "All long routes have already been driven. Nothing to do."
        exit 0
    fi

    echo ""
    echo "========================================================"
    echo "  LONG ROUTES SESSION PLAN  ($SESSION_NAME)"
    echo "  ${#PENDING_XMLS[@]} route(s) remaining of ${#LONG_XMLS[@]} total"
    echo "========================================================"
    idx=1
    for xml in "${PENDING_XMLS[@]}"; do
        printf "  %2d. %s\n" "$idx" "$(basename "$xml" .xml)"
        idx=$((idx + 1))
    done
    echo "========================================================"
    echo ""

    total_run=0
    for xml in "${PENDING_XMLS[@]}"; do
        echo "========================================================"
        echo "Long Route $((total_run + 1))/${#PENDING_XMLS[@]}: $(basename "$xml" .xml)"
        echo "========================================================"
        run_route "$xml" "long_route"
        total_run=$((total_run + 1))
    done

    echo "========================================================"
    echo "Long routes session complete!"
    echo "  Routes driven this session : $total_run  (x3 personas = $((total_run * 3)) total drives)"
    echo "  Data saved to              : $DATASET_ROOT"
    echo "  Log file                   : $LOG_FILE"
    echo "========================================================"
    transfer_to_archive
    exit 0
fi

# ---------- devtest mode ----------
if [ "$NUM_ROUTES" -eq 0 ]; then
    if [ ! -f "$DEVTEST_XML" ]; then
        echo "ERROR: devtest XML not found: $DEVTEST_XML"
        exit 1
    fi
    run_route "$DEVTEST_XML" "devtest"
    echo "Devtest complete."
    exit 0
fi

# ---------- weighted route selection ----------
if [ ${#BASE_ROUTES_DIRS[@]} -eq 0 ] || [ -z "${BASE_ROUTES_DIRS[0]}" ]; then
    echo "ERROR: ROUTES_DIRS is not set, so there are no routes to select from."
    echo "       The route sets are SimLingo's released route XMLs and are not bundled;"
    echo "       point ROUTES_DIRS at them in env.sh (see env.example.sh)."
    echo "       To smoke-test the rig without them: $0 $SESSION_NAME 0"
    exit 1
fi

echo "Selecting $NUM_ROUTES route(s) weighted by global scenario coverage..."
echo ""

# Dataset dirs are still passed so the selector can fall back to a live scan if the
# cache is missing or unreadable.
DATASET_DIRS_ARGS=("--dataset-dirs" "$DATASET_ROOT")
[ -n "$ARCHIVE_ROOT" ] && DATASET_DIRS_ARGS+=("$ARCHIVE_ROOT")

# Offer partially-driven routes again so their missing personas get filled in;
# run_route() then drives only what the cache says is outstanding.
INCLUDE_PARTIAL_ARGS=()
[ "${INCLUDE_PARTIAL:-1}" = "1" ] && INCLUDE_PARTIAL_ARGS+=("--include-partial")

# Call the Python selector - "xml_path<TAB>scenario_type" per line on stdout,
# diagnostics (coverage table, weights) on stderr so they appear in the terminal.
mapfile -t SELECTION_LINES < <(
    python3 "${PKG_ROOT}/select_routes.py" \
        --routes-dirs "${BASE_ROUTES_DIRS[@]}" \
        "${DATASET_DIRS_ARGS[@]}" \
        --driven-cache "$DRIVEN_CACHE" \
        --index-cache "$INDEX_CACHE" \
        "${INCLUDE_PARTIAL_ARGS[@]}" \
        --n "$NUM_ROUTES"
)

if [ ${#SELECTION_LINES[@]} -eq 0 ]; then
    echo "ERROR: Route selection returned no routes. Check that the directories in"
    echo "       ROUTES_DIRS exist and still contain undriven routes."
    exit 1
fi

echo ""
echo "========================================================"
echo "  SESSION PLAN  ($SESSION_NAME)"
echo "========================================================"
idx=1
for line in "${SELECTION_LINES[@]}"; do
    route_xml="${line%%$'\t'*}"
    scenario_type="${line##*$'\t'}"
    printf "  %2d. %-38s  %s\n" "$idx" "$(basename "$route_xml" .xml)" "$scenario_type"
    idx=$((idx + 1))
done
echo "========================================================"
echo ""

# ---------- drive the selected routes ----------
total_run=0
for line in "${SELECTION_LINES[@]}"; do
    route_xml="${line%%$'\t'*}"
    scenario_type="${line##*$'\t'}"

    echo "========================================================"
    echo "Route $((total_run + 1))/${#SELECTION_LINES[@]}: $scenario_type"
    echo "========================================================"

    run_route "$route_xml" "$scenario_type"
    total_run=$((total_run + 1))
done

echo "========================================================"
echo "Session complete!"
echo "  Routes driven this session : $total_run  (x3 personas = $((total_run * 3)) total drives)"
echo "  Data saved to              : $DATASET_ROOT"
echo "  Log file                   : $LOG_FILE"
echo "========================================================"

transfer_to_archive
