# PersonaDrive data collection - environment template.
#
#   cp env.example.sh env.sh
#   $EDITOR env.sh
#   source env.sh
#
# Then every script in this directory works. A stale shell is the usual cause of a
# "not set" error.
#
# Everything the rig needs is bundled in this repository: the CARLA Leaderboard 2.0 and
# scenario_runner trees, SimLingo's team_code, the map rasters, the route XML sets and
# CARLA's pure-Python API helpers. Only the simulator itself and an output directory
# cannot be bundled, so those are the only two variables you must set.

# ---------------------------------------------------------------------------
# Required
# ---------------------------------------------------------------------------

# Where sessions are recorded. All participants and all personas share this root:
#   $DATASET_ROOT/<persona>/<participant>/<scenario_type>/<route>/
# Keep it OUTSIDE this checkout: recorded paths embed participant names.
export DATASET_ROOT="$HOME/personadrive_dataset"

# ---------------------------------------------------------------------------
# Optional - bundled copies are used when these are unset
# ---------------------------------------------------------------------------

# Unpacked CARLA 0.9.15 simulator (the directory holding CarlaUE4.sh). Nothing in the
# runtime path reads this, and on a split setup (server on Windows, client in WSL2) there
# is no CarlaUE4.sh on this side. Set it only if your own tooling wants it.
# export CARLA_ROOT="$HOME/carla/CARLA_0.9.15"

# Vendored trees. Set one only to point at your own checkout instead. Note that the
# bundled leaderboard already carries the two collection patches it needs
# (route_scenario.py debug drawing, route_parser.py Python 3.9); an external checkout
# does not, and collection is wrong without them.
# export LEADERBOARD_ROOT="$PWD/leaderboard"
# export SCENARIO_RUNNER_ROOT="$PWD/scenario_runner"
# export SIMLINGO_ROOT="$PWD/simlingo"

# CARLA Python client package. Its version MUST match the running server exactly - a
# 0.9.16 client cannot talk to a 0.9.15 server. Defaults to the bundled PythonAPI, which
# supplies the pure-Python `agents` helpers only; install the client itself with
# `pip install carla==0.9.15`, matching your server.
# export CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"

# Route pool for the weighted selection: a space-separated list of directories, each
# searched recursively for *.xml. Defaults to the bundled SimLingo route sets
# (13,458 routes under simlingo/data/simlingo/). Override to use your own pool.
# export ROUTES_DIRS="$PWD/simlingo/data/simlingo/training_1_scenario/routes_training $PWD/simlingo/data/simlingo/training_3_scenarios/routes_training"

# Route set used by `collect_session.sh <name> long`. Defaults to the bundled long routes.
# export LONG_ROUTES_DIR="$PWD/simlingo/data/simlingo/long_routes"

# Single route used by `collect_session.sh <name> 0`, for smoke-testing the rig.
# export DEVTEST_XML="$LEADERBOARD_ROOT/data/routes_devtest.xml"

# Second dataset root. Set this when you record onto a fast local disk and keep the
# dataset somewhere larger: finished routes are rsynced there at the end of a session,
# and both roots are scanned for already-driven routes. Unset means everything stays
# in DATASET_ROOT.
# export ARCHIVE_ROOT="/mnt/archive/personadrive_dataset"

# Post-route step that counts LiDAR points inside each recorded bounding box. Defaults
# to compute_bbox_points.py next to the scripts. Empty string skips the step.
# export COMPUTE_SCRIPT="$PWD/compute_bbox_points.py"

# Driven-route cache. Derived data: delete it and it rebuilds on the next run. Keep it
# outside the checkout - it records dataset paths, which embed participant names.
# export CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/personadrive"

# Scratch output for run_scenario.sh (demos and screenshots, never the dataset).
# export SCRATCH_ROOT="$HOME/personadrive_scratch"

# CARLA server address, used by the evaluator and the route preview. Defaults to
# 127.0.0.1:2000. Do NOT use "localhost" under WSL2: it resolves to ::1 first, and a
# server on the Windows host listens only on IPv4, so the client times out even though
# the simulator is running.
# export CARLA_HOST=127.0.0.1
# export CARLA_PORT=2000

# Pygame needs a real display. Under WSL2 this is the WSLg X server, already correct
# by default; set them only if your setup differs.
# export DISPLAY=:0
# export SDL_VIDEODRIVER=x11

# Set to 0 to leave partially-driven routes out of the weighted selection instead of
# offering them again so their missing personas get filled in.
# export INCLUDE_PARTIAL=1

# ---------------------------------------------------------------------------
# Annotation stage (label_gen/, convert_for_hpc.py) - only for post-processing
# ---------------------------------------------------------------------------

# Where the VQA and commentary generators write, holding vqa/ and commentary/ subtrees.
# Keep it outside this checkout: label paths embed participant names.
# export LABELS_ROOT="$HOME/personadrive_labels"

# Destination root on the cluster, used to rewrite member paths inside the tars that
# convert_for_hpc.py writes.
# export HPC_ROOT="/path/on/hpc/database/simlingo"

# ---------------------------------------------------------------------------
# Diagnostics (leave unset for ordinary collection)
# ---------------------------------------------------------------------------

# Log every frame's raw and decoded wheel values to a CSV. One line per frame.
# export CARLA_WHEEL_TRACE=/tmp/wheel.csv

# Capture the pygame window (including the speed/command overlay) to PNGs.
# CARLA_SCREENSHOT_EVERY=0 means P-key-only; N saves every Nth frame.
# export CARLA_SCREENSHOT_DIR=/tmp/window_shots
# export CARLA_SCREENSHOT_EVERY=0
