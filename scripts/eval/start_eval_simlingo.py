# %%
import json
import os
import random
import subprocess
import time
import shutil

# `ujson` is faster but not in the personadrive env; fall back to stdlib json with the
# same `.load()` surface area we use elsewhere.
try:
    import ujson  # type: ignore
except ImportError:
    ujson = json  # type: ignore
from tqdm.autonotebook import tqdm

# %%
def get_num_jobs(job_name, username):
    len_usrn = len(username)
    num_running_jobs = int(
        subprocess.check_output(
            f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name} | wc -l",
            shell=True,
        ).decode('utf-8').replace('\n', ''))
    try:
        with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
            max_num_parallel_jobs = int(f.read())
    except:
        max_num_parallel_jobs = 1

    return num_running_jobs, max_num_parallel_jobs

# %%
def bash_file_bench2drive(job, port, tm_port):
    cfg = job["cfg"]
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    viz_path = job["viz_path"]
    result_file = job["result_file"]
    log_file = job["log_file"]
    err_file = job["err_file"]
    job_file = job["job_file"]

    # ICL profile env line - only emit if set (so old configs keep working)
    icl_env_lines = ""
    icl_profile = cfg.get("icl_profile")
    if icl_profile:
        icl_env_lines = f"export PERSONADRIVE_ICL_PROFILE={icl_profile}\n"
    if cfg.get("simlingo_icl_root"):
        icl_env_lines += f"export PERSONADRIVE_ICL_ROOT={cfg['simlingo_icl_root']}\n"
    if cfg.get("simlingo_icl_ckpt"):
        icl_env_lines += f"export PERSONADRIVE_ICL_CKPT={cfg['simlingo_icl_ckpt']}\n"
    # Number-of-in-context-points override. Only emit when explicitly set so
    # runs that rely on the training-config default keep working unchanged.
    num_icl = cfg.get("num_icl_examples")
    if num_icl is not None:
        icl_env_lines += f"export PERSONADRIVE_NUM_ICL={num_icl}\n"
    # Job-name tag so a K-sweep is distinguishable in `squeue` (e.g. _k3).
    icl_k_tag = "" if num_icl is None else f"_k{num_icl}"

    # Per-queue SLURM directives (partition / account / gres) come from the
    # queue this job was assigned to. The shared knobs (mem, tmp, cpus,
    # conda env, time) live in BASE_CFG and apply uniformly.
    queue = job["slurm_queue"]
    with open(job_file, 'w', encoding='utf-8') as rsh:
            rsh.write(f'''#!/bin/bash
#SBATCH --job-name={cfg["agent"]}_{icl_profile or "noicl"}{icl_k_tag}_{seed}_{cfg["benchmark"]}_{route_id}
#SBATCH --partition={queue["partition"]}
#SBATCH --account={queue["account"]}
#SBATCH -o {log_file}
#SBATCH -e {err_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cfg["slurm_cpus_per_task"]}
#SBATCH --mem={cfg["slurm_mem"]}
#SBATCH --gres={queue["gres"]}
#SBATCH --exclude={cfg["slurm_exclude_nodes"]}
#SBATCH --time=02:00:00

echo JOB ID $SLURM_JOB_ID

source ~/.bashrc
conda activate {cfg["conda_env"]}
cd {cfg["repo_root"]}


export CARLA_ROOT={cfg["carla_root"]}
# PYTHONPATH order matches scripts/eval/test_one_lb1_route.sh exactly:
#   repo_root and repo_root/team_code FIRST so the agent's
#   "from team_code.config_simlingo import GlobalConfig" resolves to OUR
#   team_code, not the cut-down team_code shipped under
#   Bench2Drive/leaderboard/. Otherwise the import dies with
#   ModuleNotFoundError: team_code.config_simlingo.
# Do not add carla-0.9.15-py3.7-linux-x86_64.egg to PYTHONPATH.
# The egg is Python-3.7-only and pulls libjpeg.so.8; the carla==0.9.15 pip
# wheel in the personadrive env supplies the right Py3.10 bindings via site-packages.
export PYTHONPATH={cfg["repo_root"]}:{cfg["repo_root"]}/team_code:{cfg["repo_root"]}/Bench2Drive/scenario_runner:{cfg["repo_root"]}/Bench2Drive/leaderboard:{cfg["carla_root"]}/PythonAPI/carla:${{PYTHONPATH:-}}
export SCENARIO_RUNNER_ROOT={cfg["repo_root"]}/Bench2Drive/scenario_runner

export SAVE_PATH={viz_path}
# Enable per-tick debug PNG dumps so the 4-profile viewer has frames to play.
# Saves go through the agent's background thread queue (see
# _BackgroundImageSaver in agent_simlingo.py) so they're effectively free in
# wall-clock terms.
export PERSONADRIVE_AGENT_DEBUG=1
{icl_env_lines}

python -u {cfg["repo_root"]}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py --routes={route} \
--repetitions=1 \
--track=SENSORS \
--checkpoint={result_file} \
--timeout=600 \
--agent={cfg["agent_file"]} \
--agent-config={cfg["checkpoint"]} \
--traffic-manager-seed={seed} \
--port={port} \
--traffic-manager-port={tm_port}
''')


# %%
def get_running_jobs():
    running_jobs = subprocess.check_output(f'squeue --me',shell=True).decode('utf-8').splitlines()
    running_jobs = set(x.strip().split(" ")[0] for x in running_jobs[1:])
    return running_jobs

# %%
def filter_completed(jobs):
    filtered_jobs = []

    running_jobs = get_running_jobs()
    for job in jobs:

        # If job is running we keep it in list (other function does killing)
        if "job_id" in job:
           if job["job_id"] in running_jobs:
              filtered_jobs.append(job)
              continue

        # Keep failed jobs to resubmit
        result_file = job["result_file"]
        if os.path.exists(result_file):
            try:
                with open(result_file, "r") as f:
                    evaluation_data = ujson.load(f)
            except:
                if job["tries"] > 0:
                    filtered_jobs.append(job)
                    continue


            progress = evaluation_data['_checkpoint']['progress']

            need_to_resubmit = False
            if len(progress) < 2 or progress[0] < progress[1]:
                need_to_resubmit = True
            else:
                for record in evaluation_data['_checkpoint']['records']:
                    if record['status'] == 'Failed - Agent couldn\'t be set up':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed - Simulation crashed':
                        need_to_resubmit = True
                    elif record['status'] == 'Failed - Agent crashed':
                        need_to_resubmit = True

            if need_to_resubmit and job["tries"] > 0:
                filtered_jobs.append(job)
        # Results file doesnt exist
        elif job["tries"] > 0:
            filtered_jobs.append(job)
    return filtered_jobs

# %%
def kill_dead_jobs(jobs):

    running_jobs = get_running_jobs()

    for job in jobs:

        if "job_id" in job:
            job_id = job["job_id"]

        elif os.path.exists(job["log_file"]):
            with open(job["log_file"], "r") as f:
                job_id = f.readline().strip().replace("JOB ID ", "")
        
        else:
            continue

        if job_id not in running_jobs:
            continue

        log = job["log_file"]
        if not os.path.exists(job["log_file"]):
            continue

        with open(log) as f:
            lines = f.readlines()
        if len(lines)==0:
            continue

        # TODO this needs to be improved, maybe also consider the err file?
        if any(["Watchdog exception" in line for line in lines]) or \
            "Engine crash handling finished; re-raising signal 11 for the default handler. Good bye.\n" in lines or \
            "[91mStopping the route, the agent has crashed:\n" in lines:

            subprocess.Popen(f"scancel {job_id}", shell=True)

# ONE base config - adjust paths/seeds to match your setup. Then PROFILES
# below fans this out into multiple configs, one per ICL profile, each
# writing to its own out_root sub-folder (normal/, fast/, slow/) so the
# three runs are easy to compare side-by-side in the viewer.
#
# Path defaults read from environment variables (see .env.example). Anything
# unset falls back to a path relative to PERSONADRIVE_ROOT (default: this checkout).
_REPO_ROOT_DEFAULT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _env(name, default):
    """Value of an environment variable; unset and empty (`NAME=` in .env) both give the default."""
    return os.environ.get(name) or default


REPO_ROOT  = _env("PERSONADRIVE_ROOT", _REPO_ROOT_DEFAULT)
CARLA_ROOT = _env("CARLA_ROOT",         f"{REPO_ROOT}/software/carla0915")
EVAL_OUT   = _env("PERSONADRIVE_EVAL_OUT_ROOT", f"{REPO_ROOT}/eval_results/single_route")
ROUTE_PATH = _env("PERSONADRIVE_ROUTE_PATH",    f"{REPO_ROOT}/leaderboard/data/bench2drive_split")
CKPT       = _env("PERSONADRIVE_CKPT",          f"{REPO_ROOT}/pretrained/personadrive_icl_sft/checkpoints/epoch=000.ckpt/pytorch_model.pt")
ICL_ROOT   = _env("PERSONADRIVE_ICL_ROOT",      f"{REPO_ROOT}/persona_drive_rag")
ICL_CKPT   = _env("PERSONADRIVE_ICL_CKPT",      f"{ICL_ROOT}/checkpoints/best_model.pt")
CONDA_ENV  = _env("PERSONADRIVE_CONDA_ENV",     "/path/to/conda/envs/personadrive")

# Number of in-context (ICL) points knob
# Controls how many retrieved in-context examples the agent prepends per tick.
# Propagated to the agent as the PERSONADRIVE_NUM_ICL env var (see bash template),
# which overrides cfg.data_module.base_dataset.num_icl_examples at eval time.
#   unset / ""  -> use the value baked into the training config (no override)
#   "0"         -> disable ICL entirely (K=0 baseline, no retrieval)
#   "1".."N"    -> retrieve exactly N in-context points
# Sweep K without editing the training config:
#   PERSONADRIVE_NUM_ICL=0 python scripts/eval/start_eval_simlingo.py
#   PERSONADRIVE_NUM_ICL=3 python scripts/eval/start_eval_simlingo.py
NUM_ICL = os.environ.get("PERSONADRIVE_NUM_ICL")
if NUM_ICL is not None:
    NUM_ICL = NUM_ICL.strip()
    if NUM_ICL == "":
        NUM_ICL = None
    else:
        try:
            _num_icl_int = int(NUM_ICL)
        except ValueError:
            raise ValueError(f"PERSONADRIVE_NUM_ICL must be an integer, got {NUM_ICL!r}")
        if _num_icl_int < 0:
            raise ValueError(f"PERSONADRIVE_NUM_ICL must be >= 0, got {_num_icl_int}")
        NUM_ICL = str(_num_icl_int)
# Filename / output-dir suffix so different-K sweeps don't clobber each other.
# Empty when NUM_ICL is unset -> preserves the historical layout exactly.
ICL_TAG = "" if NUM_ICL is None else f"_k{NUM_ICL}"
# Tag the whole output root (single_route_kN/) so an entire K-sweep is
# self-contained under one directory - every route/profile folder and the
# bookkeeping files live inside it. Empty tag -> historical layout unchanged.
EVAL_OUT = f"{EVAL_OUT}{ICL_TAG}"
print(f"Num in-context points (PERSONADRIVE_NUM_ICL): "
      f"{NUM_ICL if NUM_ICL is not None else 'config default'}")
print(f"Output root: {EVAL_OUT}")

BASE_CFG = {
    "agent": "simlingo",
    # Single-file .pt; for the new agent you can also point at the DeepSpeed
    # dir (last.ckpt/) and it will auto-consolidate.
    "checkpoint":  CKPT,
    "benchmark":   "bench2drive",
    "route_path":  ROUTE_PATH,
    "seeds":       [1],          # paper uses [1, 2, 3]; one is plenty for an ICL-profile A/B/C compare
    "tries":       3,            # was 2 - bumped after k54-08 ate two attempts of the same route
    "out_root":    EVAL_OUT,
    "carla_root":  CARLA_ROOT,
    "repo_root":   REPO_ROOT,
    "agent_file":  f"{REPO_ROOT}/team_code/agent_simlingo.py",
    "team_code":   "team_code",
    "agent_config":"not_used",
    "username":    os.environ.get("USER", "user"),
    # ICL retriever knobs - propagated into the SBATCH script as env vars.
    # Leave None to use the agent's hardcoded default in icl_retrieval.py.
    "simlingo_icl_root": ICL_ROOT,
    "simlingo_icl_ckpt": ICL_CKPT,
    # Number of in-context points to retrieve. None = use the training-config
    # default; "0" = ICL off (K=0); "N" = exactly N points. Propagated as the
    # PERSONADRIVE_NUM_ICL env var into the SBATCH script.
    "num_icl_examples": NUM_ICL,
    # Sampling knob
    # Set to an int (e.g. 10) to randomly sample down to that many routes
    # before fanning out across profiles + seeds. None = run every route in
    # route_path. Useful for a smoke run before committing the full sweep.
    "max_routes": None,
    # Queue restriction knob
    # Set to a partition name (e.g. "gpu" or "gpu32") to send every job to
    # ONLY that SLURM_QUEUES entry. None = use all queues round-robin per
    # the random route_split. Handy when one pool is flaky and you want to
    # ride the other one for a while without removing it from SLURM_QUEUES.
    "force_queue": None,
    # Shared SLURM knobs (apply to every queue below)
    "slurm_mem":           "32G",
    "slurm_cpus_per_task": 16,
    # Comma-separated nodelist that SLURM will refuse to schedule onto
    # (e.g. nodes with a broken CUDA driver). Empty = no exclusion.
    "slurm_exclude_nodes": os.environ.get("PERSONADRIVE_SLURM_EXCLUDE", ""),
    # Conda env activated inside the SBATCH script - must contain torch +
    # transformers + faiss + the bundled imageio_ffmpeg used by the viewer.
    "conda_env": CONDA_ENV,
}

# Two SLURM queues. Jobs are split round-robin across them so roughly half
# land on each GPU pool. Each queue equivalent to:
#   sbatch -p <partition> --account=<account> --gres=<gres>
# Add or remove queues here to change the split (e.g. drop one to send
# everything to the other; add a third to balance across three pools).
#
# Per-queue knob:
#   max_running  - cap on jobs from this pool that may be in `squeue --me`
#                  simultaneously. The submit loop assigns each route to
#                  whichever pool has the most free slots when it's about
#                  to be fired (dynamic / pull-based scheduling), so the
#                  long-run distribution naturally matches max_running.
SLURM_QUEUES = [
    {  # gpu32 / L40S pool
        "partition":   "gpu32",
        "account":     os.environ.get("PERSONADRIVE_SLURM_ACCOUNT_GPU32", "YOUR_SLURM_ACCOUNT_GPU32"),
        "gres":        "gpu:L40S:1",
        "max_running": 5,
    },
    {  # gpu / A100 pool
        "partition":   "gpu",
        "account":     os.environ.get("PERSONADRIVE_SLURM_ACCOUNT_GPU", "YOUR_SLURM_ACCOUNT_GPU"),
        "gres":        "gpu:A100:1",
        "max_running": 8,
    },
    {  # gpu / A30 pool (A30 shares partition `gpu` with A100/V100)
        "partition":   "gpu",
        "account":     os.environ.get("PERSONADRIVE_SLURM_ACCOUNT_GPU", "YOUR_SLURM_ACCOUNT_GPU"),
        "gres":        "gpu:A30:1",
        "max_running": 40,
    },
]

# Apply the BASE_CFG.force_queue restriction (if any) BEFORE the route-split
# block sees SLURM_QUEUES - this keeps the rest of the pipeline (job-build,
# progress.json) profile-aware without special-casing a "single-queue" mode.
_FORCE_QUEUE = BASE_CFG.get("force_queue")
if _FORCE_QUEUE:
    _filtered = [q for q in SLURM_QUEUES if q["partition"] == _FORCE_QUEUE]
    if not _filtered:
        raise ValueError(
            f"BASE_CFG.force_queue={_FORCE_QUEUE!r} matched no entries in "
            f"SLURM_QUEUES (available: {[q['partition'] for q in SLURM_QUEUES]})"
        )
    print(f"[force_queue] using only partition={_FORCE_QUEUE}  "
          f"(dropped {len(SLURM_QUEUES) - len(_filtered)} other queue(s))")
    SLURM_QUEUES = _filtered

# Which ICL profiles to sweep. Must match a directory name under
# <simlingo_icl_root>/faiss_index/<profile>/ (and the entry in PROFILE_NAMES
# inside icl_retrieval.py).
#
# Override at launch with the PERSONADRIVE_PROFILES env var (comma-separated):
#   PERSONADRIVE_PROFILES=slow            python scripts/eval/start_eval_simlingo.py
#   PERSONADRIVE_PROFILES=fast,slow       python scripts/eval/start_eval_simlingo.py
# The progress.json and route_queue_assignment.json filenames pick up a
# suffix derived from the active profile list (see RUN_TAG below) so a
# subset run never clobbers a parallel full-sweep run.
PROFILES_ALL = ["default", "normal", "fast", "slow"]
_profiles_env = os.environ.get("PERSONADRIVE_PROFILES")
if _profiles_env:
    PROFILES = [p.strip() for p in _profiles_env.split(",") if p.strip()]
    _unknown = [p for p in PROFILES if p not in PROFILES_ALL]
    if _unknown:
        raise ValueError(
            f"PERSONADRIVE_PROFILES contains unknown profile(s) {_unknown}; "
            f"valid options are {PROFILES_ALL}"
        )
else:
    # Edit this line to change the default profile set when launching
    # without PERSONADRIVE_PROFILES set.
    PROFILES = ["slow"]
print(f"Profiles this run: {PROFILES}")

# Filename tag - keeps progress.json / route_queue_assignment.json from
# different profile-subset runs out of each other's way. A full sweep gets
# the historical "all" suffix; a "slow"-only run writes _slow files.
RUN_TAG = os.environ.get("PERSONADRIVE_RUN_TAG") or (
    "all" if set(PROFILES) == set(PROFILES_ALL) else "_".join(PROFILES)
)


def fan_out_profiles(base, profiles):
    """Make one config per profile.

    Profile is now propagated through cfg["icl_profile"] but NOT into out_root
    - each (route, profile) gets its own parent dir under base.out_root in the
    single_route layout (see job-build loop).

    Output tree after the sweep finishes (matches scripts/eval/run_three_profiles.sh):
        <base.out_root>/<route_name>_<profile>/<STAMP>/
            ├── run/eval.sh
            ├── res/result.json
            ├── log/agent.log     (stdout + stderr merged)
            └── viz/RouteScenario_.../debug_viz/.../images
    """
    out = []
    for p in profiles:
        c = dict(base)
        c["icl_profile"] = p
        out.append(c)
    return out


configs = fan_out_profiles(BASE_CFG, PROFILES)


# Active route list.
#
# We persist *which* routes are in scope for this run (deterministic shuffle,
# optionally trimmed to max_routes) so re-launches replay the same set. We do
# NOT pre-bind routes to GPU pools - the submit loop assigns each route to
# whichever pool has free capacity at submit time. This keeps both pools
# saturated even when one drains faster than the other.
#
# File format ($BASE_CFG.out_root/route_list_<tag>.json):
#   {
#     "seed": 42,
#     "max_routes": <int|None>,
#     "force_queue": <str|None>,
#     "n_routes": <int>,
#     "routes": ["<basename>.xml", ...]   # in shuffle order
#   }
ROUTE_SPLIT_SEED = 42
ROUTE_LIST_FILE = os.path.join(
    BASE_CFG["out_root"], f"route_list_{RUN_TAG}.json"
)
os.makedirs(BASE_CFG["out_root"], exist_ok=True)

_all_routes_disk = sorted(x for x in os.listdir(BASE_CFG["route_path"]) if x.endswith(".xml"))
_max_routes      = BASE_CFG.get("max_routes")

# A saved list is only reusable when (seed, max_routes, force_queue) all
# match the current config. force_queue isn't strictly needed under dynamic
# scheduling but we keep it in the validation tuple so a single-pool replay
# vs a two-pool replay are clearly different runs.
_saved = None
if os.path.exists(ROUTE_LIST_FILE):
    with open(ROUTE_LIST_FILE) as _f:
        _candidate = json.load(_f)
    if (_candidate.get("seed") == ROUTE_SPLIT_SEED
            and _candidate.get("max_routes") == _max_routes
            and _candidate.get("force_queue") == _FORCE_QUEUE):
        _saved = _candidate

if _saved is not None:
    _active_route_order = list(_saved["routes"])
    print(f"Loaded active route list from {ROUTE_LIST_FILE}")
else:
    _rng = random.Random(ROUTE_SPLIT_SEED)
    _active_route_order = list(_all_routes_disk)
    _rng.shuffle(_active_route_order)
    if _max_routes is not None:
        _active_route_order = _active_route_order[: int(_max_routes)]
    with open(ROUTE_LIST_FILE, "w") as _f:
        json.dump({
            "seed":         ROUTE_SPLIT_SEED,
            "max_routes":   _max_routes,
            "force_queue":  _FORCE_QUEUE,
            "n_routes":     len(_active_route_order),
            "queues":       SLURM_QUEUES,
            "routes":       _active_route_order,
        }, _f, indent=2)
    print(f"Wrote active route list to {ROUTE_LIST_FILE}")

# `_active_routes` is the canonical set of route filenames that we'll actually
# run - every other entry on disk is skipped during job-build.
_active_routes = set(_active_route_order)

_total_disk = len(_all_routes_disk)
_total_active = len(_active_routes)
if _max_routes is not None and _total_active < _total_disk:
    print(f"Active routes: {_total_active} / {_total_disk} sampled "
          f"(max_routes={_max_routes}, seed={ROUTE_SPLIT_SEED})")
else:
    print(f"Active routes: {_total_active} (full sweep)")
print(f"Concurrency caps (dynamic assignment at submit time):")
for i, q in enumerate(SLURM_QUEUES):
    print(f"  -p {q['partition']:<8s}  --gres={q['gres']:<14s}  "
          f"max_running={q.get('max_running', '?')}")
_total_cap = sum(q.get("max_running", 0) for q in SLURM_QUEUES)
print(f"  → up to {_total_cap} concurrent jobs across all pools")
print()


# A single sweep-wide stamp shared across all jobs in this orchestrator
# invocation. Each (route, profile) parent gets ONE <STAMP>/ subdir per
# orchestrator run. Re-runs (after a process restart) get a new stamp ->
# preserves prior runs as siblings (the viewer auto-picks the latest).
SWEEP_STAMP = time.strftime("%Y%m%d_%H%M%S")
print(f"Sweep stamp: {SWEEP_STAMP}")
print()


# %%
job_queue = []
for cfg_idx, cfg in enumerate(configs):
    route_path = cfg["route_path"]
    routes = [x for x in os.listdir(route_path) if x[-4:]==".xml"] #########################################

    if cfg["benchmark"] == "bench2drive":
        fill_zeros = 3
    else:
        fill_zeros = 2

    profile = cfg["icl_profile"]   # 'default' / 'normal' / 'fast' / 'slow'

    for seed in cfg["seeds"]:
        seed = str(seed)

        for route in routes:
            # Honour max_routes - only run the subset chosen during the
            # route-split block above. Anything else on disk is skipped here.
            if route not in _active_routes:
                continue
            route_id   = route.split("_")[-1][:-4].zfill(fill_zeros)
            route_name = route[:-4]                          # strip ".xml"
            route      = os.path.join(route_path, route)

            # Single_route-style layout, matching scripts/eval/run_three_profiles.sh:
            #   <out_root>/<route_name>_<profile>[_seed<N>]/<STAMP>/{run,res,log,viz}
            stamp_dir = SWEEP_STAMP if seed == "1" else f"{SWEEP_STAMP}_seed{seed}"
            base_dir  = os.path.join(
                cfg["out_root"], f"{route_name}_{profile}", stamp_dir,
            )
            os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "log"), exist_ok=True)

            viz_path    = os.path.join(base_dir, "viz")
            os.makedirs(viz_path, exist_ok=True)

            result_file = os.path.join(base_dir, "res", "result.json")
            log_file    = os.path.join(base_dir, "log", "agent.log")
            err_file    = log_file                            # SBATCH -o & -e to same file -> merged
            job_file    = os.path.join(base_dir, "run", "eval.sh")

            # No pre-assigned queue - slurm_queue is set at submit time by
            # the dynamic scheduler in the main loop, so each route lands on
            # whichever pool happens to have free capacity right then.
            job = {
                "cfg": cfg,
                "route": route,
                "route_id": route_id,
                "seed": seed,
                "viz_path": viz_path,
                "result_file": result_file,
                "log_file": log_file,
                "err_file": err_file,
                "job_file": job_file,
                "tries": cfg["tries"],
            }

            job_queue.append(job)

# %%
carla_world_ports = set(range(10000, 20000, 50))
carla_streaming_ports = set(range(20000, 30000, 50))
carla_tm_ports = set(range(30000, 40000, 50))

# Resumable progress tracking.
#
# We keep a snapshot of every (profile, seed, route) triple in progress.json
# alongside route_queue_assignment.json under BASE_CFG["out_root"]. The file
# is rewritten atomically every loop iteration, so a crash never corrupts it.
#
# On restart, the orchestrator already skips successfully-finished routes
# (filter_completed reads each result_file). progress.json gives a one-glance
# rolled-up view: how many jobs are running / done / waiting / exhausted.
#
# Status values:
#   pending          - never submitted, in queue with tries left
#   running          - has job_id and shows up in `squeue --me`
#   succeeded        - result_file exists and shows complete + no failures
#   failed-resubmit  - result_file shows failure but tries > 0
#   exhausted        - failed and tries == 0 (orchestrator drops these)
#   stale            - was submitted but no longer in squeue and no result yet
PROGRESS_FILE = os.path.join(BASE_CFG["out_root"], f"progress_{RUN_TAG}.json")

# Snapshot of every job at startup - `_all_jobs` mirrors the initial queue and
# is used to compute progress.json even after entries are dropped from the
# live queue (because they succeeded).
_all_jobs = list(job_queue)


def _job_status(job, running_jobs_set):
    """Classify a single job into one of the status values above."""
    if "job_id" in job and job["job_id"] in running_jobs_set:
        return "running"
    rf = job["result_file"]
    if os.path.exists(rf):
        try:
            data = ujson.load(open(rf))
            ckpt = data.get('_checkpoint', {})
            prog = ckpt.get('progress', [])
            recs = ckpt.get('records',  [])
            failed = any(rec.get('status', '').startswith('Failed') for rec in recs)
            complete = len(prog) >= 2 and prog[0] >= prog[1]
            if complete and not failed:
                return "succeeded"
        except Exception:
            pass
        return "failed-resubmit" if job["tries"] > 0 else "exhausted"
    if "job_id" in job:
        return "stale" if job["tries"] <= 0 else "failed-resubmit"
    return "pending" if job["tries"] > 0 else "exhausted"


def _atomic_write_json(path, data):
    """Write JSON via a tmp file + rename - guarantees readers never see a
    half-written file even if the process gets killed mid-dump."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def dump_progress(all_jobs, running_jobs_set):
    """Walk every tracked job and write progress.json."""
    counts = {"pending": 0, "running": 0, "succeeded": 0,
              "failed-resubmit": 0, "exhausted": 0, "stale": 0}
    rows = []
    for j in all_jobs:
        st = _job_status(j, running_jobs_set)
        counts[st] = counts.get(st, 0) + 1
        # `slurm_queue` is only set after the job has been submitted - pending
        # jobs report queue=null in progress.json, which is also how a UI can
        # tell at a glance which routes haven't been dispatched yet.
        rows.append({
            "profile":      j["cfg"].get("icl_profile", "noicl"),
            "seed":         j["seed"],
            "route":        os.path.basename(j["route"]),
            "route_id":     j["route_id"],
            "queue":        j.get("slurm_queue"),
            "status":       st,
            "tries_left":   j["tries"],
            "job_id":       j.get("job_id"),
            "result_file":  j["result_file"],
            "log_file":     j["log_file"],
        })
    _atomic_write_json(PROGRESS_FILE, {
        "updated_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total":       len(rows),
        "counts":      counts,
        "queues":      SLURM_QUEUES,
        "profiles":    PROFILES,
        "jobs":        rows,
    })


# Initial dump so progress.json exists before the first loop iteration.
dump_progress(_all_jobs, get_running_jobs())
print(f"Progress tracker: {PROGRESS_FILE}")
print()


# %%
jobs = len(job_queue)
progress = tqdm(total = jobs)
print(f"Total jobs to dispatch: {jobs}  "
      f"(= {len(_active_routes)} routes × {len(PROFILES)} profiles × "
      f"{len(BASE_CFG['seeds'])} seeds)")
print(f"Routes are dispatched dynamically — each pool stays at max_running "
      f"until the active queue is drained.")
print()


def _pick_pool_with_headroom(running_per_queue):
    """Return the index of the pool with the largest absolute free-slot count
    (max_running − currently_running), or None if every pool is at its cap.

    Picking the pool with the most absolute headroom - rather than the most
    *fractional* - biases new submissions toward bigger pools when both have
    space, which matches the user's intent of keeping the larger pool busier
    overall (e.g. 8 A100 slots vs 5 L40S slots)."""
    best_idx = None
    best_free = 0
    for i, q in enumerate(SLURM_QUEUES):
        free = q.get("max_running", 1) - running_per_queue[i]
        if free > best_free:
            best_free = free
            best_idx = i
    return best_idx


while job_queue:
    kill_dead_jobs(job_queue)
    job_queue = filter_completed(job_queue)

    progress.update(jobs - len(job_queue) - progress.n)

    running_jobs = get_running_jobs()
    # Rewrite the rolled-up progress snapshot (atomic). Cheap - a few KB.
    dump_progress(_all_jobs, running_jobs)

    used_ports = set()
    # Per-pool running counts. As soon as a job finishes, the count drops
    # and the next loop tick is free to fire a replacement onto that pool.
    running_per_queue = {i: 0 for i in range(len(SLURM_QUEUES))}
    for job in job_queue:
        if "job_id" in job and job["job_id"] in running_jobs:
            used_ports.update(job["ports"])
            # We can only count this against a pool if it was already
            # dispatched (slurm_queue is set at submit time).
            if "slurm_queue" in job:
                for _qi, _q in enumerate(SLURM_QUEUES):
                    if _q is job["slurm_queue"]:
                        running_per_queue[_qi] += 1
                        break

    # If every pool is full, idle briefly and re-check.
    if _pick_pool_with_headroom(running_per_queue) is None:
        time.sleep(5)
        continue

    for job in job_queue:
        if job["tries"] <= 0:
            continue

        if "job_id" in job and job["job_id"] in running_jobs: # Das funktioniert nicht wenn man neu startet...
            continue

        if os.path.exists(job["log_file"]):
            with open(job["log_file"], "r") as f:
                job_id = f.readline().strip().replace("JOB ID ", "")
                if job_id in running_jobs:
                    print(f"{job['log_file']} already started.")
                    continue

        # Pick a pool with headroom NOW. If none, every pool is full - break
        # out, sleep, retry next tick.
        _qi = _pick_pool_with_headroom(running_per_queue)
        if _qi is None:
            break

        # Bind the queue to the job before writing the bash file (the
        # template reads partition / account / gres from job["slurm_queue"]).
        job["slurm_queue"] = SLURM_QUEUES[_qi]

        # Need to submit this job
        # Make bash file:
        carla_world_port_start = next(iter(carla_world_ports.difference(used_ports)))
        carla_streaming_port_start = next(iter(carla_streaming_ports.difference(used_ports)))
        carla_tm_port_start = next(iter(carla_tm_ports.difference(used_ports)))

        if job["cfg"]["benchmark"].lower() == "bench2drive":
            bash_file_bench2drive(job, carla_tm_port_start, carla_world_port_start)
            job["ports"] = {carla_world_port_start, carla_tm_port_start}
        else:
            raise NotImplementedError(f"Benchmark {job['cfg']['benchmark']} not implemented.")

        # submit
        # A retry must not inherit the previous attempt's log: kill_dead_jobs scans job["log_file"] for
        # crash markers, and SLURM truncates it only when the new job starts, so a queued retry would be
        # cancelled for the old attempt's crash. Keep the old log beside it as agent.log.attemptN.
        if os.path.exists(job["log_file"]):
            n_prev = 1
            while os.path.exists(f'{job["log_file"]}.attempt{n_prev}'):
                n_prev += 1
            os.rename(job["log_file"], f'{job["log_file"]}.attempt{n_prev}')
        shutil.rmtree(job["viz_path"])
        os.mkdir(job["viz_path"])
        job_id = subprocess.check_output(f'sbatch {job["job_file"]}', shell=True).decode('utf-8').strip().rsplit(' ', maxsplit=1)[-1]

        job["job_id"] = job_id
        job["tries"] -= 1

        # Reserve the ports and bump the per-queue counter so subsequent
        # iterations of THIS for-loop pick fresh ports and the next pool.
        used_ports.update(job["ports"])
        running_per_queue[_qi] += 1

        print(f'submit {job["job_file"]}  '
              f'[{SLURM_QUEUES[_qi]["partition"]} '
              f'{running_per_queue[_qi]}/{SLURM_QUEUES[_qi].get("max_running", 1)}]')
        print(len(job_queue))

        # Keep scanning - fire as many jobs as caps allow this tick. Stop
        # once every pool is at its cap so we don't pointlessly walk the rest
        # of the list waiting for SLURM to free a slot.
        if _pick_pool_with_headroom(running_per_queue) is None:
            break

    time.sleep(10)

# Final dump so progress.json reflects the terminal state of every job
# after the orchestrator exits (e.g. counts of succeeded vs exhausted).
dump_progress(_all_jobs, get_running_jobs())
print(f"Final progress: {PROGRESS_FILE}")