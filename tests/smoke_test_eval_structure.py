"""Smoke test for scripts/eval/start_eval_simlingo.py output structure + K plumbing.

Runs the REAL orchestrator end-to-end with sbatch/squeue mocked and a tiny
fake route set, then asserts:

  * output root is single_route_k<N>/  (tag moved to the root)
  * per-route dirs are <route>_<profile>/  (NO _kN suffix)
  * bookkeeping (route_list / progress) lives inside the tagged root
  * generated eval.sh contains `export PERSONADRIVE_NUM_ICL=<N>`
  * the SLURM job-name carries the _k<N> tag

No SLURM, no CARLA, no GPU. The mocked sbatch writes a "completed"
result.json so the orchestrator's submit loop drains and exits.
"""
import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts/eval/start_eval_simlingo.py"
K = "2"  # the value under test


def build_fake_routes(route_dir, n=2):
    route_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (route_dir / f"bench2drive_{i:02d}.xml").write_text("<routes/>\n")


def make_mock_check_output(real):
    def mock(cmd, *a, **kw):
        c = cmd if isinstance(cmd, str) else " ".join(cmd)
        if "squeue" in c:
            return b"JOBID PARTITION NAME\n"  # header only -> no running jobs
        if c.strip().startswith("sbatch "):
            job_file = c.split()[-1]
            base_dir = os.path.dirname(os.path.dirname(job_file))
            res = os.path.join(base_dir, "res", "result.json")
            os.makedirs(os.path.dirname(res), exist_ok=True)
            with open(res, "w") as f:
                json.dump({
                    "_checkpoint": {
                        "global_record": {"scores": {"score_composed": 100.0}},
                        "progress": [1, 1],
                        "records": [{"status": "Completed"}],
                    },
                    "values": ["100.0"], "labels": ["Score composed"],
                }, f)
            return b"Submitted batch job 99999"
        return b""
    return mock


def main():
    tmp = Path(tempfile.mkdtemp(prefix="smoke_eval_"))
    eval_out = tmp / "eval_results" / "single_route"
    route_dir = tmp / "routes"
    build_fake_routes(route_dir, n=2)

    os.environ.update({
        "PERSONADRIVE_EVAL_OUT_ROOT": str(eval_out),
        "PERSONADRIVE_ROUTE_PATH": str(route_dir),
        "PERSONADRIVE_NUM_ICL": K,
        "PERSONADRIVE_PROFILES": "slow",
        "PERSONADRIVE_ROOT": str(REPO_ROOT),
    })

    # Mock external side-effecting calls so nothing hits SLURM, and speed up.
    real_co = subprocess.check_output
    subprocess.check_output = make_mock_check_output(real_co)
    subprocess.Popen = lambda *a, **kw: type("P", (), {"wait": lambda self: 0})()
    time.sleep = lambda *a, **kw: None

    print(f"[smoke] tmp root: {tmp}")
    print(f"[smoke] running {SCRIPT.name} with PERSONADRIVE_NUM_ICL={K} …\n")
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        subprocess.check_output = real_co

    print("\n[smoke] ── assertions ──")
    failures = []

    def check(desc, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {desc}")
        if not cond:
            failures.append(desc)

    tagged_root = eval_out.parent / f"single_route_k{K}"
    untagged_root = eval_out

    check(f"tagged root exists: {tagged_root.name}/", tagged_root.is_dir())
    check("untagged root NOT created", not untagged_root.exists())

    # Per-route dirs: <route>_<profile> with NO _kN
    route_parents = sorted(p.name for p in tagged_root.iterdir() if p.is_dir())
    check(f"per-route dirs present ({route_parents})", len(route_parents) >= 1)
    check("per-route dirs have NO _k suffix",
          all(not n.endswith(f"_k{K}") for n in route_parents)
          and all(n.endswith("_slow") for n in route_parents))

    # Bookkeeping inside tagged root
    check("route_list_slow.json in tagged root",
          (tagged_root / "route_list_slow.json").exists())
    check("progress_slow.json in tagged root",
          (tagged_root / "progress_slow.json").exists())

    # eval.sh content
    eval_shs = list(tagged_root.glob("*/*/run/eval.sh"))
    check(f"eval.sh generated ({len(eval_shs)})", len(eval_shs) >= 1)
    if eval_shs:
        txt = eval_shs[0].read_text()
        check(f"eval.sh exports PERSONADRIVE_NUM_ICL={K}",
              f"export PERSONADRIVE_NUM_ICL={K}" in txt)
        check(f"job-name carries _k{K} tag",
              f"_k{K}_" in txt.split("--job-name=")[1].split("\n")[0])
        # result.json written by mock -> orchestrator saw it complete
        res = eval_shs[0].parent.parent / "res" / "result.json"
        check("result.json present for the route", res.exists())

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST PASSED — orchestrator emits single_route_k"
          f"{K}/ with correct structure and K plumbing.")
    print(f"[smoke] (inspect tree at {tagged_root})")


if __name__ == "__main__":
    main()
