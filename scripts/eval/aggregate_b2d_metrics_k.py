"""
Aggregate Bench2Drive metrics per context-points value (k=1,2,3,4,...).

Auto-discovers every `eval_results/single_route_k<N>` directory and reports
one row per k. Within each k, aggregates across every `bench2drive_XX_*`
route folder (regardless of profile suffix).

Reuses the per-route metric helpers from scripts/eval/aggregate_b2d_metrics.py.

Usage:
    python scripts/eval/aggregate_b2d_metrics_k.py
    python scripts/eval/aggregate_b2d_metrics_k.py --csv per_k.csv --per-route-csv per_k_routes.csv
    python scripts/eval/aggregate_b2d_metrics_k.py --k 1 2 3        # restrict to specific k
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
from aggregate_b2d_metrics import (  # noqa: E402
    PROFILE_RE,
    best_run_with_result,
    find_metric_info,
    find_telemetry,
    kinematics_from_metric_info,
    mean_speed_from_telemetry,
)

REPO_ROOT = HERE.parents[1]
K_DIR_RE = re.compile(r"^single_route_k(\d+)$")
ROUTE_RE = re.compile(r"^bench2drive_\d+_[a-zA-Z]+$")


def collect_for_root(root: Path) -> list[dict]:
    """Pick best run for every bench2drive_* under `root`, compute metrics."""
    out = []
    for entry in sorted(root.iterdir()):
        if not ROUTE_RE.match(entry.name):
            continue
        m = PROFILE_RE.match(entry.name)
        profile = m.group(1) if m else "unknown"
        picked = best_run_with_result(entry)
        if picked is None:
            out.append({"route": entry.name, "profile": profile, "missing": True})
            continue
        run_dir, parsed = picked
        parsed["route"] = entry.name
        parsed["profile"] = profile
        parsed["run_dir"] = str(run_dir)

        metric_path = find_metric_info(run_dir)
        if metric_path:
            kin = kinematics_from_metric_info(metric_path)
            parsed["comfort"] = kin["comfort"]
            parsed["lon_accel"] = kin["lon_accel"]
            speed_fb = kin["speed"]
        else:
            parsed["comfort"] = None
            parsed["lon_accel"] = None
            speed_fb = None

        tel_path = find_telemetry(run_dir)
        speed = mean_speed_from_telemetry(tel_path) if tel_path else None
        parsed["speed"] = speed if speed is not None else speed_fb
        out.append(parsed)
    return out


def summarize_routes(routes: list[dict], label: str) -> dict:
    valid = [r for r in routes if not r.get("missing") and r.get("ds") is not None]
    n_total = len(routes)
    n_valid = len(valid)
    if not valid:
        return {"label": label, "n_total": n_total, "n_valid": 0}

    ds_vals = [r["ds"] for r in valid]
    rc_vals = [r["rc"] for r in valid if r["rc"] is not None]
    eff_vals = [r["efficiency"] for r in valid if r["efficiency"] is not None]
    comf_vals = [r["comfort"] for r in valid if r["comfort"] is not None]
    speed_vals = [r["speed"] for r in valid if r.get("speed") is not None]
    lon_vals = [r["lon_accel"] for r in valid if r.get("lon_accel") is not None]
    coll_vals = [r["collisions"] for r in valid if r.get("collisions") is not None]
    n_success = sum(1 for r in valid if r["success"])
    n_routes_with_collision = sum(1 for r in valid if (r.get("collisions") or 0) > 0)

    return {
        "label": label,
        "n_total": n_total,
        "n_valid": n_valid,
        "DS": float(np.mean(ds_vals)),
        "RouteCompletion": float(np.mean(rc_vals)) if rc_vals else float("nan"),
        "SuccessRate%": 100.0 * n_success / n_valid,
        "Efficiency": float(np.mean(eff_vals)) if eff_vals else float("nan"),
        "Efficiency_n": len(eff_vals),
        "Comfortness": float(np.mean(comf_vals)) if comf_vals else float("nan"),
        "Comfortness_n": len(comf_vals),
        "Speed_mps": float(np.mean(speed_vals)) if speed_vals else float("nan"),
        "Speed_n": len(speed_vals),
        "LonAccel_mps2": float(np.mean(lon_vals)) if lon_vals else float("nan"),
        "LonAccel_n": len(lon_vals),
        "Collisions_total": int(sum(coll_vals)),
        "Collisions_per_route": float(np.mean(coll_vals)) if coll_vals else float("nan"),
        "Routes_with_collision": n_routes_with_collision,
    }


def discover_k_dirs(base: Path, want: list[int] | None) -> list[tuple[int, Path]]:
    found = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        m = K_DIR_RE.match(d.name)
        if not m:
            continue
        k = int(m.group(1))
        if want is not None and k not in want:
            continue
        found.append((k, d))
    found.sort(key=lambda kv: kv[0])
    return found


def print_table(rows: list[dict], label_header: str = "k") -> None:
    header = (f"{label_header:>10}  {'used/total':>10}  "
              f"{'DS':>8}  {'RC':>8}  {'SR%':>8}  "
              f"{'Eff(n)':>14}  {'Comf(n)':>14}  "
              f"{'Speed(n)':>14}  {'LonAcc(n)':>14}  "
              f"{'Coll(routes)':>14}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if "DS" not in r:
            print(f"{r['label']:>10}  {0:>4}/{r['n_total']:<5}  (no valid runs)")
            continue

        def fmt(val, n, prec):
            if val != val:
                return "nan(0)"
            return f"{val:.{prec}f}({n})"

        coll_cell = f"{r['Collisions_total']}({r['Routes_with_collision']}r)"
        print(f"{r['label']:>10}  "
              f"{r['n_valid']:>4}/{r['n_total']:<5}  "
              f"{r['DS']:>8.2f}  "
              f"{r['RouteCompletion']:>8.2f}  "
              f"{r['SuccessRate%']:>8.2f}  "
              f"{fmt(r['Efficiency'], r['Efficiency_n'], 2):>14}  "
              f"{fmt(r['Comfortness'], r['Comfortness_n'], 4):>14}  "
              f"{fmt(r['Speed_mps'], r['Speed_n'], 2):>14}  "
              f"{fmt(r['LonAccel_mps2'], r['LonAccel_n'], 3):>14}  "
              f"{coll_cell:>14}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO_ROOT / "eval_results"),
                    help="parent dir holding the single_route_k* dirs")
    ap.add_argument("--k", type=int, nargs="*", default=None,
                    help="restrict to specific k values (default: all found)")
    ap.add_argument("--csv", default=None, help="per-k summary CSV")
    ap.add_argument("--per-route-csv", default=None, help="per-route CSV")
    ap.add_argument("--by-profile", action="store_true",
                    help="also break down each k by profile (default/normal/fast/slow)")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.is_dir():
        sys.exit(f"missing base dir: {base}")

    k_dirs = discover_k_dirs(base, args.k)
    if not k_dirs:
        sys.exit(f"no single_route_k<N> directories found under {base}")

    all_routes: list[dict] = []      # for per-route CSV
    summary_rows: list[dict] = []    # for stdout table + summary CSV

    for k, root in k_dirs:
        routes = collect_for_root(root)
        for r in routes:
            r["k"] = k
        all_routes.extend(routes)

        row = summarize_routes(routes, label=f"k{k}")
        row["k"] = k
        summary_rows.append(row)

        if args.by_profile:
            by_prof: dict[str, list[dict]] = defaultdict(list)
            for r in routes:
                by_prof[r["profile"]].append(r)
            for prof in sorted(by_prof):
                sub = summarize_routes(by_prof[prof], label=f"k{k}/{prof}")
                sub["k"] = k
                sub["profile"] = prof
                summary_rows.append(sub)

    print_table(summary_rows, label_header="k/profile")

    if args.csv:
        import csv
        keys = ["k", "label", "n_total", "n_valid",
                "DS", "RouteCompletion", "SuccessRate%",
                "Efficiency", "Efficiency_n",
                "Comfortness", "Comfortness_n",
                "Speed_mps", "Speed_n",
                "LonAccel_mps2", "LonAccel_n",
                "Collisions_total", "Collisions_per_route",
                "Routes_with_collision"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow({k_: r.get(k_, "") for k_ in keys})
        print(f"wrote {args.csv}")

    if args.per_route_csv:
        import csv
        with open(args.per_route_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "profile", "route", "status", "ds", "route_completion",
                        "penalty", "success", "efficiency", "comfort",
                        "speed_mps", "lon_accel_mps2", "collisions", "run_dir"])
            for r in all_routes:
                if r.get("missing"):
                    w.writerow([r.get("k", ""), r.get("profile", ""),
                                r.get("route", ""), "MISSING",
                                "", "", "", "", "", "", "", "", "", ""])
                    continue
                w.writerow([r["k"], r["profile"], r["route"], r["status"],
                            r["ds"], r["rc"], r["penalty"], int(r["success"]),
                            r["efficiency"] if r["efficiency"] is not None else "",
                            r["comfort"] if r["comfort"] is not None else "",
                            r["speed"] if r.get("speed") is not None else "",
                            r["lon_accel"] if r.get("lon_accel") is not None else "",
                            r.get("collisions", 0),
                            r["run_dir"]])
        print(f"wrote {args.per_route_csv}")


if __name__ == "__main__":
    main()
