"""
Aggregate Bench2Drive metrics per profile (default / normal / fast / slow)
across `eval_results/single_route/bench2drive_*` runs.

Per-route metrics follow the conventions already in the repo:
  * Driving Score (DS), Success Rate    -> Bench2Drive/tools/merge_route_json.py
  * Efficiency, Comfortness             -> Bench2Drive/tools/efficiency_smoothness_benchmark.py

Per profile we average across the routes that produced a usable result.json
(efficiency averages only over routes that actually emitted min_speed_infractions,
matching the reference benchmark).

Usage:
    python scripts/eval/aggregate_b2d_metrics.py
    python scripts/eval/aggregate_b2d_metrics.py --root /path/to/single_route --csv out.csv
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make the repo's reference comfort implementation importable.
# Don't resolve() - the project lives behind symlinks and resolving lands
# the path on a different filesystem where Bench2Drive/ doesn't appear.
REPO_ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Bench2Drive" / "tools"))
from efficiency_smoothness_benchmark import seg_compute_comfort_metric  # noqa: E402

PROFILES = ("default", "normal", "fast", "slow")
PROFILE_RE = re.compile(r"^bench2drive_\d+_(default|normal|fast|slow)$")
PERCENT_RE = re.compile(r"\b(\d+\.?\d*)%")


def best_run_with_result(route_dir: Path) -> tuple[Path, dict] | None:
    """Pick the timestamped subdir with the highest score_composed.

    Some bench2drive_*/<ts>/ directories are partial (no res/result.json) or
    contain a result.json with no records. We loop over every subdir, parse
    the route record, and keep the one with the largest DS. Ties are broken
    by the most recent timestamp.
    """
    best: tuple[Path, dict] | None = None
    best_key: tuple[float, str] | None = None
    for sub in route_dir.iterdir():
        if not sub.is_dir():
            continue
        rf = sub / "res" / "result.json"
        if not rf.is_file():
            continue
        try:
            doc = json.loads(rf.read_text())
        except Exception:
            continue
        records = doc.get("_checkpoint", {}).get("records", [])
        if not records:
            continue
        parsed = parse_record(records[0])
        ds = parsed["ds"]
        if ds is None:
            continue
        key = (ds, sub.name)
        if best_key is None or key > best_key:
            best_key = key
            best = (sub, parsed)
    return best


def find_metric_info(run_dir: Path) -> Path | None:
    """Locate viz*/debug_viz/.../metric/metric_info.json under a run dir."""
    matches = list(run_dir.glob("viz*/debug_viz/*/iter_/_*/metric/metric_info.json"))
    if not matches:
        # Fall back to any depth in case the layout drifts.
        matches = list(run_dir.rglob("metric_info.json"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def parse_record(rec: dict) -> dict:
    """Extract per-route DS, success flag, and min-speed efficiency value."""
    scores = rec.get("scores", {}) or {}
    ds = scores.get("score_composed")
    rc = scores.get("score_route")
    penalty = scores.get("score_penalty")

    status = rec.get("status", "")
    infractions = rec.get("infractions", {}) or {}
    success = status in ("Completed", "Perfect")
    if success:
        for k, v in infractions.items():
            if k == "min_speed_infractions":
                continue
            if v:
                success = False
                break

    collisions = (
        len(infractions.get("collisions_layout") or [])
        + len(infractions.get("collisions_pedestrian") or [])
        + len(infractions.get("collisions_vehicle") or [])
    )

    eff = None
    msi = infractions.get("min_speed_infractions") or []
    if msi:
        vals = []
        for s in msi:
            m = PERCENT_RE.search(s)
            if not m:
                continue
            x = float(m.group(1))
            if x > 1000:  # matches the reference benchmark filter
                continue
            vals.append(x)
        if vals:
            eff = float(np.mean(vals))

    return {
        "ds": float(ds) if ds is not None else None,
        "rc": float(rc) if rc is not None else None,
        "penalty": float(penalty) if penalty is not None else None,
        "success": bool(success) and ds is not None,
        "efficiency": eff,
        "collisions": int(collisions),
        "status": status,
    }


def kinematics_from_metric_info(path: Path) -> dict:
    """Read metric_info.json once and derive comfort + lon-acc + speed.

    Returns:
        comfort   - fraction of 2-second windows that pass the NuPlan-style
                    bounds (same as before; None if too few steps).
        lon_accel - mean |a · forward_vector| in m/s² over the whole route
                    (longitudinal accel magnitude).
        speed     - mean |Δlocation|/Δt over the whole route, in m/s. Used
                    only as a fallback when telemetry.csv is missing.
    """
    out = {"comfort": None, "lon_accel": None, "speed": None}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return out
    fields = ("acceleration", "angular_velocity", "forward_vector",
              "right_vector", "location", "rotation")
    buckets = {f: [] for f in fields}
    for k in sorted(data.keys(), key=lambda s: int(s) if str(s).isdigit() else s):
        v = data[k]
        for f in fields:
            buckets[f].append(v[f])
    arrs = {f: np.array(buckets[f]) for f in fields}
    n = len(arrs["acceleration"])
    if n < 2:
        return out

    # Longitudinal accel magnitude in the xy plane (z is gravity-dominated).
    acc2d = arrs["acceleration"][:, :2]
    fwd2d = arrs["forward_vector"][:, :2]
    lon = np.einsum("ij,ij->i", acc2d, fwd2d)
    out["lon_accel"] = float(np.mean(np.abs(lon)))

    # Fallback speed from |Δlocation|/Δt. CARLA ticks at 20 Hz here.
    locs = arrs["location"][:, :2]
    if len(locs) >= 2:
        dxy = np.linalg.norm(np.diff(locs, axis=0), axis=1)
        out["speed"] = float(np.mean(dxy) * 20.0)

    if n >= 8:
        try:
            c = seg_compute_comfort_metric(**arrs)
            out["comfort"] = float(c) if isinstance(c, float) else float(c)
        except Exception:
            pass
    return out


def find_telemetry(run_dir: Path) -> Path | None:
    matches = list(run_dir.glob("viz*/debug_viz/*/iter_/_*/metric/telemetry.csv"))
    if not matches:
        matches = list(run_dir.rglob("telemetry.csv"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def mean_speed_from_telemetry(path: Path) -> float | None:
    """Mean of telemetry.csv `speed_mps`, after the agent's startup brake.

    The agent forces brake=1.0 for `inital_frames_delay` ticks at the start
    of every route - those ticks have speed_mps=0 and would skew the mean
    downward, so trim the leading run of (brake>=0.99 AND speed<0.05).
    """
    try:
        with open(path) as f:
            header = f.readline().rstrip().split(",")
            try:
                idx_speed = header.index("speed_mps")
                idx_brake = header.index("brake")
            except ValueError:
                return None
            rows = []
            for line in f:
                parts = line.rstrip().split(",")
                if len(parts) <= max(idx_speed, idx_brake):
                    continue
                try:
                    rows.append((float(parts[idx_speed]), float(parts[idx_brake])))
                except ValueError:
                    continue
    except Exception:
        return None
    if not rows:
        return None
    start = 0
    for i, (s, b) in enumerate(rows):
        if not (b >= 0.99 and s < 0.05):
            start = i
            break
    speeds = [s for s, _ in rows[start:]]
    if not speeds:
        return None
    return float(np.mean(speeds))


def collect(root: Path) -> dict:
    """Return {profile: [per-route dict, ...]} for every bench2drive_* folder."""
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for entry in sorted(root.iterdir()):
        m = PROFILE_RE.match(entry.name)
        if not m:
            continue
        profile = m.group(1)
        picked = best_run_with_result(entry)
        if picked is None:
            by_profile[profile].append({"route": entry.name, "missing": True})
            continue
        run_dir, parsed = picked
        parsed["route"] = entry.name
        parsed["run_dir"] = str(run_dir)

        metric_path = find_metric_info(run_dir)
        if metric_path:
            kin = kinematics_from_metric_info(metric_path)
            parsed["comfort"] = kin["comfort"]
            parsed["lon_accel"] = kin["lon_accel"]
            speed_fallback = kin["speed"]
        else:
            parsed["comfort"] = None
            parsed["lon_accel"] = None
            speed_fallback = None

        tel_path = find_telemetry(run_dir)
        speed = mean_speed_from_telemetry(tel_path) if tel_path else None
        parsed["speed"] = speed if speed is not None else speed_fallback
        by_profile[profile].append(parsed)
    return by_profile


def summarize(per_profile: dict) -> list[dict]:
    rows = []
    for profile in PROFILES:
        routes = per_profile.get(profile, [])
        valid = [r for r in routes if not r.get("missing") and r.get("ds") is not None]
        n_total = len(routes)
        n_valid = len(valid)
        if not valid:
            rows.append({"profile": profile, "n_total": n_total, "n_valid": 0})
            continue

        ds_vals = [r["ds"] for r in valid]
        rc_vals = [r["rc"] for r in valid if r["rc"] is not None]
        eff_vals = [r["efficiency"] for r in valid if r["efficiency"] is not None]
        comf_vals = [r["comfort"] for r in valid if r["comfort"] is not None]
        speed_vals = [r["speed"] for r in valid if r.get("speed") is not None]
        lon_vals = [r["lon_accel"] for r in valid if r.get("lon_accel") is not None]
        coll_vals = [r["collisions"] for r in valid if r.get("collisions") is not None]
        n_success = sum(1 for r in valid if r["success"])
        n_routes_with_collision = sum(1 for r in valid if (r.get("collisions") or 0) > 0)

        rows.append({
            "profile": profile,
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
        })
    return rows


def print_table(rows: list[dict]) -> None:
    header = (f"{'profile':>10}  {'used/total':>10}  "
              f"{'DS':>8}  {'RC':>8}  {'SR%':>8}  "
              f"{'Eff(n)':>14}  {'Comf(n)':>14}  "
              f"{'Speed(n)':>14}  {'LonAcc(n)':>14}  "
              f"{'Coll(routes)':>14}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if "DS" not in r:
            print(f"{r['profile']:>10}  {0:>4}/{r['n_total']:<5}  (no valid runs)")
            continue

        def fmt(val, n, prec):
            if val != val:
                return "nan(0)"
            return f"{val:.{prec}f}({n})"

        coll_cell = f"{r['Collisions_total']}({r['Routes_with_collision']}r)"
        print(f"{r['profile']:>10}  "
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
    ap.add_argument("--root", default=str(REPO_ROOT / "eval_results" / "single_route"))
    ap.add_argument("--csv", default=None, help="optional path to dump rows as CSV")
    ap.add_argument("--per-route-csv", default=None,
                    help="optional path to dump per-route values as CSV")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"missing root dir: {root}")

    per_profile = collect(root)
    rows = summarize(per_profile)
    print_table(rows)

    if args.csv:
        import csv
        keys = ["profile", "n_total", "n_valid", "DS", "RouteCompletion",
                "SuccessRate%", "Efficiency", "Efficiency_n",
                "Comfortness", "Comfortness_n",
                "Speed_mps", "Speed_n", "LonAccel_mps2", "LonAccel_n",
                "Collisions_total", "Collisions_per_route", "Routes_with_collision"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"wrote {args.csv}")

    if args.per_route_csv:
        import csv
        with open(args.per_route_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["profile", "route", "status", "ds", "route_completion",
                        "penalty", "success", "efficiency", "comfort",
                        "speed_mps", "lon_accel_mps2", "collisions", "run_dir"])
            for profile, routes in per_profile.items():
                for r in routes:
                    if r.get("missing"):
                        w.writerow([profile, r.get("route", ""), "MISSING",
                                    "", "", "", "", "", "", "", "", "", ""])
                        continue
                    w.writerow([profile, r["route"], r["status"], r["ds"], r["rc"],
                                r["penalty"], int(r["success"]),
                                r["efficiency"] if r["efficiency"] is not None else "",
                                r["comfort"] if r["comfort"] is not None else "",
                                r["speed"] if r.get("speed") is not None else "",
                                r["lon_accel"] if r.get("lon_accel") is not None else "",
                                r.get("collisions", 0),
                                r["run_dir"]])
        print(f"wrote {args.per_route_csv}")


if __name__ == "__main__":
    main()
