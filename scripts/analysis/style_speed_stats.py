"""
Speed statistics of the style-instructed human driving data (paper, appendix
table "Speed statistics of the style-instructed human driving data").

Routes are matched across the three instructions: only routes (scenario/route id)
that exist under all of slow, default and fast are counted. Every frame of every
matched route contributes one speed sample.

    Mean        mean speed over all frames                      (mph)
    In motion   mean speed over frames with speed >= --stop-threshold
    Std. dev.   population standard deviation over all frames
    Stopped     share of frames with speed <  --stop-threshold  (%)

Input layout (datasets/README.md, style-dataset archive):
    ${PERSONADRIVE_DATA_ROOT}/simlingo/data/simlingo/<style>/driver*/routes/<Scenario>/<id>/<Town..>/measurements.json[.gz]

Usage:
    python scripts/analysis/style_speed_stats.py
    python scripts/analysis/style_speed_stats.py --data-root /path/to/database --all-routes --json out.json
"""

import argparse
import gzip
import json
import os
import statistics
from pathlib import Path

MPS_TO_MPH = 2.236936
STYLES = (("slow", "Conservative"), ("default", "Neutral"), ("fast", "Aggressive"))


def load_measurements(path: Path) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        doc = json.load(f)
    return doc if isinstance(doc, list) else [doc]


def find_routes(style_dir: Path) -> dict:
    """(scenario, route id) -> list of measurements files of that route."""
    routes = {}
    for m in sorted(style_dir.glob("driver*/routes/*/*/*/measurements.json*")):
        scenario, route_id = m.parts[-4], m.parts[-3]
        routes.setdefault((scenario, route_id), []).append(m)
    return routes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("PERSONADRIVE_DATA_ROOT", "database").split(":")[0],
                    help="dataset root containing simlingo/data/simlingo/<style>/ (default: $PERSONADRIVE_DATA_ROOT)")
    ap.add_argument("--stop-threshold", type=float, default=0.1,
                    help="speed in m/s below which a frame counts as stopped (default 0.1)")
    ap.add_argument("--all-routes", action="store_true",
                    help="use every route of each style instead of the route-matched subset")
    ap.add_argument("--json", default=None, help="optional path to write the table as JSON")
    args = ap.parse_args()

    base = Path(args.data_root) / "simlingo" / "data" / "simlingo"
    routes = {style: find_routes(base / style) for style, _ in STYLES}
    for style, _ in STYLES:
        if not routes[style]:
            raise SystemExit(f"no human routes under {base / style}/driver*/routes")
    matched = set.intersection(*(set(r) for r in routes.values()))
    print(f"routes per style: " + ", ".join(f"{s} {len(routes[s])}" for s, _ in STYLES)
          + f"; present in all three: {len(matched)}")

    rows = []
    for style, label in STYLES:
        speeds = []
        for key, files in routes[style].items():
            if not args.all_routes and key not in matched:
                continue
            for m in files:
                speeds += [float(fr["speed"]) for fr in load_measurements(m)]
        moving = [v for v in speeds if v >= args.stop_threshold]
        rows.append({
            "instruction": label, "profile": style, "frames": len(speeds),
            "mean_mph": statistics.mean(speeds) * MPS_TO_MPH,
            "in_motion_mph": statistics.mean(moving) * MPS_TO_MPH,
            "std_mph": statistics.pstdev(speeds) * MPS_TO_MPH,
            "stopped_pct": 100.0 * (1 - len(moving) / len(speeds)),
        })

    print(f"\n{'Instruction':<13}{'Frames':>8}{'Mean':>8}{'In motion':>11}{'Std. dev.':>11}{'Stopped (%)':>13}")
    for r in rows:
        print(f"{r['instruction']:<13}{r['frames']:>8}{r['mean_mph']:>8.1f}{r['in_motion_mph']:>11.1f}"
              f"{r['std_mph']:>11.1f}{r['stopped_pct']:>13.1f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"args": vars(args), "matched_routes": len(matched), "rows": rows}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
