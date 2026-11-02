#!/usr/bin/env python3
"""
rag_data_processor_v2.py - Process simlingo routes (aggregated OR per-frame).

Backward-compatible with the original rag_data_processor.py per-frame format,
and adds support for the new aggregated single-file layout. Format is
auto-detected per route, so a profile can mix both layouts.

Per-route layouts:
  Aggregated: <route>/measurements.json  +  <drivelm>/<route>/vqa.json  +  <comm>/<route>/commentary.json
  Per-frame : <route>/measurements/0000.json[.gz]  + <drivelm>/<route>/vqa/0000.json + ...

Usage:
  python rag_data_processor_v2.py \\
      --data-root  /share/.../database/simlingo \\
      --profile    fast \\
      --frame-step 5 \\
      --output-dir processed_routes/fast
"""

import argparse
import ast
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Reuse helpers from the original module so behaviour stays identical
# wherever the formats overlap.
from rag_data_processor import (
    _load_file,
    _resolve_path,
    get_command_string,
    extract_future_waypoints,
    extract_temporal_waypoints,
    extract_geometric_waypoints,
    load_measurement,
    _build_initial_cache,
    _advance_cache,
    _get_contiguous_window,
)


# Layout detection
def detect_route_layout(route_data_dir: str) -> str | None:
    """Return 'aggregated', 'per_frame', or None for a route data directory."""
    if os.path.exists(os.path.join(route_data_dir, "measurements.json")) or \
       os.path.exists(os.path.join(route_data_dir, "measurements.json.gz")):
        return "aggregated"
    if os.path.isdir(os.path.join(route_data_dir, "measurements")):
        return "per_frame"
    return None


def find_routes(profile_data_dir: str) -> list[str]:
    """Walk the profile data subtree and return every directory that holds a route.

    A route directory contains an ``rgb/`` subdir AND either ``measurements.json``
    (aggregated) or ``measurements/`` (per-frame). Walking stops at each route
    so we don't descend into ``rgb/`` etc.
    """
    routes: list[str] = []
    for dirpath, dirnames, filenames in os.walk(profile_data_dir):
        is_route = "rgb" in dirnames and (
            "measurements.json"    in filenames
            or "measurements.json.gz" in filenames
            or "measurements"      in dirnames
        )
        if is_route:
            routes.append(dirpath)
            dirnames[:] = []  # don't descend further; rgb/ etc. are not routes
    return sorted(routes)


# Aggregated readers / parsers
def _parse_qa_field(qa):
    """Aggregated vqa items store ``QA`` as a stringified Python dict.

    Falls back to {} on parse errors. ast.literal_eval is safe for python literals.
    """
    if isinstance(qa, dict):
        return qa
    if isinstance(qa, str):
        try:
            parsed = ast.literal_eval(qa)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _load_aggregated_or_empty(path: str):
    """_load_file but returns [] when the file is missing rather than raising."""
    p = _resolve_path(path) or _resolve_path(path[:-3]) if path.endswith(".gz") else _resolve_path(path)
    if p is None:
        return []
    try:
        return _load_file(p)
    except Exception as e:
        print(f"  [WARN] failed to load {p}: {e}")
        return []


# Record builder (shared between layouts)
def _build_record(
    frame_number: int,
    route_data_dir: str,
    scenario: str,
    run: str,
    profile: str,
    driver: str,
    measurements_now: dict,
    future_window: list,
    vqa_qa: dict,
    commentary_text: str,
    future_horizon: int,
) -> dict | None:
    """Build a single processed record from already-loaded frame data.

    Returns None if temporal waypoints can't be computed (caller should stop).
    """
    future_wp_data = extract_future_waypoints(future_window, future_horizon)
    temporal_wp_data = extract_temporal_waypoints(future_window)
    if temporal_wp_data is None:
        return None

    geo_wp_data = extract_geometric_waypoints(measurements_now, num_route_points=20)

    # ── Ego state sentence section ──
    speed    = measurements_now.get("speed",    "N/A")
    throttle = measurements_now.get("throttle", "N/A")
    steer    = measurements_now.get("steer",    "N/A")
    aim_wp   = measurements_now.get("aim_wp",   "N/A")

    speed_str    = f"{speed:.4f}"    if isinstance(speed,    (int, float)) else str(speed)
    throttle_str = f"{throttle:.4f}" if isinstance(throttle, (int, float)) else str(throttle)
    steer_str    = f"{steer:.4f}"    if isinstance(steer,    (int, float)) else str(steer)
    aim_wp_str   = (
        f"[{aim_wp[0]:.4f}, {aim_wp[1]:.4f}]"
        if isinstance(aim_wp, list) and len(aim_wp) == 2
        else str(aim_wp)
    )

    ego_state = (
        f"--- EGO VEHICLE STATE ---\n"
        f"Speed: {speed_str}\n"
        f"Throttle: {throttle_str}\n"
        f"Brake: {'Brake is clicked' if measurements_now.get('brake') else 'Brake is not clicked'}\n"
        f"Steer: {steer_str}\n"
        f"Current Command: {get_command_string(measurements_now.get('command', 'N/A'))}\n"
        f"Next Aim Waypoint: {aim_wp_str}\n"
    )

    # ── Q&A section ──
    qa_parts: list[str] = []
    for section in ("perception", "prediction", "planning"):
        qa_parts.append(f"--- {section.upper()} ---\n")
        items = vqa_qa.get(section, []) if isinstance(vqa_qa, dict) else []
        if not items:
            qa_parts.append(f"No data found for {section}.\n")
        else:
            for item in items:
                if isinstance(item, dict):
                    qa_parts.append(f"Q: {item.get('Q', 'N/A')}\nA: {item.get('A', 'N/A')}\n\n")

    # ── Commentary section ──
    commentary_section = f"--- COMMENTARY ---\n{commentary_text or 'No commentary found.'}\n"

    # ── Future waypoints section ──
    wp_str = ", ".join(
        f"[{x:.4f}, {y:.4f}]" for x, y in future_wp_data["future_waypoints"]
    )
    future_wp_section = f"--- FUTURE WAYPOINTS ---\nFuture Waypoints: ({wp_str})\n"

    full_sentence = ego_state + "".join(qa_parts) + commentary_section + future_wp_section

    frame_id   = f"{frame_number:04d}"
    frame_path = os.path.join(route_data_dir, "rgb", f"{frame_id}.jpg")

    return {
        "frame_path":             frame_path,
        "sentence":               full_sentence,
        "scenario":               scenario,
        "profile":                profile,
        "driver":                 driver,
        "run":                    f"{run}/{frame_id}",
        "future_waypoints":       future_wp_data["future_waypoints"].tolist(),
        "temporal_waypoints_org": temporal_wp_data["waypoints_org"].tolist(),
        "route_adjusted":         geo_wp_data["route_adjusted"].tolist(),
        "target_point":           measurements_now.get("target_point"),
        "target_point_next":      measurements_now.get("target_point_next"),
        "command":                measurements_now.get("command"),
        "next_command":           measurements_now.get("next_command"),
    }


# Aggregated-layout per-route processor
def process_route_aggregated(
    route_data_dir: str,
    route_drivelm_dir: str,
    route_commentary_dir: str,
    scenario: str,
    run: str,
    profile: str,
    driver: str,
    start_frame: int = 10,
    future_horizon: int = 10,
    frame_step: int = 5,
) -> list[dict]:
    route_name = os.path.basename(route_data_dir)

    meas_list = _load_aggregated_or_empty(os.path.join(route_data_dir,       "measurements.json"))
    vqa_list  = _load_aggregated_or_empty(os.path.join(route_drivelm_dir,    "vqa.json"))
    comm_list = _load_aggregated_or_empty(os.path.join(route_commentary_dir, "commentary.json"))

    if not isinstance(meas_list, list) or not meas_list:
        return []
    n_max = min(
        len(meas_list),
        len(vqa_list) if isinstance(vqa_list, list) else 0,
        len(comm_list) if isinstance(comm_list, list) else 0,
    )
    if n_max == 0:
        return []

    records: list[dict] = []
    fn = start_frame
    while fn + future_horizon < n_max:
        try:
            window = meas_list[fn : fn + future_horizon + 1]
            if len(window) < future_horizon + 1:
                break

            vqa_qa = _parse_qa_field(vqa_list[fn].get("QA", {}) if isinstance(vqa_list[fn], dict) else {})
            comm_item = comm_list[fn]
            commentary_text = comm_item.get("commentary", "") if isinstance(comm_item, dict) else ""

            rec = _build_record(
                fn, route_data_dir, scenario, run, profile, driver,
                window[0], window, vqa_qa, commentary_text, future_horizon,
            )
            if rec is None:
                break
            records.append(rec)
        except Exception as e:
            print(f"[{route_name}] aggregated frame {fn:04d}: {e}")
            break
        fn += frame_step

    return records


# Per-frame layout per-route processor (legacy compatible)
def process_route_per_frame(
    route_data_dir: str,
    route_drivelm_dir: str,
    route_commentary_dir: str,
    scenario: str,
    run: str,
    profile: str,
    driver: str,
    start_frame: int = 10,
    future_horizon: int = 10,
    frame_step: int = 5,
) -> list[dict]:
    route_name       = os.path.basename(route_data_dir)
    measurements_dir = os.path.join(route_data_dir,       "measurements")
    vqa_dir          = os.path.join(route_drivelm_dir,    "vqa")
    commentary_dir   = os.path.join(route_commentary_dir, "commentary")

    if not os.path.isdir(measurements_dir):
        return []

    records: list[dict] = []
    fn = start_frame
    cache = _build_initial_cache(measurements_dir, fn, future_horizon)

    while True:
        frame_id = f"{fn:04d}"

        vqa_path  = _resolve_path(os.path.join(vqa_dir,        f"{frame_id}.json"))
        comm_path = _resolve_path(os.path.join(commentary_dir, f"{frame_id}.json"))
        if vqa_path is None or comm_path is None:
            break

        try:
            vqa_data  = _load_file(vqa_path)
            comm_data = _load_file(comm_path)

            window, ok = _get_contiguous_window(cache, fn, future_horizon)
            if not ok:
                break
            for off in range(frame_step):
                _advance_cache(
                    cache,
                    evict_frame=fn + off,
                    measurements_dir=measurements_dir,
                    next_frame=fn + future_horizon + 1 + off,
                )

            vqa_qa = vqa_data.get("QA", {}) if isinstance(vqa_data, dict) else {}
            if isinstance(vqa_qa, str):
                vqa_qa = _parse_qa_field(vqa_qa)
            commentary_text = comm_data.get("commentary", "") if isinstance(comm_data, dict) else ""

            rec = _build_record(
                fn, route_data_dir, scenario, run, profile, driver,
                window[0], window, vqa_qa, commentary_text, future_horizon,
            )
            if rec is None:
                break
            records.append(rec)
        except Exception as e:
            print(f"[{route_name}] per_frame frame {frame_id}: {e}")
            break

        fn += frame_step

    return records


# Worker
def _process_one_route(args: tuple) -> list[dict]:
    (route_data_dir, route_drivelm_dir, route_commentary_dir,
     layout, scenario, run, profile, driver, frame_step) = args
    if layout == "aggregated":
        return process_route_aggregated(
            route_data_dir, route_drivelm_dir, route_commentary_dir,
            scenario, run, profile, driver, frame_step=frame_step,
        )
    if layout == "per_frame":
        return process_route_per_frame(
            route_data_dir, route_drivelm_dir, route_commentary_dir,
            scenario, run, profile, driver, frame_step=frame_step,
        )
    return []


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process simlingo routes (aggregated or per-frame, auto-detected)"
    )
    parser.add_argument("--data-root",   type=str, required=True,
                        help="Root containing data/, drivelm/, commentary/ subtrees "
                             "(e.g. /share/.../database/simlingo)")
    parser.add_argument("--profile",     type=str, required=True,
                        help="Profile subdir under data/simlingo/<profile> "
                             "(fast | slow | default | training_1_scenario)")
    parser.add_argument("--frame-step",  type=int, default=5,
                        help="Stride between sampled frames (default 5)")
    parser.add_argument("--output-dir",  type=str, required=True,
                        help="Where to write <profile>_dataset_processed.json")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    profile_data_dir       = os.path.join(args.data_root, "data",       "simlingo", args.profile)
    profile_drivelm_dir    = os.path.join(args.data_root, "drivelm",    "simlingo", args.profile)
    profile_commentary_dir = os.path.join(args.data_root, "commentary", "simlingo", args.profile)

    if not os.path.isdir(profile_data_dir):
        print(f"ERROR: profile data dir not found: {profile_data_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"DATA_ROOT  : {args.data_root}")
    print(f"PROFILE    : {args.profile}")
    print(f"DATA       : {profile_data_dir}")
    print(f"DRIVELM    : {profile_drivelm_dir}")
    print(f"COMMENTARY : {profile_commentary_dir}")
    print(f"OUTPUT_DIR : {args.output_dir}")
    print(f"workers={args.num_workers} | frame_step={args.frame_step}")

    # ── Discover routes ──
    print("\nDiscovering routes...")
    route_dirs = find_routes(profile_data_dir)
    print(f"Found {len(route_dirs)} route directories")

    # ── Build worker tasks (auto-detect layout per route) ──
    tasks: list[tuple] = []
    layout_counts = {"aggregated": 0, "per_frame": 0, "skip": 0}
    for route_data_dir in route_dirs:
        layout = detect_route_layout(route_data_dir)
        if layout is None:
            layout_counts["skip"] += 1
            continue
        layout_counts[layout] += 1

        rel = os.path.relpath(route_data_dir, profile_data_dir)
        route_drivelm_dir    = os.path.join(profile_drivelm_dir,    rel)
        route_commentary_dir = os.path.join(profile_commentary_dir, rel)

        # Heuristics for scenario / run / driver from the path components
        # (rel is relative to data/simlingo/<profile>/):
        #   scenario = parts[-3] when available (".../<scenario>/<id>/<rep>")
        #              else parts[-2] ("routes_training/<scenario>/<route>")
        #   driver   = the first folder under the profile (driverN, lb1_split,
        #              routes_training, ...). Captures the top-level grouping
        #              regardless of whether the layout uses driver subdirs.
        parts = rel.split(os.sep)
        scenario = (
            parts[-3] if len(parts) >= 3
            else parts[-2] if len(parts) >= 2
            else "unknown"
        )
        driver = parts[0] if parts else ""
        run    = rel.replace(os.sep, "/")

        tasks.append((
            route_data_dir, route_drivelm_dir, route_commentary_dir,
            layout, scenario, run, args.profile, driver, args.frame_step,
        ))

    print(
        f"Layouts: aggregated={layout_counts['aggregated']} "
        f"per_frame={layout_counts['per_frame']} skip={layout_counts['skip']}"
    )
    print(f"Routes to process: {len(tasks)}")
    if not tasks:
        print("No routes to process.")
        return

    # ── Parallel processing ──
    all_records: list[dict] = []
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(_process_one_route, t) for t in tasks]
        completed = 0
        for fut in as_completed(futures):
            try:
                recs = fut.result()
                all_records.extend(recs)
            except Exception as e:
                print(f"  [ERROR] task: {e}")
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - t0
                rate    = completed / elapsed if elapsed > 0 else 0
                eta     = (len(tasks) - completed) / rate if rate > 0 else float("inf")
                print(
                    f"  [{completed}/{len(tasks)}] records={len(all_records)} "
                    f"| {elapsed:.1f}s | ETA={eta:.0f}s"
                )

    # ── Save ──
    out_file = os.path.join(args.output_dir, f"{args.profile}_dataset_processed.json")
    with open(out_file, "w") as f:
        json.dump(all_records, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"\n[DONE] Wrote {len(all_records)} records → {out_file}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()