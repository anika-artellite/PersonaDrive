

import json
import os
import gzip
import time
import numpy as np
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Unified JSON loader - one code path for both gzip and plain files,
# with optional orjson fast path. Everything reads in binary mode so
# orjson (which expects bytes) and the fallback both work identically.
# ---------------------------------------------------------------------------
try:
    import orjson
    def _load_json_bytes(data: bytes):
        return orjson.loads(data)
    _USING_ORJSON = True
except ImportError:
    def _load_json_bytes(data: bytes):
        return json.loads(data.decode('utf-8'))
    _USING_ORJSON = False

def _load_file(path: str):
    """
    Load a .json or .json.gz file and return a parsed dict.
    Always reads in binary mode so the same _load_json_bytes()
    is used regardless of compression or json backend.
    Raises on any error - callers decide how to handle it.
    """
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return _load_json_bytes(f.read())
    else:
        with open(path, 'rb') as f:
            return _load_json_bytes(f.read())

def _resolve_path(base_path: str):
    """
    Given a base path (without extension), returns the actual path
    that exists on disk (.json.gz preferred over .json), or None.
    """
    gz = base_path + '.gz'
    if os.path.exists(gz):
        return gz
    if os.path.exists(base_path):
        return base_path
    return None


# ---------------------------------------------------------------------------
# RoadOption
# ---------------------------------------------------------------------------
class RoadOption(Enum):
    VOID            = -1
    LEFT            =  1
    RIGHT           =  2
    STRAIGHT        =  3
    LANEFOLLOW      =  4
    CHANGELANELEFT  =  5
    CHANGELANERIGHT =  6

_COMMAND_MAP = {
    RoadOption.VOID:            "Void",
    RoadOption.LEFT:            "Go Left",
    RoadOption.RIGHT:           "Go Right",
    RoadOption.STRAIGHT:        "Go Straight",
    RoadOption.LANEFOLLOW:      "Follow Lane",
    RoadOption.CHANGELANELEFT:  "Change Lane Left",
    RoadOption.CHANGELANERIGHT: "Change Lane Right",
}

def get_command_string(command_int):
    try:
        return _COMMAND_MAP.get(RoadOption(command_int), "Unknown Command")
    except ValueError:
        return "Unknown Command"


# ---------------------------------------------------------------------------
# Measurement loading - unified loader, early validation
# ---------------------------------------------------------------------------
def load_measurement(measurements_dir: str, frame_number: int):
    """
    Load a single measurement dict for the given frame number.
    Returns None (without raising) if the file does not exist.
    Raises RuntimeError if the file exists but is malformed,
    so the caller can stop the route and log a clear message.
    """
    base = os.path.join(measurements_dir, f"{frame_number:04d}.json")
    path = _resolve_path(base)
    if path is None:
        return None
    try:
        data = _load_file(path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {path}: {e}") from e

    # Validate ego_matrix early so downstream numpy ops never crash silently
    if 'ego_matrix' not in data:
        raise RuntimeError(f"Missing 'ego_matrix' in {path}")
    matrix = np.array(data['ego_matrix'])
    if matrix.shape[0] < 3 or matrix.shape[1] < 4:
        raise RuntimeError(
            f"'ego_matrix' in {path} has unexpected shape {matrix.shape} (expected >=3x4)"
        )
    return data


# ---------------------------------------------------------------------------
# Sliding window cache
# ---------------------------------------------------------------------------
def _build_initial_cache(measurements_dir: str, start_frame: int,
                         future_horizon: int) -> dict:
    """
    Pre-fill the sliding window cache with frames
    [start_frame .. start_frame + future_horizon].
    Stops early if a frame is missing (end of route).
    """
    cache = {}
    for offset in range(future_horizon + 1):
        fn = start_frame + offset
        m  = load_measurement(measurements_dir, fn)
        if m is None:
            break
        cache[fn] = m
    return cache


def _advance_cache(cache: dict, evict_frame: int,
                   measurements_dir: str, next_frame: int):
    """
    Evict the oldest frame and load the next one at the far end.
    Mutates cache in-place. One file read per frame advance.
    """
    cache.pop(evict_frame, None)
    m = load_measurement(measurements_dir, next_frame)
    if m is not None:
        cache[next_frame] = m


def _get_contiguous_window(cache: dict, start_frame: int,
                           future_horizon: int):
    """
    Extract a strictly contiguous list from the cache starting at start_frame.
    Stops at the first gap - never skips frames silently.

    Returns (window: list[dict], is_complete: bool)
    is_complete == True means all future_horizon+1 frames are present with no gaps.
    """
    window = []
    for offset in range(future_horizon + 1):
        fn = start_frame + offset
        if fn not in cache:
            break   # gap - stop here to preserve temporal alignment
        window.append(cache[fn])
    return window, len(window) == future_horizon + 1


# ---------------------------------------------------------------------------
# Waypoint extraction
# ---------------------------------------------------------------------------
def get_waypoints_from_measurements(measurements_list: list) -> list:
    """
    Transforms all measurement positions into the ego frame of index 0.
    Returns a list of float32 (2,) arrays; index 0 is the origin [0, 0].
    """
    origin   = measurements_list[0]
    om       = np.array(origin['ego_matrix'])[:3]   # (3, 4)
    origin_t = om[:, 3:4]                           # (3, 1)
    origin_R = om[:, :3]                            # (3, 3)

    waypoints = []
    for meas in measurements_list:
        wp  = np.array(meas['ego_matrix'])[:3, 3:4]
        ego = origin_R.T @ (wp - origin_t)
        waypoints.append(ego[:2, 0].astype(np.float32))
    return waypoints


def extract_future_waypoints(window: list, future_horizon: int) -> dict:
    """
    Compute future waypoints from a contiguous measurement window.
    window[0] = current/ego-origin frame; window[1:] = future frames.

    Returns:
        future_waypoints     (N, 2) float32 - ego-frame positions, origin excluded
        future_waypoints_1d  (N,)  float32 - cumulative arc-length in metres
        num_future_frames    int
    """
    num_future = len(window) - 1
    if num_future < 1:
        return {
            'future_waypoints':    np.zeros((0, 2), dtype=np.float32),
            'future_waypoints_1d': np.zeros((0,),   dtype=np.float32),
            'num_future_frames':   0,
        }

    all_wps    = get_waypoints_from_measurements(window)
    future_wps = np.array(all_wps[1:], dtype=np.float32)       # (N, 2)

    with_origin = np.vstack([np.zeros((1, 2), dtype=np.float32), future_wps])
    step_dists  = np.linalg.norm(np.diff(with_origin, axis=0), axis=1)
    cumulative  = np.cumsum(step_dists).astype(np.float32)

    return {
        'future_waypoints':    future_wps,
        'future_waypoints_1d': cumulative,
        'num_future_frames':   num_future,
    }


def extract_temporal_waypoints(window: list):
    """
    Mirrors BaseDataset.extract_temporal_waypoints() with hist_len=1.
    Accepts the same contiguous window as extract_future_waypoints.

    Returns None if fewer than 3 frames (need at least one waypoint
    after the [1:-1] slice). Otherwise returns:
        waypoints        (N-2, 2)    - positions [1:-1] in ego frame
        waypoints_org    (N-2, 2)    - same (no augmentation here)
        waypoints_1d     (N-2, 2)    - [[cumulative_dist, 0], ...]
        ego_waypoints    (N-1, 4, 4) - homogeneous matrices
    """
    if len(window) < 3:
        return None

    all_wps   = get_waypoints_from_measurements(window)
    inner_wps = np.array(all_wps[1:-1], dtype=np.float32)      # (N-2, 2)

    step_dists = [
        np.linalg.norm(all_wps[i + 1] - all_wps[i])
        for i in range(len(all_wps) - 1)
    ]
    cumsum = np.cumsum(step_dists).astype(np.float32)           # (N-1,)
    wps_1d = np.array([[d, 0.0] for d in cumsum],
                      dtype=np.float32)[:-1]                    # (N-2, 2)

    ego_wps = np.array([
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]
        for x, y in all_wps
    ], dtype=np.float32)[:-1]                                   # (N-1, 4, 4)

    return {
        'waypoints':     inner_wps,
        'waypoints_org': inner_wps,
        'waypoints_1d':  wps_1d,
        'ego_waypoints': ego_wps,
    }


# ---------------------------------------------------------------------------
# Geometric route extraction (mirrors BaseDataset exactly)
# ---------------------------------------------------------------------------
def equal_spacing_route(points: np.ndarray, spacing: float = 1.0,
                        num_points: int = 20) -> np.ndarray:
    """
    Resamples a route to `num_points` equally spaced by `spacing` metres.
    Prepends the origin (0,0) before interpolating, matching BaseDataset.
    """
    route = np.vstack([np.zeros((1, 2), dtype=np.float32), points])  # prepend origin
    shift = np.roll(route, 1, axis=0)
    shift[0] = shift[1]
    dists = np.linalg.norm(route - shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(len(dists)) * 1e-4   # ensure strictly increasing
    x = np.arange(0, num_points, spacing)
    resampled = np.array([
        np.interp(x, dists, route[:, 0]),
        np.interp(x, dists, route[:, 1]),
    ], dtype=np.float32).T                  # (num_points, 2)
    return resampled


def extract_geometric_waypoints(
    current_measurement: dict,
    num_route_points: int = 20,
) -> dict:
    """
    Extracts geometric (route-based) waypoints from the current measurement.
    Mirrors BaseDataset.extract_geometric_waypoints() without augmentation.

    Args:
        current_measurement: dict with keys 'route_original' (Nx2) and 'route' (Mx2).
        num_route_points:    how many points to take from route_original before resampling.

    Returns dict with:
        route              (20, 2) - resampled route_original
        route_adjusted     (20, 2) - resampled adjusted route
    """
    for key in ('route_original', 'route'):
        if key not in current_measurement:
            raise RuntimeError(
                f"extract_geometric_waypoints: missing key '{key}' in measurement"
            )

    # --- route_original branch ---
    route_orig = np.array(current_measurement['route_original'], dtype=np.float32)
    if len(route_orig) < num_route_points:
        pad        = num_route_points - len(route_orig)
        route_orig = np.vstack([route_orig, np.tile(route_orig[-1], (pad, 1))])
    else:
        route_orig = route_orig[:num_route_points]

    # --- route_adjusted branch ---
    route_adj = np.array(current_measurement['route'], dtype=np.float32)

    return {
        'route':          equal_spacing_route(route_orig),
        'route_adjusted': equal_spacing_route(route_adj),
    }


# def process_route_data(
#     base_route_path: str,
#     scenario: str,
#     run: str,
#     start_frame: int = 10,
#     future_horizon: int = 10,
#     frame_step: int = 3,
#     data_root: str = None,
#     split_name: str = "lb1_split",
#     pivot_frame: int | None = None,   # <- NEW: if set, process only this frame

# ) -> list:
#     """
#     Process all frames for a single route and return a list of records.

#     Args:
#         base_route_path : absolute path to the route directory
#         scenario        : scenario folder name (e.g. 'noScenarios')
#         run             : route folder name used as record key
#         start_frame     : first frame to process (default 10)
#         future_horizon  : number of contiguous future frames needed for
#                           waypoint extraction (default 10).
#         frame_step      : how many frames to advance between records
#                           (default 5 -> 1.25 s apart at 4 Hz / 0.25 s/frame).
#         data_root       : Root directory containing drivelm/, commentary/, data/.
#                   If None, auto-detected from base_route_path (fragile).
#         split_name      : Dataset split folder name (e.g. "lb1_split").
#     """
    
# # Extract route_name from base_route_path
#     route_name = os.path.basename(base_route_path)
    
# # Resolve data_root
#     if data_root is None:
#         # Fragile auto-detection (legacy behavior) - 6 levels up
#         data_root = os.path.abspath(os.path.join(base_route_path, '..', '..', '..', '..', '..', '..'))
    
#     drivelm_base    = os.path.join(data_root, 'drivelm')
#     commentary_base = os.path.join(data_root, 'commentary')
#     data_base       = os.path.join(data_root, 'data')
    
# # Build the actual directory paths
#     vqa_dir          = os.path.join(drivelm_base,    'simlingo', split_name, 'routes_training', scenario, route_name, 'vqa')
#     commentary_dir   = os.path.join(commentary_base, 'simlingo', split_name, 'routes_training', scenario, route_name, 'commentary')
#     measurements_dir = os.path.join(data_base,       'simlingo', split_name, 'routes_training', scenario, route_name, 'measurements')
    
# # Validate paths exist
#     for path, name in [
#         (drivelm_base, 'drivelm'),
#         (commentary_base, 'commentary'),
#         (data_base, 'data'),
#     ]:
#         if not os.path.isdir(path):
#             raise FileNotFoundError(f"Required directory not found: {path}")
    
#     # Optional: also validate the specific subdirs exist
#     for path, name in [
#         (vqa_dir, 'vqa_dir'),
#         (commentary_dir, 'commentary_dir'),
#         (measurements_dir, 'measurements_dir'),
#     ]:
#         if not os.path.isdir(path):
#             print(f"[{route_name}] Warning: {name} not found: {path}")
#             return []  # Return empty if data directories don't exist

#     all_records      = []
#     frame_number     = start_frame
#     route_start_time = time.perf_counter()
#     frame_times      = []

#     # Pre-fill the sliding window cache.
#     meas_cache = _build_initial_cache(measurements_dir, start_frame, future_horizon)

#     while True:
#         frame_id = f"{frame_number:04d}"

#         # Resolve VQA + commentary paths
#         vqa_path        = _resolve_path(os.path.join(vqa_dir,        f"{frame_id}.json"))
#         commentary_path = _resolve_path(os.path.join(commentary_dir, f"{frame_id}.json"))

#         if vqa_path is None or commentary_path is None:
#             msg = "No data found" if frame_number == start_frame else "File missing - stopping"
#             print(f"[{route_name}] frame {frame_id}: {msg} (VQA or commentary).")
#             break

#         try:
#             frame_start_time = time.perf_counter()

#             vqa_data        = _load_file(vqa_path)
#             commentary_data = _load_file(commentary_path)

#             future_measurements, is_complete = _get_contiguous_window(
#                 meas_cache, frame_number, future_horizon
#             )
#             if not is_complete:
#                 print(
#                     f"[{route_name}] frame {frame_id}: only "
#                     f"{len(future_measurements)-1} contiguous future frames "
#                     f"(need {future_horizon}). Stopping route."
#                 )
#                 break

#             for step_offset in range(frame_step):
#                 _advance_cache(
#                     meas_cache,
#                     evict_frame=frame_number + step_offset,
#                     measurements_dir=measurements_dir,
#                     next_frame=frame_number + future_horizon + 1 + step_offset,
#                 )

#             future_wp_data   = extract_future_waypoints(future_measurements, future_horizon)
#             temporal_wp_data = extract_temporal_waypoints(future_measurements)

#             if temporal_wp_data is None:
#                 print(f"[{route_name}] frame {frame_id}: insufficient frames for temporal waypoints. Stopping.")
#                 break

#             measurements_data = future_measurements[0]

#             geo_wp_data = extract_geometric_waypoints(
#                 current_measurement=measurements_data,
#                 num_route_points=20,
#             )

#             # ---- Ego state ----
#             speed    = measurements_data.get('speed',    'N/A')
#             throttle = measurements_data.get('throttle', 'N/A')
#             steer    = measurements_data.get('steer',    'N/A')
#             aim_wp   = measurements_data.get('aim_wp',   'N/A')

#             speed_str    = f"{speed:.4f}"    if isinstance(speed,    (int, float)) else str(speed)
#             throttle_str = f"{throttle:.4f}" if isinstance(throttle, (int, float)) else str(throttle)
#             steer_str    = f"{steer:.4f}"    if isinstance(steer,    (int, float)) else str(steer)
#             aim_wp_str   = (f"[{aim_wp[0]:.4f}, {aim_wp[1]:.4f}]"
#                             if isinstance(aim_wp, list) and len(aim_wp) == 2
#                             else str(aim_wp))

#             ego_state = (
#                 f"--- EGO VEHICLE STATE ---\n"
#                 f"Speed: {speed_str}\n"
#                 f"Throttle: {throttle_str}\n"
#                 f"Brake: {'Brake is clicked' if measurements_data.get('brake') else 'Brake is not clicked'}\n"
#                 f"Steer: {steer_str}\n"
#                 f"Current Command: {get_command_string(measurements_data.get('command', 'N/A'))}\n"
#             )

#             # ---- Q&A ----
#             qa_parts = []
#             for section in ("perception", "prediction", "planning"):
#                 qa_parts.append(f"--- {section.upper()} ---\n")
#                 items = vqa_data.get("QA", {}).get(section, [])
#                 if not items:
#                     qa_parts.append(f"No data found for {section}.\n")
#                 else:
#                     for item in items:
#                         qa_parts.append(f"Q: {item.get('Q', 'N/A')}\nA: {item.get('A', 'N/A')}\n\n")

#             # ---- Commentary ----
#             commentary_text = commentary_data.get("commentary", "No commentary found.")
#             commentary_section = (
#                 f"--- COMMENTARY ---\n"
#                 f"Next Aim Waypoint: {aim_wp_str}\n"
#                 f"{commentary_text}\n"
#             )

#             # ---- Future waypoints ----
#             wp_str = ", ".join(
#                 f"[{x:.4f}, {y:.4f}]"
#                 for x, y in future_wp_data['future_waypoints']
#             )
#             future_wp_section = f"--- FUTURE WAYPOINTS ---\nFuture Waypoints: ({wp_str})\n"

#             full_sentence = ego_state + "".join(qa_parts) + commentary_section + future_wp_section

#             frame_path = commentary_data.get("image", "N/A")
#             if "_v2_2025_01_10" in frame_path:
#                 frame_path = frame_path.replace("_v2_2025_01_10", "")

#             record = {
#                 "frame_path":             frame_path,
#                 "sentence":               full_sentence,
#                 "scenario":               scenario,
#                 "run":                    f"{run}/{frame_id}",
#                 "future_waypoints":       future_wp_data['future_waypoints'].tolist(),
#                 "temporal_waypoints_org": temporal_wp_data['waypoints_org'].tolist(),
#                 "route_adjusted":         geo_wp_data['route_adjusted'].tolist(),
#                 "target_point":           measurements_data.get('target_point'),
#                 "target_point_next":      measurements_data.get('target_point_next'),
#                 "command":                measurements_data.get('command'),
#                 "next_command":           measurements_data.get('next_command'),
#             }
#             all_records.append(record)

#             frame_elapsed = time.perf_counter() - frame_start_time
#             frame_times.append(frame_elapsed)
#             if len(frame_times) % 50 == 0:
#                 avg = sum(frame_times) / len(frame_times)
#                 print(
#                     f"[{route_name}] frame {frame_id} | "
#                     f"processed: {len(frame_times)} | "
#                     f"avg: {avg*1000:.1f}ms/frame | "
#                     f"elapsed: {time.perf_counter()-route_start_time:.1f}s"
#                 )

#             frame_number += frame_step

#         except RuntimeError as e:
#             print(f"[{route_name}] Data error at frame {frame_id}: {e}. Stopping route.")
#             break
#         except Exception as e:
#             print(f"[{route_name}] Unexpected error at frame {frame_id}: {e}. Stopping route.")
#             break

#     route_elapsed = time.perf_counter() - route_start_time
#     avg_frame     = (sum(frame_times) / len(frame_times) * 1000) if frame_times else 0.0
#     print(
#         f"[{route_name}] DONE | "
#         f"records: {len(all_records)} | "
#         f"step: {frame_step} | "
#         f"avg: {avg_frame:.1f}ms/frame | "
#         f"total: {route_elapsed:.2f}s"
#     )
#     return all_records
def process_route_data(
    base_route_path: str,
    scenario: str,
    run: str,
    start_frame: int = 10,
    future_horizon: int = 10,
    frame_step: int = 3,
    data_root: str = None,
    split_name: str = "lb1_split",
    pivot_frame: int | None = None,   # <- if set, process only this frame
    route_sub: str | None = None,     # <- sub-path under scenario/; None = backward-compatible
) -> list:
    """
    Process all frames for a single route and return a list of records.

    Args:
        base_route_path : absolute path to the route directory
        scenario        : scenario folder name (e.g. 'noScenarios')
        run             : route folder name used as record key
        start_frame     : first frame to process (default 10)
        future_horizon  : number of contiguous future frames needed for
                          waypoint extraction (default 10).
        frame_step      : how many frames to advance between records
                          (default 5 -> 1.25 s apart at 4 Hz / 0.25 s/frame).
        data_root       : Root directory containing drivelm/, commentary/, data/.
                          If None, auto-detected from base_route_path (fragile).
        split_name      : Dataset split folder name (e.g. "lb1_split").
        pivot_frame     : If set, process ONLY this single frame (no sliding window).
                          future_horizon should already be set to end - pivot by caller.
    """

    # Extract route_name from base_route_path
    route_name = os.path.basename(base_route_path)
    route_sub  = route_sub if route_sub is not None else route_name

    # Resolve data_root
    if data_root is None:
        data_root = os.path.abspath(os.path.join(base_route_path, '..', '..', '..', '..', '..', '..'))

    drivelm_base    = os.path.join(data_root, 'drivelm')
    commentary_base = os.path.join(data_root, 'commentary')
    data_base       = os.path.join(data_root, 'data')

    # Build the actual directory paths
    vqa_dir          = os.path.join(drivelm_base,    'simlingo', split_name, 'routes_training', scenario, route_sub, 'vqa')
    commentary_dir   = os.path.join(commentary_base, 'simlingo', split_name, 'routes_training', scenario, route_sub, 'commentary')
    measurements_dir = os.path.join(data_base,       'simlingo', split_name, 'routes_training', scenario, route_sub, 'measurements')

    # Validate paths exist
    for path, name in [
        (drivelm_base, 'drivelm'),
        (commentary_base, 'commentary'),
        (data_base, 'data'),
    ]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required directory not found: {path}")

    for path, name in [
        (vqa_dir, 'vqa_dir'),
        (commentary_dir, 'commentary_dir'),
        (measurements_dir, 'measurements_dir'),
    ]:
        if not os.path.isdir(path):
            print(f"[{route_name}] Warning: {name} not found: {path}")
            return []

    all_records      = []
    route_start_time = time.perf_counter()
    frame_times      = []

    # Pivot mode: override frame_number and seed cache at pivot
    # Sliding mode: start from start_frame as before
    frame_number = pivot_frame if pivot_frame is not None else start_frame
    meas_cache = _build_initial_cache(measurements_dir, frame_number, future_horizon)

    while True:
        frame_id = f"{frame_number:04d}"

        vqa_path        = _resolve_path(os.path.join(vqa_dir,        f"{frame_id}.json"))
        commentary_path = _resolve_path(os.path.join(commentary_dir, f"{frame_id}.json"))

        if vqa_path is None or commentary_path is None:
            msg = "No data found" if frame_number == (pivot_frame or start_frame) else "File missing — stopping"
            print(f"[{route_name}] frame {frame_id}: {msg} (VQA or commentary).")
            break

        try:
            frame_start_time = time.perf_counter()

            vqa_data        = _load_file(vqa_path)
            commentary_data = _load_file(commentary_path)

            future_measurements, is_complete = _get_contiguous_window(
                meas_cache, frame_number, future_horizon
            )
            if not is_complete:
                print(
                    f"[{route_name}] frame {frame_id}: only "
                    f"{len(future_measurements)-1} contiguous future frames "
                    f"(need {future_horizon}). Stopping route."
                )
                break

            # Only advance the cache in sliding window mode
            if pivot_frame is None:
                for step_offset in range(frame_step):
                    _advance_cache(
                        meas_cache,
                        evict_frame=frame_number + step_offset,
                        measurements_dir=measurements_dir,
                        next_frame=frame_number + future_horizon + 1 + step_offset,
                    )

            future_wp_data   = extract_future_waypoints(future_measurements, future_horizon)
            temporal_wp_data = extract_temporal_waypoints(future_measurements)

            if temporal_wp_data is None:
                print(f"[{route_name}] frame {frame_id}: insufficient frames for temporal waypoints. Stopping.")
                break

            measurements_data = future_measurements[0]

            geo_wp_data = extract_geometric_waypoints(
                current_measurement=measurements_data,
                num_route_points=20,
            )

            # ---- Ego state ----
            speed    = measurements_data.get('speed',    'N/A')
            throttle = measurements_data.get('throttle', 'N/A')
            steer    = measurements_data.get('steer',    'N/A')
            aim_wp   = measurements_data.get('aim_wp',   'N/A')

            speed_str    = f"{speed:.4f}"    if isinstance(speed,    (int, float)) else str(speed)
            throttle_str = f"{throttle:.4f}" if isinstance(throttle, (int, float)) else str(throttle)
            steer_str    = f"{steer:.4f}"    if isinstance(steer,    (int, float)) else str(steer)
            aim_wp_str   = (f"[{aim_wp[0]:.4f}, {aim_wp[1]:.4f}]"
                            if isinstance(aim_wp, list) and len(aim_wp) == 2
                            else str(aim_wp))

            ego_state = (
                f"--- EGO VEHICLE STATE ---\n"
                f"Speed: {speed_str}\n"
                f"Throttle: {throttle_str}\n"
                f"Brake: {'Brake is clicked' if measurements_data.get('brake') else 'Brake is not clicked'}\n"
                f"Steer: {steer_str}\n"
                f"Current Command: {get_command_string(measurements_data.get('command', 'N/A'))}\n"
                f"Next Aim Waypoint: {aim_wp_str}\n"

            )

            # ---- Q&A ----
            qa_parts = []
            for section in ("perception", "prediction", "planning"):
                qa_parts.append(f"--- {section.upper()} ---\n")
                items = vqa_data.get("QA", {}).get(section, [])
                if not items:
                    qa_parts.append(f"No data found for {section}.\n")
                else:
                    for item in items:
                        qa_parts.append(f"Q: {item.get('Q', 'N/A')}\nA: {item.get('A', 'N/A')}\n\n")

            # ---- Commentary ----
            commentary_text = commentary_data.get("commentary", "No commentary found.")
            commentary_section = (
                f"--- COMMENTARY ---\n"
                f"{commentary_text}\n"
            )

            # ---- Future waypoints ----
            wp_str = ", ".join(
                f"[{x:.4f}, {y:.4f}]"
                for x, y in future_wp_data['future_waypoints']
            )
            future_wp_section = f"--- FUTURE WAYPOINTS ---\nFuture Waypoints: ({wp_str})\n"

            full_sentence = ego_state + "".join(qa_parts) + commentary_section + future_wp_section

            frame_path = commentary_data.get("image", "N/A")
            if "_v2_2025_01_10" in frame_path:
                frame_path = frame_path.replace("_v2_2025_01_10", "")

            record = {
                "frame_path":             frame_path,
                "sentence":               full_sentence,
                "scenario":               scenario,
                "run":                    f"{run}/{frame_id}",
                "future_waypoints":       future_wp_data['future_waypoints'].tolist(),
                "temporal_waypoints_org": temporal_wp_data['waypoints_org'].tolist(),
                "route_adjusted":         geo_wp_data['route_adjusted'].tolist(),
                "target_point":           measurements_data.get('target_point'),
                "target_point_next":      measurements_data.get('target_point_next'),
                "command":                measurements_data.get('command'),
                "next_command":           measurements_data.get('next_command'),
            }
            all_records.append(record)

            frame_elapsed = time.perf_counter() - frame_start_time
            frame_times.append(frame_elapsed)
            if len(frame_times) % 50 == 0:
                avg = sum(frame_times) / len(frame_times)
                print(
                    f"[{route_name}] frame {frame_id} | "
                    f"processed: {len(frame_times)} | "
                    f"avg: {avg*1000:.1f}ms/frame | "
                    f"elapsed: {time.perf_counter()-route_start_time:.1f}s"
                )

            # Pivot mode: done after one frame
            if pivot_frame is not None:
                break
            frame_number += frame_step

        except RuntimeError as e:
            print(f"[{route_name}] Data error at frame {frame_id}: {e}. Stopping route.")
            break
        except Exception as e:
            print(f"[{route_name}] Unexpected error at frame {frame_id}: {e}. Stopping route.")
            break

    route_elapsed = time.perf_counter() - route_start_time
    avg_frame     = (sum(frame_times) / len(frame_times) * 1000) if frame_times else 0.0
    print(
        f"[{route_name}] DONE | "
        f"records: {len(all_records)} | "
        f"step: {'pivot' if pivot_frame is not None else frame_step} | "
        f"avg: {avg_frame:.1f}ms/frame | "
        f"total: {route_elapsed:.2f}s"
    )
    return all_records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG data processor for simlingo dataset")
    parser.add_argument(
        "--data-root", type=str,
        default=os.path.join(os.environ.get("PERSONADRIVE_DATA_ROOT", "/path/to/database").split(":")[0], "simlingo"),
        help="Root dir containing drivelm/, commentary/, data/ (or fast_dataset/, slow_dataset/)"
    )
    parser.add_argument("--frame-step",  type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="Where to save output JSON files (default: same as --data-root)")
    cli = parser.parse_args()

    DATA_ROOT   = cli.data_root
    FRAME_STEP  = cli.frame_step
    NUM_WORKERS = cli.num_workers
    OUTPUT_DIR  = cli.output_dir or DATA_ROOT

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"DATA_ROOT  : {DATA_ROOT}")
    print(f"OUTPUT_DIR : {OUTPUT_DIR}")
    print(f"workers={NUM_WORKERS} | frame_step={FRAME_STEP} | orjson={_USING_ORJSON}")

    # Detect layout
    # Layout A (new): DATA_ROOT/{fast_dataset,slow_dataset}/data/simlingo/...
    # Layout B (old): DATA_ROOT/data/simlingo/...
    DATASET_SPLITS = []   # list of (split_label, data_root_for_split)

    for dataset_split in ["fast_dataset", "slow_dataset"]:
        candidate = os.path.join(DATA_ROOT, dataset_split)
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "data")):
            DATASET_SPLITS.append((dataset_split, candidate))

    if not DATASET_SPLITS:
        # Fall back to old layout - treat DATA_ROOT itself as the single root
        DATASET_SPLITS = [("all", DATA_ROOT)]

    print(f"Dataset splits found: {[s for s, _ in DATASET_SPLITS]}")

    global_start = time.perf_counter()

    for dataset_label, split_root in DATASET_SPLITS:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_label}  ({split_root})")
        print(f"{'='*60}")

        data_simlingo_path = os.path.join(split_root, "data", "simlingo")
        if not os.path.isdir(data_simlingo_path):
            print(f"  Skipping — data/simlingo not found: {data_simlingo_path}")
            continue

        # Auto-discover internal splits (e.g. lb1_split)
        internal_splits = [
            d for d in os.listdir(data_simlingo_path)
            if os.path.isdir(os.path.join(data_simlingo_path, d))
        ]
        print(f"  Internal splits: {internal_splits}")

        split_records = []
        tasks         = []

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for split_name in internal_splits:
                routes_training_path = os.path.join(
                    split_root, "data", "simlingo", split_name, "routes_training"
                )
                if not os.path.isdir(routes_training_path):
                    print(f"  Skipping {split_name}: no routes_training folder")
                    continue

                print(f"\n  ── Split: {split_name}")

                for scenario_folder in sorted(os.listdir(routes_training_path)):
                    scenario_path = os.path.join(routes_training_path, scenario_folder)
                    if not os.path.isdir(scenario_path):
                        continue

                    for entry in sorted(os.listdir(scenario_path)):
                        entry_path = os.path.join(scenario_path, entry)
                        if not os.path.isdir(entry_path):
                            continue

                        # Check if entry contains measurements/ directly (old 2-level layout)
                        # or contains sub-folders with measurements/ (new 3-level layout)
                        if os.path.isdir(os.path.join(entry_path, "measurements")):
                            # Old layout: routes_training/{scenario}/{route}/measurements/
                            route_path = entry_path
                            route_sub  = None   # backward-compatible
                            run_key    = entry
                            tasks.append(executor.submit(
                                process_route_data,
                                route_path, scenario_folder, run_key,
                                10, 10, FRAME_STEP,
                                split_root, split_name, None, route_sub,
                            ))
                        else:
                            # New layout: routes_training/{scenario}/{scen_folder}/{rep_folder}/measurements/
                            for rep_folder in sorted(os.listdir(entry_path)):
                                rep_path = os.path.join(entry_path, rep_folder)
                                if not os.path.isdir(rep_path):
                                    continue
                                route_sub = os.path.join(entry, rep_folder)
                                run_key   = f"{entry}/{rep_folder}"
                                tasks.append(executor.submit(
                                    process_route_data,
                                    rep_path, scenario_folder, run_key,
                                    10, 10, FRAME_STEP,
                                    split_root, split_name, None, route_sub,
                                ))

            completed = 0
            for future in as_completed(tasks):
                try:
                    split_records.extend(future.result())
                    completed += 1
                    if completed % 10 == 0 or completed == len(tasks):
                        elapsed   = time.perf_counter() - global_start
                        rate      = completed / elapsed if elapsed > 0 else 0
                        remaining = (len(tasks) - completed) / rate if rate > 0 else float("inf")
                        print(
                            f"  [PROGRESS] {completed}/{len(tasks)} routes | "
                            f"records: {len(split_records)} | "
                            f"elapsed: {elapsed:.1f}s | ETA: {remaining:.0f}s"
                        )
                except Exception as e:
                    print(f"  [ERROR] Task exception: {e}")

        # Save per-dataset-split output
        out_file = os.path.join(OUTPUT_DIR, f"{dataset_label}_processed.json")
        with open(out_file, "w") as f:
            json.dump(split_records, f, indent=4)
        print(f"  Saved {len(split_records)} records → {out_file}")

    global_elapsed = time.perf_counter() - global_start
    print(f"\n[DONE] Total time: {global_elapsed:.1f}s")