"""Complete the released `normal` retrieval index from your own copy of the SimLingo dataset.

The `normal` (no-style) index retrieves from a slice of the SimLingo dataset (PDM-lite autopilot data). The
SimLingo licence does not allow redistributing that data, so the Hugging Face release holds only the index
vectors (`driving_index.faiss`) and the list of frames they belong to (`frames.json.gz`). This script reads
those frames' measurement and commentary files from your SimLingo download and writes the metadata the
retriever loads (`driving_index.meta.pkl`) next to a copy of the vectors.

    python scripts/retrieval/fill_normal_index.py --release downloads/personadrive_model/retrieval/normal \\
        --out "${PERSONADRIVE_ICL_ROOT:-persona_drive_rag}/faiss_index/normal"

PERSONADRIVE_DATA_ROOT must contain simlingo/{data,commentary}/simlingo/ with the SimLingo splits
training_1_scenario, training_3_scenarios and lb1_split (see datasets/README.md).
"""
import argparse
import gzip
import json
import os
import pickle
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "persona_drive_rag"))
from build_faiss_index import (_derive_commentary_path, _load_gz, _load_measurement_state,  # noqa: E402
                               _normalize_path)
from rag_data_processor import get_command_string  # noqa: E402


def future_waypoints(meas_dir: Path, frame: int, n_future: int) -> list:
    """Ego-frame positions of the next n_future frames (float32), as rag_data_processor computes them."""
    window = [_load_gz(str(meas_dir / f"{frame + i:04d}.json.gz")) for i in range(n_future + 1)]
    if any(not m or "ego_matrix" not in m for m in window):
        return []
    om = np.array(window[0]["ego_matrix"])[:3]
    origin_t, origin_R = om[:, 3:4], om[:, :3]
    out = []
    for m in window[1:]:
        wp = np.array(m["ego_matrix"])[:3, 3:4]
        ego = (origin_R.T @ (wp - origin_t))[:2, 0].astype(np.float32)
        out.append([float(ego[0]), float(ego[1])])
    return out


def entry(frame: dict) -> dict:
    path = Path(_normalize_path(frame["frame_path"]))
    if not path.is_file():
        raise FileNotFoundError(frame["frame_path"])
    n = int(path.stem)
    meas_dir = path.parent.parent / "measurements"
    m = _load_gz(str(meas_dir / f"{n:04d}.json.gz"))
    if not m:
        raise FileNotFoundError(str(meas_dir / f"{n:04d}.json.gz"))
    com = _load_gz(_derive_commentary_path(str(path))) or {}
    return {
        "frame_id":          frame["frame_id"],
        "frame_path":        frame["frame_path"],
        "scenario":          path.parts[-4],
        "state": {
            "speed":    float(f"{m['speed']:.4f}"),
            "throttle": float(f"{m['throttle']:.4f}"),
            "steer":    float(f"{m['steer']:.4f}"),
            "brake":    bool(m["brake"]),
            "command":  get_command_string(m["command"]),
        },
        "state_1_back":      _load_measurement_state(str(path), -1),
        "state_2_back":      _load_measurement_state(str(path), -2),
        "command":           m.get("command"),
        "next_command":      m.get("next_command"),
        "target_point":      m.get("target_point"),
        "target_point_next": m.get("target_point_next"),
        "future_waypoints":  future_waypoints(meas_dir, n, frame["n_future"]),
        "commentary":        com.get("commentary", ""),
        "scenario_name":     com.get("scenario_name", ""),
        "cause_object":      com.get("cause_object_string", ""),
    }


def entry_or_missing(frame: dict):
    try:
        return entry(frame), None
    except FileNotFoundError as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, help="folder with driving_index.faiss, driving_index.info.json, frames.json.gz")
    ap.add_argument("--out", required=True, help="index folder to write, e.g. persona_drive_rag/faiss_index/normal")
    ap.add_argument("--workers", type=int, default=8, help="parallel readers; raise it on network file systems")
    args = ap.parse_args()

    release, out = Path(args.release), Path(args.out)
    with gzip.open(release / "frames.json.gz", "rt") as f:
        frames = json.load(f)
    meta, missing = [], []
    with Pool(args.workers) as pool:
        for i, (e, miss) in enumerate(pool.imap(entry_or_missing, frames, chunksize=32)):
            if miss:
                missing.append(miss)
            else:
                meta.append(e)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(frames)} frames", flush=True)
    if missing:
        print(f"ERROR: {len(missing)} of {len(frames)} frames not found under PERSONADRIVE_DATA_ROOT, e.g.\n  "
              + "\n  ".join(missing[:5]), file=sys.stderr)
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    for name in ("driving_index.faiss", "driving_index.info.json"):
        shutil.copyfile(release / name, out / name)
    with open(out / "driving_index.meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"{len(meta)} frames -> {out}")


if __name__ == "__main__":
    main()
