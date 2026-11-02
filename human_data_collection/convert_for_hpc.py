#!/usr/bin/env python3
"""
convert_for_hpc.py - repackage driver9-15 into the layout used by driver1-8 on HPC3.

Differences from our local layout, all handled here:

  1. a `routes/` level is inserted between <driver> and <scenario>
  2. boxes/NNNN.json.gz  -> boxes.json         (single uncompressed array, frame-indexed)
     measurements/*.gz   -> measurements.json  (ditto)
  3. lidar/ is dropped - driver1-8 have none anywhere (812 rgb, 812 rgb_augmented, 0 lidar)
  4. labels collapse the same way:
       vqa.json         list[n_frames]      None where no label was generated
       commentary.json  list[n_frames - 9]  None for the first 10 skipped frames
  5. absolute image paths inside the labels are rewritten to the HPC3 root

IMPORTANT: measurements.json carries the EXPERT (autopilot) controls, taken from the
.json.gz, because that is what driver1-8 already have on the cluster. The HUMAN controls
live in the sibling uncompressed measurements/NNNN.json and are NOT shipped by this
script. A human-valued variant can be produced from the same source by reading the .json
instead of the .json.gz - see the Output section of README.md, which explains why the two
files are structurally indistinguishable.

Writes one tar per driver with transformed member paths, streaming from the source: no
staging copy of the ~50k jpgs. Extract at <hpc>/database/simlingo/ and the three trees
(data/, drivelm/, commentary/) land in place.

    export DATASET_ROOT=/path/to/collective_dataset
    export LABELS_ROOT=/path/to/labels      # holds vqa/ and commentary/
    export HPC_ROOT=/path/on/hpc/database/simlingo
    python3 convert_for_hpc.py --out /path/to/hpc_tars
    python3 convert_for_hpc.py --out /path/to/hpc_tars --drivers driver9 --dry-run
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time

# Paths come from the environment; see the usage block above. Nothing is hardcoded,
# so no machine-specific or participant-identifying path ships with this script.
SRC_DEFAULT = os.environ.get("DATASET_ROOT", "")
_LABELS_ROOT = os.environ.get("LABELS_ROOT", "")
VQA_DEFAULT = os.path.join(_LABELS_ROOT, "vqa") if _LABELS_ROOT else ""
COM_DEFAULT = os.path.join(_LABELS_ROOT, "commentary") if _LABELS_ROOT else ""
HPC_ROOT_DEFAULT = os.environ.get("HPC_ROOT", "")

PERSONAS = ("slow", "default", "fast")
COMMENTARY_FUTURE_LEN = 9      # generator cannot label the final 9 frames of a route
FRAME_RE = re.compile(r"^(\d+)\.")


def frames_in(d, suffix):
    """Sorted frame indices present in directory d with the given suffix."""
    if not os.path.isdir(d):
        return []
    out = []
    for f in os.listdir(d):
        if not f.endswith(suffix):
            continue
        m = FRAME_RE.match(f)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def rewrite_paths(obj, src_root, hpc_data_root):
    """Point embedded absolute image paths at HPC3, inserting the `routes/` level.

    local: <src>/<persona>/<driver>/<scenario>/<route>/<town>/rgb/NNNN.jpg
    hpc  : <hpc>/data/simlingo/<persona>/<driver>/routes/<scenario>/<route>/<town>/rgb/NNNN.jpg
    """
    if isinstance(obj, str):
        if obj.startswith(src_root):
            rel = obj[len(src_root):].lstrip("/").split("/")
            if len(rel) >= 3:
                rel = rel[:2] + ["routes"] + rel[2:]
                return hpc_data_root + "/" + "/".join(rel)
        return obj
    if isinstance(obj, dict):
        return {k: rewrite_paths(v, src_root, hpc_data_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_paths(v, src_root, hpc_data_root) for v in obj]
    return obj


def collapse_labels(label_root, persona, driver, scenario, route, town, sub,
                    n_frames, length, src_root, hpc_data_root):
    """Build the frame-indexed array for one label kind, or None if nothing exists."""
    d = os.path.join(label_root, persona, driver, scenario, route, town, sub)
    if not os.path.isdir(d):
        return None
    by_frame = {}
    for f in os.listdir(d):
        m = FRAME_RE.match(f)
        if m and f.endswith(".json.gz"):
            by_frame[int(m.group(1))] = os.path.join(d, f)
    if not by_frame:
        return None
    out = []
    for i in range(length):
        p = by_frame.get(i)
        out.append(rewrite_paths(load_json(p), src_root, hpc_data_root) if p else None)
    return out


def add_bytes(tar, arcname, data, mtime):
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = mtime
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--vqa", default=VQA_DEFAULT)
    ap.add_argument("--commentary", default=COM_DEFAULT)
    ap.add_argument("--out", required=True, help="directory to write per-driver tars into")
    ap.add_argument("--hpc-root", default=HPC_ROOT_DEFAULT)
    ap.add_argument("--drivers", nargs="*", default=None,
                    help="default: every driverN found under the persona dirs")
    ap.add_argument("--town", default="Town12_Rep0_None")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    missing = [n for n, v in (("--src (or $DATASET_ROOT)", args.src),
                              ("--hpc-root (or $HPC_ROOT)", args.hpc_root)) if not v]
    if missing:
        sys.exit("ERROR: not set: " + ", ".join(missing))

    src = args.src.rstrip("/")
    hpc_data_root = args.hpc_root.rstrip("/") + "/data/simlingo"
    os.makedirs(args.out, exist_ok=True)

    drivers = args.drivers
    if not drivers:
        found = set()
        for p in PERSONAS:
            d = os.path.join(src, p)
            if os.path.isdir(d):
                found |= {x for x in os.listdir(d) if x.startswith("driver")}
        drivers = sorted(found, key=lambda s: int(s.replace("driver", "")))
    print(f"drivers: {', '.join(drivers)}")
    print(f"source : {src}")
    print(f"hpc    : {args.hpc_root}\n")

    manifest = []
    grand = {"routes": 0, "frames": 0, "jpgs": 0, "vqa": 0, "com": 0}

    for driver in drivers:
        tar_path = os.path.join(args.out, f"{driver}.tar")
        stats = {"routes": 0, "frames": 0, "jpgs": 0, "vqa": 0, "com": 0}
        t0 = time.time()
        tar = None if args.dry_run else tarfile.open(tar_path, "w")

        for persona in PERSONAS:
            base = os.path.join(src, persona, driver)
            if not os.path.isdir(base):
                continue
            for scenario in sorted(os.listdir(base)):
                sdir = os.path.join(base, scenario)
                if not os.path.isdir(sdir):
                    continue
                for route in sorted(os.listdir(sdir)):
                    rdir = os.path.join(sdir, route)
                    town_dir = os.path.join(rdir, args.town)
                    if not os.path.isdir(town_dir):
                        continue

                    box_frames = frames_in(os.path.join(town_dir, "boxes"), ".json.gz")
                    if not box_frames:
                        print(f"  SKIP (no boxes): {persona}/{driver}/{scenario}/{route}")
                        continue
                    n = len(box_frames)

                    stem = f"{persona}/{driver}/routes/{scenario}/{route}"
                    dst_route = f"data/simlingo/{stem}"
                    dst_town = f"{dst_route}/{args.town}"
                    mtime = int(os.path.getmtime(town_dir))

                    # --- collapsed boxes + measurements (expert controls) ---
                    boxes = [load_json(os.path.join(town_dir, "boxes", f"{i:04d}.json.gz"))
                             for i in range(n)]
                    meas = [load_json(os.path.join(town_dir, "measurements", f"{i:04d}.json.gz"))
                            for i in range(n)]
                    if not args.dry_run:
                        add_bytes(tar, f"{dst_town}/boxes.json",
                                  json.dumps(boxes).encode(), mtime)
                        add_bytes(tar, f"{dst_town}/measurements.json",
                                  json.dumps(meas).encode(), mtime)

                    # --- images, streamed straight from source (lidar dropped) ---
                    for sub in ("rgb", "rgb_augmented"):
                        sd = os.path.join(town_dir, sub)
                        if not os.path.isdir(sd):
                            continue
                        for f in sorted(os.listdir(sd)):
                            if not f.endswith(".jpg"):
                                continue
                            if not args.dry_run:
                                tar.add(os.path.join(sd, f), arcname=f"{dst_town}/{sub}/{f}")
                            stats["jpgs"] += 1

                    # --- result.json at route level ---
                    rj = os.path.join(rdir, "result.json")
                    if os.path.isfile(rj) and not args.dry_run:
                        tar.add(rj, arcname=f"{dst_route}/result.json")

                    # --- collapsed labels ---
                    vqa = collapse_labels(args.vqa, persona, driver, scenario, route,
                                          args.town, "vqa", n, n, src, hpc_data_root)
                    if vqa is not None:
                        if not args.dry_run:
                            add_bytes(tar, f"drivelm/simlingo/{stem}/{args.town}/vqa.json",
                                      json.dumps(vqa).encode(), mtime)
                        stats["vqa"] += sum(1 for x in vqa if x is not None)

                    com_len = max(n - COMMENTARY_FUTURE_LEN, 0)
                    com = collapse_labels(args.commentary, persona, driver, scenario, route,
                                          args.town, "commentary", n, com_len, src, hpc_data_root)
                    if com is not None:
                        if not args.dry_run:
                            add_bytes(tar, f"commentary/simlingo/{stem}/{args.town}/commentary.json",
                                      json.dumps(com).encode(), mtime)
                        stats["com"] += sum(1 for x in com if x is not None)

                    manifest.append({"driver": driver, "persona": persona,
                                     "scenario": scenario, "route": route,
                                     "frames": n,
                                     "vqa_labels": stats["vqa"], "com_len": com_len})
                    stats["routes"] += 1
                    stats["frames"] += n

        if tar:
            tar.close()
        size = os.path.getsize(tar_path) / 2**30 if (tar and os.path.exists(tar_path)) else 0
        print(f"{driver:9} {stats['routes']:3} routes  {stats['frames']:6} frames  "
              f"{stats['jpgs']:6} jpgs  vqa={stats['vqa']:6} com={stats['com']:6}  "
              f"{size:5.2f} GiB  {time.time()-t0:5.0f}s")
        for k in grand:
            grand[k] += stats[k]

    print(f"\nTOTAL: {grand['routes']} routes, {grand['frames']} frames, "
          f"{grand['jpgs']} jpgs, {grand['vqa']} vqa labels, {grand['com']} commentary labels")

    if not args.dry_run:
        mpath = os.path.join(args.out, "manifest.json")
        with open(mpath, "w") as fh:
            json.dump({"hpc_root": args.hpc_root, "drivers": drivers,
                       "totals": grand, "routes": manifest}, fh, indent=1)
        print(f"manifest -> {mpath}")
        # checksums for far-side verification
        spath = os.path.join(args.out, "SHA256SUMS")
        with open(spath, "w") as fh:
            for d in drivers:
                p = os.path.join(args.out, f"{d}.tar")
                if not os.path.exists(p):
                    continue
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                fh.write(f"{h.hexdigest()}  {d}.tar\n")
        print(f"checksums -> {spath}")


if __name__ == "__main__":
    sys.exit(main())
