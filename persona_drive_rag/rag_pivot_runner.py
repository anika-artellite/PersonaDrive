#!/usr/bin/env python3
"""
Process pivot-based entries from trimmed JSON files in a folder.

Each JSON file contains a list of dicts:
  {"path": "data/simlingo/<split>/routes_<xxx>/<scenario>/<route>",
   "start": int, "end": int, "pivot": int}

Only the pivot frame is processed; sliding window is disabled.

Usage:
  # Single file (Slurm mode - one job per file):
  python rag_pivot_runner.py --json-file /path/to/CHANGE_LANE_LEFT_trimmed.json

  # Whole folder (local mode):
  python rag_pivot_runner.py --json-folder /path/to/split_jsons/

  # Dispatch one Slurm job per JSON file in the folder:
  python rag_pivot_runner.py --dispatch --json-folder /path/to/split_jsons/
"""
import argparse, json, os, stat, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from rag_data_processor import process_route_data, _USING_ORJSON

# Paths
DATA_ROOT   = os.path.join(os.environ.get("PERSONADRIVE_DATA_ROOT", "/path/to/database").split(":")[0], "simlingo")
RUNNER_DIR  = os.path.dirname(os.path.abspath(__file__))
JSON_FOLDER = os.path.join(RUNNER_DIR, "split_jsons")


# Entry parser
def parse_entry(entry: dict, data_root: str) -> dict:
    """
    Parse one JSON entry into kwargs for process_route_data.

    path format: data/simlingo/<split>/routes_<xxx>/<scenario>/<route_folder>
    """
    rel_path = entry["path"]
    parts = rel_path.split("/")
    # ['data', 'simlingo', '<split>', 'routes_<xxx>', '<scenario>', '<route>']
    split_name      = parts[2]
    scenario        = parts[4]
    route_folder    = parts[5]
    base_route_path = os.path.join(data_root, rel_path)

    pivot_frame    = entry["pivot"]
    future_horizon = entry["end"] - entry["pivot"]   # e.g. 67 - 57 = 10

    return dict(
        base_route_path=base_route_path,
        scenario=scenario,
        run=route_folder,
        start_frame=entry["start"],
        future_horizon=future_horizon,
        frame_step=1,           # irrelevant in pivot mode
        data_root=data_root,
        split_name=split_name,
        pivot_frame=pivot_frame,
    )


# Slurm dispatcher
def dispatch_slurm(json_folder: str, data_root: str, runner_dir: str):
    """Submit one Slurm job per JSON file found in json_folder."""

    if not os.path.isdir(json_folder):
        raise FileNotFoundError(f"JSON folder not found: {json_folder}")

    slurm_logs = os.path.join(runner_dir, "slurm_logs")
    os.makedirs(slurm_logs, exist_ok=True)

    submitted_count = 0
    job_ids = []

    print("================================================================")
    print("Dispatching RAG Pivot Processing Jobs (one job per JSON file)")
    print(f"DATA_ROOT  : {data_root}")
    print(f"JSON_FOLDER: {json_folder}")
    print("================================================================\n")

    for fname in sorted(os.listdir(json_folder)):
        if not fname.endswith(".json"):
            continue

        json_file = os.path.join(json_folder, fname)
        job_name  = f"pivot_{fname.replace('.json', '')}"
        out_log   = os.path.join(slurm_logs, f"{job_name}_%j.out")
        err_log   = os.path.join(slurm_logs, f"{job_name}_%j.err")

        script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={out_log}
#SBATCH --error={err_log}
#SBATCH --time=4:00:00
#SBATCH -A {os.environ.get('PERSONADRIVE_SLURM_ACCOUNT', 'YOUR_SLURM_ACCOUNT')}
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "JSON file: {fname}"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================================================"

source ~/.bashrc
conda activate {os.environ.get('PERSONADRIVE_CONDA_ENV', '/path/to/conda/envs/personadrive')}

echo "Python: $(which python)"
echo "Python version: $(python --version)"

cd "{runner_dir}"

python rag_pivot_runner.py \\
  --json-file  "{json_file}" \\
  --data-root  "{data_root}" \\
  --num-workers 8

echo "================================================================"
echo "Job completed at: $(date)"
echo "================================================================"
"""

        temp_slurm = os.path.join(slurm_logs, f"temp_{job_name}.slurm")
        with open(temp_slurm, "w") as f:
            f.write(script)

        print(f"  Submitting: {fname}")
        result = subprocess.run(["sbatch", temp_slurm], capture_output=True, text=True)
        output = result.stdout.strip() + result.stderr.strip()

        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            job_ids.append(job_id)
            print(f"    Job ID: {job_id}")
            submitted_count += 1
        else:
            print(f"    FAILED: {output}")

    print("\n================================================================")
    print("DISPATCH SUMMARY")
    print("================================================================")
    print(f"Jobs submitted: {submitted_count}")
    if job_ids:
        print(f"\nMonitor : squeue -u $USER")
        print(f"Cancel  : scancel {' '.join(job_ids)}")
    print("================================================================")


# Processing logic
def load_entries_from_file(fpath: str) -> list[tuple[str, dict]]:
    """Load (source_name, entry) pairs from a single JSON file."""
    fname = os.path.basename(fpath)
    with open(fpath) as f:
        entries = json.load(f)
    print(f"  {fname}: {len(entries)} entries")
    source = fname.replace(".json", "")
    return [(source, e) for e in entries]


def process_entries(all_entries: list[tuple[str, dict]], data_root: str,
                    num_workers: int, output_dir: str):
    """Run process_route_data for every entry and save results."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nTotal entries: {len(all_entries)} | workers: {num_workers} | orjson: {_USING_ORJSON}")

    all_records: list[dict] = []
    records_by_file: dict[str, list] = {}
    global_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_key = {}
        for source_file, entry in all_entries:
            kwargs = parse_entry(entry, data_root)
            fut = executor.submit(process_route_data, **kwargs)
            future_to_key[fut] = (source_file, entry["pivot"], entry["path"])

        completed = 0
        for future in as_completed(future_to_key):
            source_file, pivot, path = future_to_key[future]
            try:
                records = future.result()
                all_records.extend(records)
                records_by_file.setdefault(source_file, []).extend(records)
                completed += 1
                elapsed = time.perf_counter() - global_start
                print(f"[{completed}/{len(all_entries)}] pivot={pivot:04d} | {path} | "
                      f"records={len(records)} | {elapsed:.1f}s")
            except Exception as e:
                print(f"  ERROR pivot={pivot} {path}: {e}")

    # Save one combined + one per source file
    combined_path = os.path.join(output_dir, "all_pivots_processed.json")
    with open(combined_path, "w") as f:
        json.dump(all_records, f, indent=4)
    print(f"\nSaved {len(all_records)} total records → {combined_path}")

    for source_file, records in records_by_file.items():
        out = os.path.join(output_dir, f"{source_file}_processed.json")
        with open(out, "w") as f:
            json.dump(records, f, indent=4)
        print(f"  {source_file}: {len(records)} records → {out}")


# CLI
def main():
    parser = argparse.ArgumentParser(
        description="Process pivot frames from trimmed JSON files, with optional Slurm dispatch."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--json-file",   help="Process a single JSON file (used by each Slurm job).")
    mode.add_argument("--json-folder", help="Process all JSON files in this folder (local mode).")
    mode.add_argument("--dispatch",    action="store_true",
                      help="Dispatch one Slurm job per JSON file in --json-folder (or default JSON_FOLDER).")

    parser.add_argument("--data-root",   default=DATA_ROOT)
    parser.add_argument("--runner-dir",  default=RUNNER_DIR)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir",  default=None)

    # --json-folder is also needed when --dispatch is used
    parser.add_argument("--folder-for-dispatch", default=JSON_FOLDER,
                        help="JSON folder used when --dispatch is set (default: JSON_FOLDER constant).")

    args = parser.parse_args()

    # Dispatch mode
    if args.dispatch:
        folder = args.json_folder or args.folder_for_dispatch
        dispatch_slurm(
            json_folder=folder,
            data_root=args.data_root,
            runner_dir=args.runner_dir,
        )
        return

    # Processing mode
    output_dir  = args.output_dir or os.path.join(RUNNER_DIR, "processed_pivots")
    num_workers = args.num_workers or min(8, os.cpu_count() or 1)

    if args.json_file:
        # Single file - used by each Slurm job
        all_entries = load_entries_from_file(args.json_file)
    else:
        # Whole folder - local/debug mode
        all_entries = []
        for fname in sorted(os.listdir(args.json_folder)):
            if fname.endswith(".json"):
                all_entries.extend(
                    load_entries_from_file(os.path.join(args.json_folder, fname))
                )

    process_entries(all_entries, args.data_root, num_workers, output_dir)


if __name__ == "__main__":
    main()