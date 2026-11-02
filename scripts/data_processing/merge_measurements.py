"""
scripts/data_processing/merge_measurements.py
---------------------
For every route folder under a dataset root that contains a `measurements/`
subfolder with individual per-frame files (0000.json / 0000.json.gz),
merge them in frame order into a single `measurements.json` list and write
it next to the `measurements/` folder (i.e. in the route directory).

Skips any route whose `measurements.json` already exists - safe to re-run.

Usage:
    python scripts/data_processing/merge_measurements.py \
        --data-root /path/to/database/simlingo/data/slow_dataset \
        --workers 8

    # dry-run to see what would be processed without writing anything
    python scripts/data_processing/merge_measurements.py --data-root /path/to/data --dry-run
    
    
    python scripts/data_processing/merge_measurements.py \
  --data-root /path/to/database/simlingo/data/simlingo/slow \
  --workers 8 \
  --log merge_measurements.log

nohup:
    nohup python scripts/data_processing/merge_measurements.py \
        --data-root /path/to/database/simlingo/data/simlingo/slow \
        --workers 8 --log merge_measurements.log \
        > merge_measurements_stdout.log 2>&1 &
"""

import argparse
import gzip
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import time


# ---------------------------------------------------------------------------
# Core worker: merge one route's measurements/ folder -> measurements.json
# ---------------------------------------------------------------------------

def _load_frame(path: str) -> dict:
    """Load a single frame file - supports both .json and .json.gz."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_route(route_dir: str, dry_run: bool = False):
    """
    Merge all per-frame measurement files in <route_dir>/measurements/
    into <route_dir>/measurements.json.

    Returns one of: 'SKIP', 'OK', 'EMPTY', or 'ERROR: <message>'
    """
    out_path  = os.path.join(route_dir, 'measurements.json')
    meas_dir  = os.path.join(route_dir, 'measurements')

    # Already done - skip
    if os.path.exists(out_path):
        return 'SKIP'

    if not os.path.isdir(meas_dir):
        return 'ERROR: measurements/ subfolder not found'

    # Collect all frame files, prefer .json.gz over plain .json for same stem
    stems = {}
    for fname in os.listdir(meas_dir):
        if fname.endswith('.json.gz'):
            stem = fname[:-len('.json.gz')]
            stems[stem] = os.path.join(meas_dir, fname)
        elif fname.endswith('.json') and not fname.endswith('.json.gz'):
            stem = fname[:-len('.json')]
            if stem not in stems:          # .json.gz takes priority
                stems[stem] = os.path.join(meas_dir, fname)

    if not stems:
        return 'EMPTY'

    # Sort numerically by stem name (0000, 0001, ...)
    sorted_paths = [stems[s] for s in sorted(stems)]

    if dry_run:
        return f'DRY-RUN: would merge {len(sorted_paths)} frames'

    # Load all frames
    frames = []
    for path in sorted_paths:
        try:
            frames.append(_load_frame(path))
        except Exception as e:
            return f'ERROR: failed reading {os.path.basename(path)}: {e}'

    # Atomic write: temp file -> rename
    tmp_fd, tmp_path = tempfile.mkstemp(dir=route_dir, suffix='.json.tmp')
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            json.dump(frames, f)
        os.rename(tmp_path, out_path)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return f'ERROR: write failed: {e}'

    return 'OK'


# ---------------------------------------------------------------------------
# Discovery: find all route directories that contain a measurements/ folder
# ---------------------------------------------------------------------------

def discover_routes(data_root: str):
    """
    Walk data_root recursively. Yield every directory that contains a
    `measurements` subdirectory (= a route directory).
    """
    routes = []
    for dirpath, dirnames, _ in os.walk(data_root):
        dirnames.sort()
        if 'measurements' in dirnames:
            routes.append(dirpath)
            dirnames.clear()   # don't recurse further inside a route dir
    return routes


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be top-level for pickle)
# ---------------------------------------------------------------------------

def _worker(args):
    route_dir, dry_run = args
    try:
        status = merge_route(route_dir, dry_run=dry_run)
    except Exception as e:
        status = f'ERROR: {e}'
    return route_dir, status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Merge per-frame measurement .json/.json.gz files into one measurements.json per route.'
    )
    parser.add_argument(
        '--data-root',
        default=os.getcwd(),
        help='Root directory to scan for route folders (default: cwd)',
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel worker processes (default: 4)',
    )
    parser.add_argument(
        '--log', default='merge_measurements.log',
        help='Log file path (default: merge_measurements.log)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would be done without writing any files',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info('Data root  : %s', args.data_root)
    logging.info('Workers    : %d', args.workers)
    logging.info('Dry run    : %s', args.dry_run)
    logging.info('Discovering routes...')

    t0     = time.perf_counter()
    routes = discover_routes(args.data_root)
    logging.info('Found %d route directories in %.1fs', len(routes), time.perf_counter() - t0)

    # Pre-classify: already done vs. to process
    to_process = []
    n_skip_pre = 0
    for r in routes:
        if os.path.exists(os.path.join(r, 'measurements.json')):
            n_skip_pre += 1
        else:
            to_process.append(r)

    logging.info(
        'To merge: %d  |  Already done (skip): %d',
        len(to_process), n_skip_pre,
    )

    if not to_process:
        logging.info('Nothing to do.')
        return

    work_items = [(r, args.dry_run) for r in to_process]

    n_ok = n_skip = n_empty = n_err = 0
    t1 = time.perf_counter()

    pool = multiprocessing.Pool(processes=args.workers)
    try:
        for i, (route_dir, status) in enumerate(
            pool.imap_unordered(_worker, work_items), 1
        ):
            short = os.path.relpath(route_dir, args.data_root)
            if status == 'OK':
                n_ok += 1
                logging.info('[%d/%d] OK      %s', i, len(to_process), short)
            elif status == 'SKIP':
                n_skip += 1
                logging.info('[%d/%d] SKIP    %s', i, len(to_process), short)
            elif status == 'EMPTY':
                n_empty += 1
                logging.warning('[%d/%d] EMPTY   %s', i, len(to_process), short)
            elif status.startswith('DRY-RUN'):
                n_ok += 1
                logging.info('[%d/%d] %s  (%s)', i, len(to_process), status, short)
            else:
                n_err += 1
                logging.error('[%d/%d] %s  (%s)', i, len(to_process), status, short)
    except KeyboardInterrupt:
        logging.warning('Interrupted — partial progress is safe, re-run to resume.')
        pool.terminate()
    finally:
        pool.close()
        pool.join()

    elapsed = time.perf_counter() - t1
    logging.info(
        'Done in %.1fs  |  OK=%d  SKIP=%d  EMPTY=%d  ERROR=%d',
        elapsed, n_ok, n_skip + n_skip_pre, n_empty, n_err,
    )


if __name__ == '__main__':
    main()
