"""
Batch convert all *.json.gz frame files across the full dataset into
null-padded JSON arrays (one per run per data type).

Discovers runs under these roots (relative to --dataset-root):
  data/simlingo/       -> measurements subfolder
  dreamer/simlingo/    -> dreamer subfolder
  commentary/simlingo/ -> commentary subfolder
  drivelm/simlingo/    -> vqa subfolder

Already-converted runs (output .json exists) are skipped - safe to re-run.

Usage:
    python convert_all.py [--dataset-root ROOT] [--workers N] [--log FILE]

Examples:
    # Run from dataset root with 8 parallel workers
    python convert_all.py --workers 8

    # With explicit root and log path
    python convert_all.py --dataset-root /mnt/nas/dataset --workers 8 --log convert.log

nohup:
    nohup python convert_all.py --workers 8 --log convert_all.log > convert_all_stdout.log 2>&1 &
    tail -f convert_all.log
"""
from __future__ import print_function

import argparse
import datetime
import logging
import multiprocessing
import os
import sys

from convert_run import convert_run

# (root_subdir, subfolder_name)
DATASET_CONFIGS = [
    (os.path.join('data', 'simlingo'),       'measurements'),
    (os.path.join('dreamer', 'simlingo'),    'dreamer'),
    (os.path.join('commentary', 'simlingo'), 'commentary'),
    (os.path.join('drivelm', 'simlingo'),    'vqa'),
]


def discover_pairs(dataset_root):
    """Walk all dataset roots and return (run_dir, subfolder) pairs to convert."""
    pairs = []
    for rel_root, subfolder in DATASET_CONFIGS:
        abs_root = os.path.join(dataset_root, rel_root)
        if not os.path.isdir(abs_root):
            logging.warning('Root not found, skipping: %s', abs_root)
            continue
        for dirpath, dirnames, _filenames in os.walk(abs_root):
            dirnames.sort()
            if os.path.basename(dirpath) == subfolder:
                run_dir = os.path.dirname(dirpath)
                out_path = os.path.join(run_dir, subfolder + '.json')
                pairs.append((run_dir, subfolder, os.path.exists(out_path)))
    return pairs


def _worker(args):
    """Top-level worker function - must be module-level for multiprocessing pickle."""
    run_dir, subfolder = args
    try:
        status = convert_run(run_dir, subfolder)
    except Exception as e:
        status = 'ERROR: {0}'.format(e)
    return (run_dir, subfolder, status)


def _timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Batch convert dataset JSON frames.')
    parser.add_argument('--dataset-root', default=os.getcwd(),
                        help='Path to dataset root (default: current directory)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel worker processes (default: 4)')
    parser.add_argument('--log', default='convert_all.log',
                        help='Log file path (default: convert_all.log)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log),
            logging.StreamHandler(sys.stdout),
        ]
    )

    logging.info('Dataset root : %s', args.dataset_root)
    logging.info('Workers      : %d', args.workers)
    logging.info('Log file     : %s', args.log)
    logging.info('Discovering runs...')

    all_pairs = discover_pairs(args.dataset_root)
    to_convert = [(rd, sf) for rd, sf, already_done in all_pairs if not already_done]
    n_skip = len(all_pairs) - len(to_convert)

    logging.info('Found %d total pairs: %d to convert, %d already done (skipped)',
                 len(all_pairs), len(to_convert), n_skip)

    if not to_convert:
        logging.info('Nothing to do.')
        return

    n_ok = 0
    n_err = 0

    pool = multiprocessing.Pool(processes=args.workers)
    try:
        for i, (run_dir, subfolder, status) in enumerate(
                pool.imap_unordered(_worker, to_convert), 1):
            if status == 'OK':
                n_ok += 1
            elif status == 'SKIP':
                n_skip += 1
            else:
                n_err += 1
                logging.error('[%d/%d] %s  %s/%s',
                              i, len(to_convert), status, run_dir, subfolder)
                continue
            logging.info('[%d/%d] %s  %s/%s',
                         i, len(to_convert), status, run_dir, subfolder)
    except KeyboardInterrupt:
        logging.warning('Interrupted — partial progress is safe, re-run to resume.')
        pool.terminate()
    finally:
        pool.close()
        pool.join()

    logging.info('Done. OK=%d  SKIP=%d  ERROR=%d', n_ok, n_skip, n_err)


if __name__ == '__main__':
    main()