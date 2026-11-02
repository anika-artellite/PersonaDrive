#!/usr/bin/env python3
"""
Full data pipeline:
  1. Generate VQA labels
  2. Generate commentary labels
  3. Reorganize into lb1_split directory structure
  4. (Optional) Tar the result

Usage:
  python scripts/data_processing/run_pipeline.py --base-dir /path/to/database/simlingo [--tar-output /path/to/out.tar.gz]

The script derives all intermediate paths from --base-dir:
  data-directory  = {base-dir}
  output-directory = {base-dir}/data
"""

import os
import sys
import subprocess
import argparse
import shutil
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

VQA_SCRIPT        = os.path.join(REPO_ROOT, 'dataset_generation/language_labels/drivelm/carla_vqa_generator_main.py')
COMMENTARY_SCRIPT = os.path.join(REPO_ROOT, 'dataset_generation/language_labels/commentary/carla_commentary_generator_main.py')
REORG_SCRIPT      = os.path.join(SCRIPT_DIR, 'reorganize_dataset.py')


def run(cmd, step_name):
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    print(f"CMD: {' '.join(cmd)}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[ERROR] {step_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[OK] {step_name} completed in {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description='Full simlingo labelling pipeline')
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Path to database/simlingo/ root')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of parallel workers (default: 16)')
    parser.add_argument('--tar-output', type=str, default=None,
                        help='If set, tar the base-dir parent (database/) to this path when done')
    parser.add_argument('--skip-vqa', action='store_true',
                        help='Skip VQA generation (e.g. already done)')
    parser.add_argument('--skip-commentary', action='store_true',
                        help='Skip commentary generation (e.g. already done)')
    parser.add_argument('--skip-reorg', action='store_true',
                        help='Skip reorganization step')
    args = parser.parse_args()

    base_dir    = os.path.abspath(args.base_dir)
    output_dir  = os.path.join(base_dir, 'data')

    print(f"Base dir    : {base_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Workers     : {args.num_workers}")

    # ------------------------------------------------------------------ #
    # Step 1 - VQA generation
    # ------------------------------------------------------------------ #
    if not args.skip_vqa:
        run([
            sys.executable, VQA_SCRIPT,
            '--data-directory',  base_dir,
            '--output-directory', output_dir,
        ], 'VQA generation')
    else:
        print("\n[SKIP] VQA generation")

    # ------------------------------------------------------------------ #
    # Step 2 - Commentary generation
    # ------------------------------------------------------------------ #
    if not args.skip_commentary:
        run([
            sys.executable, COMMENTARY_SCRIPT,
            '--data-directory',  base_dir,
            '--output-directory', output_dir,
        ], 'Commentary generation')
    else:
        print("\n[SKIP] Commentary generation")

    # ------------------------------------------------------------------ #
    # Step 3 - Reorganize
    # ------------------------------------------------------------------ #
    if not args.skip_reorg:
        run([
            sys.executable, REORG_SCRIPT,
            '--base-dir', base_dir,
        ], 'Dataset reorganization')
    else:
        print("\n[SKIP] Dataset reorganization")

    # ------------------------------------------------------------------ #
    # Step 4 - (Optional) Tar
    # ------------------------------------------------------------------ #
    if args.tar_output:
        database_dir    = os.path.dirname(base_dir)          # .../database
        database_parent = os.path.dirname(database_dir)      # parent of database/
        database_name   = os.path.basename(database_dir)     # "database"
        print(f"\n{'='*60}")
        print(f"  STEP: Tar archive -> {args.tar_output}")
        print(f"{'='*60}")
        start = time.time()
        result = subprocess.run([
            'tar', '-czf', args.tar_output,
            '-C', database_parent,
            database_name,
            '--checkpoint=1000', '--checkpoint-action=dot'
        ])
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"\n[ERROR] Tar failed")
            sys.exit(result.returncode)
        size_gb = os.path.getsize(args.tar_output) / 1e9
        print(f"\n[OK] Archive created: {args.tar_output} ({size_gb:.1f} GB) in {elapsed/60:.1f} min")

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
