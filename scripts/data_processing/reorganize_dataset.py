#!/usr/bin/env python3
"""
Reorganizes the simlingo dataset to match the lb1_split directory structure.

Current:
  database/simlingo/data/simlingo/{split}/{scenario}/{scen_folder}/{rep_folder}/
    boxes/, measurements/, rgb/, vqa/, commentary/, ...
  database/simlingo/data/simlingo/{split}/{scenario}/{scen_folder}/result.json

Target:
  database/simlingo/{split}/data/simlingo/lb1_split/routes_training/{scenario}/{scen_folder}/{rep_folder}/
    boxes/, measurements/, rgb/, ...
  database/simlingo/{split}/data/simlingo/lb1_split/routes_training/{scenario}/{scen_folder}/result.json

  database/simlingo/{split}/drivelm/simlingo/lb1_split/routes_training/{scenario}/{scen_folder}/{rep_folder}/vqa/

  database/simlingo/{split}/commentary/simlingo/lb1_split/routes_training/{scenario}/{scen_folder}/{rep_folder}/commentary/
"""

import os
import shutil
import glob
import argparse
from tqdm import tqdm


def move(src, dst, dry_run):
    if dry_run:
        print(f"  MOVE: {src}\n     -> {dst}")
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def reorganize(base_dir, dry_run):
    data_root = os.path.join(base_dir, 'data', 'simlingo')

    for split in ['fast_dataset', 'slow_dataset']:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping missing split: {split_dir}")
            continue

        # Roots for the three output trees - split is now directly under base_dir
        split_root    = os.path.join(base_dir, split)
        new_data_root = os.path.join(split_root, 'data', 'simlingo', 'lb1_split', 'routes_training')
        new_vqa_root  = os.path.join(split_root, 'drivelm', 'simlingo', 'lb1_split', 'routes_training')
        new_com_root  = os.path.join(split_root, 'commentary', 'simlingo', 'lb1_split', 'routes_training')

        # Collect rep folders: {scenario}/{scen_folder}/{rep_folder}
        rep_folders = sorted(glob.glob(os.path.join(split_dir, '*/*/*')))
        rep_folders = [p for p in rep_folders if os.path.isdir(p) and 'lb1_split' not in p]

        print(f"\n{split}: {len(rep_folders)} rep folders")

        for rep_path in tqdm(rep_folders, desc=split):
            rel = os.path.relpath(rep_path, split_dir)
            parts = rel.split(os.sep)
            if len(parts) != 3:
                continue
            scenario, scen_folder, rep_folder = parts

            # 1. Move vqa/ -> {split}/drivelm tree
            vqa_src = os.path.join(rep_path, 'vqa')
            if os.path.isdir(vqa_src):
                vqa_dst = os.path.join(new_vqa_root, scenario, scen_folder, rep_folder, 'vqa')
                move(vqa_src, vqa_dst, dry_run)

            # 2. Move commentary/ -> {split}/commentary tree
            com_src = os.path.join(rep_path, 'commentary')
            if os.path.isdir(com_src):
                com_dst = os.path.join(new_com_root, scenario, scen_folder, rep_folder, 'commentary')
                move(com_src, com_dst, dry_run)

            # 3. Move rep folder -> {split}/data tree
            new_rep = os.path.join(new_data_root, scenario, scen_folder, rep_folder)
            move(rep_path, new_rep, dry_run)

        # 4. Move result.json (at scen_folder level)
        result_files = glob.glob(os.path.join(split_dir, '*/*/result.json'))
        result_files = [f for f in result_files if 'lb1_split' not in f]
        for result_file in result_files:
            rel = os.path.relpath(result_file, split_dir)
            parts = rel.split(os.sep)
            scenario, scen_folder = parts[0], parts[1]
            new_result = os.path.join(new_data_root, scenario, scen_folder, 'result.json')
            move(new_result, new_result, dry_run) if dry_run else None
            if not dry_run:
                os.makedirs(os.path.dirname(new_result), exist_ok=True)
                shutil.move(result_file, new_result)
            else:
                print(f"  MOVE: {result_file}\n     -> {new_result}")

        # 5. Clean up empty leftover dirs in the old split_dir
        if not dry_run:
            for scen_dir in glob.glob(os.path.join(split_dir, '*')):
                if not os.path.isdir(scen_dir) or 'lb1_split' in scen_dir:
                    continue
                for scen_folder_dir in glob.glob(os.path.join(scen_dir, '*')):
                    if os.path.isdir(scen_folder_dir) and not os.listdir(scen_folder_dir):
                        os.rmdir(scen_folder_dir)
                remaining = [f for f in os.listdir(scen_dir) if f != '.DS_Store']
                if not remaining:
                    shutil.rmtree(scen_dir)

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=True,
                        help='Path to database/simlingo/ root')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print moves without executing them')
    args = parser.parse_args()
    reorganize(args.base_dir, args.dry_run)
