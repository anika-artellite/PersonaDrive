#!/usr/bin/env python3
"""
embed_merge.py - Merge SLURM shard outputs into unified embedding files
=============================================================================
Reads all shard directories produced by embed_single.py and concatenates them
into the format expected by embed_mine.py (the main pipeline).

Usage:
    python embed_merge.py \
        --shards_dir embeddings_shards \
        --output_dir embeddings \
        [--shuffle]

Reads from each shard directory:
    shard_meta.json
    text_embeddings.pt
    image_embeddings.pt

Writes to --output_dir:
    text_embeddings.pt    - [N_total, D_text]
    image_embeddings.pt   - [N_total, D_img]
    frame_ids.json        - full metadata index

After merging, point the main pipeline's EMB_DIR at --output_dir and the
embedding stages will cache-hit automatically.
"""

import os
import sys
import json
import random
import argparse
import logging
import time
from pathlib import Path

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge embedding shards")
    parser.add_argument("--shards_dir", type=str, required=True,
                        help="Parent dir containing shard subdirectories")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir for merged embeddings (= EMB_DIR for main pipeline)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle entries after merging (recommended)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify embedding shapes match metadata counts")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("merge_embeddings")

    shards_dir = Path(args.shards_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Discover shards
    shard_dirs = sorted(
        d for d in shards_dir.iterdir()
        if d.is_dir() and (d / "shard_meta.json").exists()
    )

    if not shard_dirs:
        log.error(f"No shard directories found in {shards_dir}")
        sys.exit(1)

    log.info(f"Found {len(shard_dirs)} shards in {shards_dir}")

    # Load all shards
    all_text_embs  = []
    all_image_embs = []
    all_frame_ids      = []
    all_image_paths    = []
    all_sentences      = []
    all_actions        = []
    all_vehicle_states = []

    total_entries = 0
    total_missing = 0
    failed_shards = []

    for shard_dir in shard_dirs:
        meta_path = shard_dir / "shard_meta.json"
        text_path = shard_dir / "text_embeddings.pt"
        img_path  = shard_dir / "image_embeddings.pt"

        # Check all files exist
        missing_files = []
        for p in [meta_path, text_path, img_path]:
            if not p.exists():
                missing_files.append(p.name)
        if missing_files:
            log.warning(f"Shard {shard_dir.name}: missing {missing_files} — skipping")
            failed_shards.append(shard_dir.name)
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        text_embs  = torch.load(text_path, map_location="cpu")
        image_embs = torch.load(img_path,  map_location="cpu")

        n = meta["n_entries"]

        # Verify shapes
        if args.verify or text_embs.shape[0] != n or image_embs.shape[0] != n:
            if text_embs.shape[0] != n:
                log.error(f"Shard {shard_dir.name}: text_embs has {text_embs.shape[0]} rows "
                          f"but meta says {n} — skipping")
                failed_shards.append(shard_dir.name)
                continue
            if image_embs.shape[0] != n:
                log.error(f"Shard {shard_dir.name}: image_embs has {image_embs.shape[0]} rows "
                          f"but meta says {n} — skipping")
                failed_shards.append(shard_dir.name)
                continue

        all_text_embs.append(text_embs)
        all_image_embs.append(image_embs)
        all_frame_ids.extend(meta["frame_ids"])
        all_image_paths.extend(meta["image_paths"])
        all_sentences.extend(meta.get("sentences", []))
        all_actions.extend(meta["actions"])
        all_vehicle_states.extend(meta["vehicle_states"])

        total_entries += n
        total_missing += meta.get("missing_paths", 0)
        log.info(f"  {shard_dir.name:<40s}  entries={n:>6d}  "
                 f"text={tuple(text_embs.shape)}  img={tuple(image_embs.shape)}")

    if not all_text_embs:
        log.error("No valid shards found — nothing to merge")
        sys.exit(1)

    # Concatenate
    log.info(f"Concatenating {len(all_text_embs)} shards  |  total={total_entries} entries")

    merged_text  = torch.cat(all_text_embs,  dim=0)
    merged_image = torch.cat(all_image_embs, dim=0)

    assert merged_text.shape[0]  == total_entries, \
        f"Text shape mismatch: {merged_text.shape[0]} vs {total_entries}"
    assert merged_image.shape[0] == total_entries, \
        f"Image shape mismatch: {merged_image.shape[0]} vs {total_entries}"

    # Optional shuffle
    if args.shuffle:
        log.info("Shuffling merged entries…")
        perm = list(range(total_entries))
        random.shuffle(perm)
        perm_t = torch.tensor(perm, dtype=torch.long)

        merged_text  = merged_text[perm_t]
        merged_image = merged_image[perm_t]
        all_frame_ids      = [all_frame_ids[i]      for i in perm]
        all_image_paths    = [all_image_paths[i]    for i in perm]
        all_sentences      = [all_sentences[i]      for i in perm] if all_sentences else []
        all_actions        = [all_actions[i]        for i in perm]
        all_vehicle_states = [all_vehicle_states[i] for i in perm]

    # Save
    text_out = output_dir / "text_embeddings.pt"
    img_out  = output_dir / "image_embeddings.pt"
    meta_out = output_dir / "frame_ids.json"

    torch.save(merged_text,  text_out)
    torch.save(merged_image, img_out)
    log.info(f"Text embeddings   →  {text_out}   shape={tuple(merged_text.shape)}")
    log.info(f"Image embeddings  →  {img_out}    shape={tuple(merged_image.shape)}")

    with open(meta_out, "w") as f:
        json.dump({
            "frame_ids":      all_frame_ids,
            "image_paths":    all_image_paths,
            "actions":        all_actions,
            "vehicle_states": all_vehicle_states,
        }, f, indent=2)
    log.info(f"Frame index       →  {meta_out}")

    # Optionally save sentences too (useful for debugging)
    if all_sentences:
        sent_out = output_dir / "sentences.json"
        with open(sent_out, "w") as f:
            json.dump(all_sentences, f)
        log.info(f"Sentences         →  {sent_out}")

    # Summary
    elapsed = time.perf_counter() - t0
    log.info("=" * 60)
    log.info(f"Merge complete  |  {elapsed:.1f}s")
    log.info(f"  Shards merged    : {len(all_text_embs)}")
    if failed_shards:
        log.warning(f"  Shards failed    : {len(failed_shards)}  ({', '.join(failed_shards)})")
    log.info(f"  Total entries    : {total_entries}")
    log.info(f"  Missing paths    : {total_missing}")
    log.info(f"  Text dim         : {merged_text.shape[1]}")
    log.info(f"  Image dim        : {merged_image.shape[1]}")
    log.info(f"  Shuffled         : {args.shuffle}")
    log.info(f"  Output dir       : {output_dir}")
    log.info("=" * 60)
    log.info("")
    log.info("Next step: run the main pipeline with EMB_DIR pointed at the output dir.")
    log.info("The embedding stages will cache-hit and skip straight to similarity + mining.")


if __name__ == "__main__":
    main()