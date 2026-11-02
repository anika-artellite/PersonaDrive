#!/usr/bin/env python3
"""
embed_single.py - SLURM-friendly per-JSON-file embedding generator
====================================================================
Processes ONE JSON file and produces partial embeddings in an output shard
directory.  Designed to be launched as a SLURM array job - each array task
handles one JSON file.

Usage:
    python embed_single.py \
        --json_file processed_routes/route_042.json \
        --output_dir embeddings_shards/route_042 \
        [--data_root /path/to/simlingo_downloads] \
        [--device cuda] \
        [--batch_size_img 512] \
        [--batch_size_txt 512]

Outputs (in --output_dir):
    shard_meta.json          - frame_ids, image_paths, actions, vehicle_states, sentences
    text_embeddings.pt       - [N_shard, D_text] float32  (L2-normalised)
    image_embeddings.pt      - [N_shard, D_img]  float32  (L2-normalised)
    embed_single.log         - per-shard log

These shards are merged by embed_merge.py before running the main pipeline.
"""

import os
import sys
import json
import re
import logging
import argparse
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ENV / DEFAULTS
# PERSONADRIVE_HF_CACHE unset: Hugging Face's default cache
if os.environ.get("PERSONADRIVE_HF_CACHE"):
    os.environ.setdefault("HF_HOME", os.environ["PERSONADRIVE_HF_CACHE"])
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

SIGLIP_MODEL = "google/siglip-base-patch16-224"
BGE_MODEL    = "BAAI/bge-m3"
HF_CACHE     = os.environ["HF_HOME"] if os.environ.get("PERSONADRIVE_HF_CACHE") else None

NUM_DATALOADER_WORKERS = 4


# HELPERS  (same as main pipeline)
def extract_action(sentence: str) -> str:
    m = re.search(r"Current Command:\s*(.+)", sentence)
    return m.group(1).strip() if m else "unknown"


def extract_vehicle_state(sentence: str) -> dict:
    state = {}
    for key, pattern in [
        ("speed",    r"Speed: ([\d.]+)"),
        ("throttle", r"Throttle: ([\d.]+)"),
        ("steer",    r"Steer: ([-.\d]+)"),
    ]:
        m = re.search(pattern, sentence)
        state[key] = float(m.group(1)) if m else 0.0

    m = re.search(r"Brake: (.+)", sentence)
    state["brake"] = bool(m and "not clicked" not in m.group(1))

    m = re.search(r"Current Command:\s*(.+)", sentence)
    state["command"] = m.group(1).strip() if m else "unknown"
    return state


def resolve_path(raw: str, roots: list[Path]) -> tuple[Path, bool]:
    """Resolve a frame path from JSON against candidate roots."""
    p     = Path(raw)
    parts = p.parts[1:] if p.is_absolute() else p.parts

    for root in roots:
        for n_strip in range(len(parts)):
            suffix    = Path(*parts[n_strip:])
            candidate = root / suffix
            if candidate.exists():
                return candidate, True

    return roots[0] / Path(*parts) if roots else Path(raw), False


# DATASET LOADING
def load_single_json(json_file: Path, data_root: Path | None,
                     strip_prefix: str | None = None):
    """Load a single JSON file and parse its entries.
    
    Args:
        strip_prefix: Optional leading path prefix to remove from frame_path
                      before resolution.  E.g. "database/simlingo" will turn
                      "database/simlingo/data/.../0010.jpg" into "data/.../0010.jpg"
                      so it resolves correctly under a DATA_ROOT that already
                      contains the stripped components.
    """
    with open(json_file) as f:
        data = json.load(f)

    # Build resolution roots
    roots: list[Path] = []
    if data_root is not None:
        roots.append(Path(data_root).resolve())
    roots.append(json_file.resolve().parent)
    roots.append(Path.cwd().resolve())
    # Deduplicate
    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            unique_roots.append(r)

    frame_ids, image_paths, sentences, actions, vehicle_states = [], [], [], [], []
    missing = 0
    first_logged = False

    for entry in data:
        raw_path = entry["frame_path"]
        original_raw = raw_path  # keep for debug

        # Strip leading prefix if requested
        if strip_prefix:
            normalized = strip_prefix.strip("/")
            if raw_path.startswith(normalized + "/"):
                raw_path = raw_path[len(normalized) + 1:]
            elif raw_path.startswith("/" + normalized + "/"):
                raw_path = raw_path[len(normalized) + 2:]

        resolved, found = resolve_path(raw_path, unique_roots)

        # Log first entry for debugging path resolution
        if not first_logged:
            import logging as _log_mod
            _log = _log_mod.getLogger("embed_single")
            _log.info(f"Path debug (first entry):")
            _log.info(f"  original frame_path : {original_raw}")
            _log.info(f"  after strip_prefix  : {raw_path}")
            _log.info(f"  resolved            : {resolved}")
            _log.info(f"  found on disk       : {found}")
            _log.info(f"  roots tried         : {[str(r) for r in unique_roots]}")
            first_logged = True

        if not found:
            missing += 1

        sentence   = entry["sentence"]
        action     = extract_action(sentence)
        frame_path = Path(entry["frame_path"])  # keep original for frame_id
        stem       = frame_path.stem
        route      = entry.get("route_folder", frame_path.parent.parent.name)
        frame_id   = f"{route}/{stem}"

        frame_ids.append(frame_id)
        image_paths.append(str(resolved))
        sentences.append(sentence)
        actions.append(action)
        vehicle_states.append(extract_vehicle_state(sentence))

    return frame_ids, image_paths, sentences, actions, vehicle_states, missing


# TEXT EMBEDDINGS
def embed_text(sentences: list[str], batch_size: int, device: str) -> torch.Tensor:
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(BGE_MODEL, use_fp16=(device == "cuda"))
    n_batches = (len(sentences) + batch_size - 1) // batch_size
    all_embs  = []

    for i in tqdm(range(0, len(sentences), batch_size),
                  desc="Text emb", total=n_batches):
        batch  = sentences[i : i + batch_size]
        output = model.encode(
            batch, batch_size=batch_size, max_length=1024,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        all_embs.append(torch.tensor(output["dense_vecs"], dtype=torch.float32))

    embeddings = torch.cat(all_embs, dim=0)
    return torch.nn.functional.normalize(embeddings, dim=-1)


# IMAGE EMBEDDINGS
class _ImageDataset(Dataset):
    def __init__(self, paths: list[str], processor):
        self.paths     = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


def embed_images(image_paths: list[str], batch_size: int, device: str) -> torch.Tensor:
    from transformers import AutoProcessor, AutoModel

    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, cache_dir=HF_CACHE)
    model     = AutoModel.from_pretrained(SIGLIP_MODEL, cache_dir=HF_CACHE).to(device)
    model.eval()

    dataset    = _ImageDataset(image_paths, processor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=(device == "cuda"),
        prefetch_factor=2,
        persistent_workers=(NUM_DATALOADER_WORKERS > 0),
    )

    all_embs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Image emb"):
            out = model.get_image_features(pixel_values=batch.to(device))
            all_embs.append(out.cpu().float())

    return torch.nn.functional.normalize(torch.cat(all_embs, dim=0), dim=-1)


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for a single JSON shard")
    parser.add_argument("--json_file",      type=str, required=True,
                        help="Path to a single JSON file from processed_routes/")
    parser.add_argument("--output_dir",     type=str, required=True,
                        help="Directory to write shard outputs (created if missing)")
    parser.add_argument("--data_root",      type=str, default=None,
                        help="Root dir containing 'database/' for image path resolution")
    parser.add_argument("--device",         type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--batch_size_img", type=int, default=512)
    parser.add_argument("--batch_size_txt", type=int, default=512)
    parser.add_argument("--strip_prefix",   type=str, default=None,
                        help="Optional leading prefix to strip from JSON frame_path "
                             "before resolution. E.g. 'database/simlingo' strips "
                             "'database/simlingo/data/...' → 'data/...'")
    args = parser.parse_args()

    json_file  = Path(args.json_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    log_path = output_dir / "embed_single.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("embed_single")

    t0 = time.perf_counter()
    log.info(f"embed_single.py  |  {json_file}  →  {output_dir}")
    log.info(f"  device={args.device}  batch_img={args.batch_size_img}  batch_txt={args.batch_size_txt}")

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU")
        args.device = "cpu"

    # 1. Load JSON
    data_root = Path(args.data_root) if args.data_root else None
    if args.strip_prefix:
        log.info(f"  strip_prefix={args.strip_prefix}")
    frame_ids, image_paths, sentences, actions, vehicle_states, missing = \
        load_single_json(json_file, data_root, strip_prefix=args.strip_prefix)
    log.info(f"Loaded {len(frame_ids)} entries  |  missing_paths={missing}")

    # 2. Text embeddings
    log.info("Generating text embeddings (BGE-M3)…")
    text_embs = embed_text(sentences, args.batch_size_txt, args.device)
    text_path = output_dir / "text_embeddings.pt"
    torch.save(text_embs, text_path)
    log.info(f"Text embeddings  |  shape={tuple(text_embs.shape)}  →  {text_path}")

    # 3. Image embeddings
    log.info("Generating image embeddings (SigLIP)…")
    img_embs = embed_images(image_paths, args.batch_size_img, args.device)
    img_path = output_dir / "image_embeddings.pt"
    torch.save(img_embs, img_path)
    log.info(f"Image embeddings  |  shape={tuple(img_embs.shape)}  →  {img_path}")

    # 4. Save metadata
    meta_path = output_dir / "shard_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "source_json":    str(json_file),
            "n_entries":      len(frame_ids),
            "missing_paths":  missing,
            "frame_ids":      frame_ids,
            "image_paths":    image_paths,
            "sentences":      sentences,
            "actions":        actions,
            "vehicle_states": vehicle_states,
        }, f, indent=2)
    log.info(f"Metadata  →  {meta_path}")

    elapsed = time.perf_counter() - t0
    log.info(f"Done  |  {elapsed:.1f}s  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()