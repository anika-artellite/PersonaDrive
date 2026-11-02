"""
Contrastive training data for the retrieval head: embeddings and triplet mining.

Speed-ups:
  1. Similarity matrix: chunked matmul on the GPU (10-50x faster)
  2. Triplet mining: global boolean masks, no per-anchor Python loop (5-20x faster)

Input:  all_routes_processed.json  (list of {frame_path, sentence, route_folder})
        OR a directory of *.json files (concatenated automatically)
Output: embeddings/text_embeddings.pt
        embeddings/image_embeddings.pt
        embeddings/frame_ids.json
        triplets/triplets.jsonl
        pipeline.log

Can also consume pre-computed embeddings from the distributed SLURM pipeline
(see: embed_single.py + embed_merge.py).  Point EMB_DIR at the merged
output directory and embedding stages will cache-hit automatically.
"""

import os
import hashlib
import sys
import json
import re
import random
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# CONFIG
# DATASET_JSON can point to either:
#   - a single JSON file (original behaviour), or
#   - a directory containing multiple *.json files to be concatenated.
DATASET_JSON = Path("processed_routes")
EMB_DIR      = Path("embeddings")
TRIPLETS_DIR = Path("triplets")
LOG_FILE     = Path("pipeline.log")

# Root directory that relative frame paths in the JSON are anchored to.
# Set DATA_ROOT to the directory that your frame paths resolve under.
# Leave as None to auto-detect from DATASET_JSON's location.
DATA_ROOT: Path | None = Path(os.environ.get("PERSONADRIVE_DATA_ROOT", "/path/to/database").split(":")[0]) / "simlingo"

# Optional: strip a leading prefix from JSON frame_path before joining with
# DATA_ROOT.  E.g. if your JSON has "database/simlingo/data/.../0010.jpg" but
# DATA_ROOT already ends at ".../database/simlingo", set this to
# "database/simlingo" so the resolved path becomes DATA_ROOT/data/.../0010.jpg.
# Set to None or "" to disable.
STRIP_PREFIX: str | None = "database/simlingo"

SIGLIP_MODEL = "google/siglip-base-patch16-224"
BGE_MODEL    = "BAAI/bge-m3"
HF_CACHE     = os.environ.get("PERSONADRIVE_HF_CACHE") or None   # None: Hugging Face's default cache

if HF_CACHE:
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

BATCH_SIZE_IMG = 512
BATCH_SIZE_TXT = 512
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

SIM_DTYPE = torch.float16   # halves VRAM vs float32; set float32 for higher precision

# Optional feature flags
# Set to True to use visual-only hard negatives (visually similar, textually
# different) instead of the default dual-threshold hard negatives.
USE_VISUAL_HARD_NEGATIVES = False

# Set to True to exclude same-route pairs from positives so that trivially
# similar adjacent frames don't dominate the positive pool.
# Requires route_folder in the JSON entries.
EXCLUDE_SAME_ROUTE_POSITIVES = False

# Minimum frame-index gap within the same route for a pair to be considered
# positive.  Only applies when EXCLUDE_SAME_ROUTE_POSITIVES is False.
# Set to 0 to disable temporal filtering entirely.
MIN_TEMPORAL_GAP = 0

# Auto-tune chunk sizes from available VRAM
# Override by setting either value to a positive integer to skip auto-tuning.
SIM_CHUNK_OVERRIDE    = None   # rows per GPU matmul in similarity computation
MINING_CHUNK_OVERRIDE = None   # anchor rows per iteration in triplet mining


def _auto_tune_chunks(vram_gb: float) -> tuple[int, int]:
    """
    Choose SIM_CHUNK and MINING_CHUNK based on available VRAM.

    SIM_CHUNK controls how many rows of the [NxN] similarity matrix are
    computed per CUDA matmul call.  Larger = fewer kernel launches = faster,
    but each call holds 2 x [chunk x N] float16 tensors in VRAM.

    MINING_CHUNK controls how many anchor rows are processed per iteration
    of the triplet mining loop.  Each iteration holds ~6 x [chunk x N]
    tensors (3 float32 sim slices + 3 bool masks).

      VRAM (GB)  |  GPU examples          | SIM_CHUNK | MINING_CHUNK
      -----------+------------------------+-----------+-------------
       < 8       |  RTX 3060, older       |   500     |   100
       8-15      |  RTX 3080, A10 (10G)   |  1000     |   250
      16-23      |  RTX 3090, A10 (24G)   |  2000     |   500
      24-39      |  A100 (40G), RTX 4090  |  4000     |  1000
      >= 40      |  A100 (80G), H100      |  8000     |  2000
    """
    if vram_gb < 8:
        return 500, 100
    elif vram_gb < 16:
        return 1000, 250
    elif vram_gb < 24:
        return 2000, 500
    elif vram_gb < 40:
        return 4000, 1000
    else:
        return 8000, 2000


def _resolve_chunks() -> tuple[int, int]:
    """Return (SIM_CHUNK, MINING_CHUNK), respecting manual overrides."""
    if DEVICE == "cuda":
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated()) / 1e9
        auto_sim, auto_mine = _auto_tune_chunks(free_gb)
    else:
        auto_sim, auto_mine = 256, 64

    sim_chunk  = SIM_CHUNK_OVERRIDE    if SIM_CHUNK_OVERRIDE    else auto_sim
    mine_chunk = MINING_CHUNK_OVERRIDE if MINING_CHUNK_OVERRIDE else auto_mine
    return sim_chunk, mine_chunk


# Resolved once at import time - available as module-level constants
SIM_CHUNK, MINING_CHUNK = _resolve_chunks()

ALPHA = 0.5
BETA  = 0.5

TOP_K_POOL    = 1000
BOTTOM_K_POOL = 200

NUM_POS_PER_ANCHOR      = 6
NUM_NEG_PER_ANCHOR      = 6
NUM_HARD_NEG_PER_ANCHOR = 6

POS_PERCENTILE  = 60
HARD_PERCENTILE = 30
NEG_PERCENTILE  = 30

ADAPTIVE_SAMPLE = 3000
DIAG_SAMPLE     = 500

NUM_DATALOADER_WORKERS = 8

# dirs created inside main() after args are parsed


# LOGGING SETUP
class _ColourFormatter(logging.Formatter):
    COLOURS = {
        logging.DEBUG:    "\033[37m",
        logging.INFO:     "\033[36m",
        logging.WARNING:  "\033[33m",
        logging.ERROR:    "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour    = self.COLOURS.get(record.levelno, "")
        levelname = f"{colour}{record.levelname:8s}{self.RESET}"
        ts        = datetime.now().strftime("%H:%M:%S")
        return f"[{ts}] {levelname} {record.getMessage()}"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("rag_pipeline")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColourFormatter())

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

log = _setup_logger()


# TIMING + VRAM HELPERS
@contextmanager
def stage(name: str, step_bar: tqdm | None = None):
    log.info(f"┌─ START  {name}")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        vram_msg = ""
        if DEVICE == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            curr_gb = torch.cuda.memory_allocated() / 1e9
            vram_msg = f"  |  VRAM peak={peak_gb:.2f}GB curr={curr_gb:.2f}GB"
        log.info(f"└─ DONE   {name}  [{elapsed:.1f}s]{vram_msg}")
        log.debug(f"STAGE_TIMING | {name} | {elapsed:.3f}s")
        if step_bar is not None:
            step_bar.update(1)


def _vram_str() -> str:
    if DEVICE != "cuda":
        return "N/A"
    alloc = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{alloc:.2f}/{total:.2f}GB"


# MASTER PIPELINE PROGRESS BAR
PIPELINE_STAGES = [
    "Load dataset",
    "Text embeddings (BGE-M3)",
    "Image embeddings (SigLIP)",
    "Similarity matrices (GPU)",
    "Diagnose similarities",
    "Adaptive thresholds",
    "Triplet mining",
    "Save outputs",
]

def make_pipeline_bar() -> tqdm:
    return tqdm(
        total=len(PIPELINE_STAGES),
        desc="▶ Pipeline",
        unit="stage",
        bar_format="{l_bar}{bar}| {n}/{total} stages  [{elapsed}<{remaining}]",
        position=0,
        leave=True,
        file=sys.stdout,
    )


# HELPERS
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


def parse_frame_index(frame_id: str) -> int | None:
    """Extract the numeric frame index from a frame_id like 'route_01/0010'."""
    parts = frame_id.rsplit("/", 1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def parse_route(frame_id: str) -> str:
    """Extract route component from frame_id like 'route_01/0010' -> 'route_01'."""
    return frame_id.rsplit("/", 1)[0] if "/" in frame_id else ""


# LOAD DATASET
def load_dataset(json_path: Path):
    """Load one or more route JSON files.

    If ``json_path`` is a file, the original behaviour is preserved.
    If it is a directory, all ``*.json`` files inside are loaded and
    their lists are concatenated.
    """

    if json_path.is_dir():
        json_files = sorted(p for p in json_path.glob("*.json") if p.is_file())
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory: {json_path}")

        log.info(f"Loading dataset from directory: {json_path} ({len(json_files)} files)")
        data = []
        for jf in json_files:
            log.info(f"  + {jf}")
            with open(jf) as f:
                chunk = json.load(f)
            log.debug(f"    entries: {len(chunk)}")
            data.extend(chunk)
        log.info(f"Total concatenated JSON entries: {len(data)}")
    else:
        log.info(f"Loading dataset from {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        log.debug(f"Raw JSON entries: {len(data)}")

    # Shuffle all entries so training does not depend on file/route order
    random.shuffle(data)
    log.info("Shuffled dataset entries")

    frame_ids, image_paths, sentences, actions, vehicle_states = [], [], [], [], []
    script_dir = Path(__file__).parent
    missing = 0

    json_dir   = json_path.resolve().parent if json_path.is_file() else json_path.resolve()
    roots: list[Path] = []
    if DATA_ROOT is not None:
        roots.append(Path(DATA_ROOT).resolve())
    roots += [json_dir, script_dir.resolve(), Path.cwd().resolve()]
    seen_roots: set[Path] = set()
    unique_roots: list[Path] = []
    for r in roots:
        if r not in seen_roots:
            seen_roots.add(r)
            unique_roots.append(r)
    log.debug(f"Path resolution roots (in priority order):")
    for r in unique_roots:
        log.debug(f"  {r}")

    def resolve_path(raw: str) -> tuple[Path, bool]:
        p = Path(raw)

        # Fast path: absolute path that already exists on disk (fast/slow profiles)
        if p.is_absolute() and p.exists():
            return p, True

        parts = p.parts[1:] if p.is_absolute() else p.parts

        for root in unique_roots:
            for n_strip in range(len(parts)):
                suffix    = Path(*parts[n_strip:])
                candidate = root / suffix
                if candidate.exists():
                    if n_strip > 0:
                        log.debug(f"Resolved by stripping {n_strip} leading component(s): {candidate}")
                    return candidate, True

        return unique_roots[0] / Path(*parts), False

    # Precompute normalised strip prefix once
    _strip = STRIP_PREFIX.strip("/") + "/" if STRIP_PREFIX else None
    _strip_slash = "/" + _strip if _strip else None
    if _strip:
        log.info(f"STRIP_PREFIX active: will remove leading '{STRIP_PREFIX}' from frame paths")

    first_logged = False
    for entry in tqdm(data, desc="  Parsing JSON", position=1, leave=False):
        raw_path = entry["frame_path"]

        # Strip leading prefix if configured
        if _strip:
            if raw_path.startswith(_strip):
                raw_path = raw_path[len(_strip):]
            elif _strip_slash and raw_path.startswith(_strip_slash):
                raw_path = raw_path[len(_strip_slash):]

        resolved, found = resolve_path(raw_path)
        if not first_logged:
            log.info(f"Path resolution sample:")
            log.info(f"  JSON raw   : {entry['frame_path']}")
            log.info(f"  Resolved   : {resolved}  ({'found' if found else 'NOT FOUND'})")
            first_logged = True
        if not found:
            missing += 1
            if missing <= 5:
                log.debug(f"Unresolved: {entry['frame_path']}")
            elif missing == 6:
                log.debug("(further unresolved paths suppressed in debug log)")

        sentence = entry["sentence"]
        action   = extract_action(sentence)
        frame_path = Path(entry["frame_path"])
        stem     = frame_path.stem
        # Use run field if available (includes scenario+rep path for uniqueness)
        # Fall back to route_folder, then parent folder name
        route    = entry.get("run",
                   entry.get("route_folder",
                   frame_path.parent.parent.name))
        # Normalise slashes so frame_id is a safe unique key
        route    = route.replace(os.sep, "/")
        frame_id = f"{route}/{stem}"
        # If still not unique (same route/frame in different scenarios),
        # append a short deterministic hash of the full path
        raw_path_str = entry["frame_path"]
        path_hash = hashlib.md5(raw_path_str.encode()).hexdigest()[:6]
        frame_id  = f"{frame_id}_{path_hash}"

        frame_ids.append(frame_id)
        image_paths.append(str(resolved))
        sentences.append(sentence)
        actions.append(action)
        vehicle_states.append(extract_vehicle_state(sentence))

    action_counts = Counter(actions)
    log.info(f"Loaded {len(frame_ids)} frames  |  unique actions: {len(action_counts)}"
             f"  |  unresolved paths: {missing}")
    for action, count in action_counts.most_common():
        log.info(f"  {action:<30s}  {count:>6d}  ({count/len(frame_ids)*100:.1f}%)")

    if missing:
        log.warning(
            f"{missing} frame paths could not be resolved on disk. "
            f"They will still be processed but image loading will fail at embedding time. "
            f"Fix: set DATA_ROOT at the top of the script to the directory that contains "
            f"'{Path(data[0]['frame_path']).parts[0]}/' "
            f"(the first path component in your JSON)."
        )

    return frame_ids, image_paths, sentences, actions, vehicle_states


# TEXT EMBEDDINGS - BGE-M3
def extract_text_embeddings(sentences: list[str], save_path: Path) -> torch.Tensor:
    from FlagEmbedding import BGEM3FlagModel

    log.info(f"Loading BGE-M3  |  device={DEVICE}  VRAM={_vram_str()}")
    model = BGEM3FlagModel(BGE_MODEL, use_fp16=(DEVICE == "cuda"))

    n_batches    = (len(sentences) + BATCH_SIZE_TXT - 1) // BATCH_SIZE_TXT
    all_embeddings = []

    for i in tqdm(range(0, len(sentences), BATCH_SIZE_TXT),
                  desc="  Text embeddings", total=n_batches, position=1, leave=False):
        batch  = sentences[i : i + BATCH_SIZE_TXT]
        output = model.encode(
            batch,
            batch_size=BATCH_SIZE_TXT,
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        emb = torch.tensor(output["dense_vecs"], dtype=torch.float32)
        all_embeddings.append(emb)
        log.debug(f"Text batch {i//BATCH_SIZE_TXT+1}/{n_batches}  shape={tuple(emb.shape)}")

    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    torch.save(embeddings, save_path)
    log.info(f"Text embeddings saved  |  shape={tuple(embeddings.shape)}  →  {save_path}")
    return embeddings.to(DEVICE)


# IMAGE EMBEDDINGS - SigLIP
class ImageDataset(Dataset):
    def __init__(self, paths: list[str], processor):
        self.paths     = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


def extract_image_embeddings(image_paths: list[str], save_path: Path) -> torch.Tensor:
    from transformers import AutoProcessor, AutoModel

    log.info(f"Loading SigLIP  |  device={DEVICE}  VRAM={_vram_str()}")
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, cache_dir=HF_CACHE)
    model     = AutoModel.from_pretrained(SIGLIP_MODEL, cache_dir=HF_CACHE).to(DEVICE)
    model.eval()
    log.info(f"SigLIP ready  |  VRAM={_vram_str()}")

    dataset    = ImageDataset(image_paths, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_IMG,
        num_workers=NUM_DATALOADER_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        prefetch_factor=2,
        persistent_workers=(NUM_DATALOADER_WORKERS > 0),
    )

    n_batches      = len(dataloader)
    all_embeddings = []

    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(
            dataloader, desc="  Image embeddings", position=1, leave=False
        )):
            out = model.get_image_features(pixel_values=batch.to(DEVICE))
            all_embeddings.append(out.cpu().float())
            log.debug(f"Image batch {b_idx+1}/{n_batches}  "
                      f"shape={tuple(out.shape)}  VRAM={_vram_str()}")

    embeddings = torch.nn.functional.normalize(torch.cat(all_embeddings, dim=0), dim=-1)
    torch.save(embeddings, save_path)
    log.info(f"Image embeddings saved  |  shape={tuple(embeddings.shape)}  →  {save_path}")
    return embeddings.to(DEVICE)


# GPU SIMILARITY MATRICES
def compute_similarity_matrices_gpu(
    text_embs: torch.Tensor,
    img_embs:  torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = text_embs.shape[0]

    text_embs = text_embs.to(DEVICE, dtype=SIM_DTYPE)
    img_embs  = img_embs.to(DEVICE,  dtype=SIM_DTYPE)

    bytes_per = 2 if SIM_DTYPE == torch.float16 else 4
    est_gb    = 3 * N * N * bytes_per / 1e9
    log.info(f"Sim matrices  |  N={N}  chunk={SIM_CHUNK}  dtype={SIM_DTYPE}"
             f"  est_VRAM={est_gb:.2f}GB  current={_vram_str()}")

    text_sim  = torch.empty(N, N, dtype=SIM_DTYPE, device=DEVICE)
    image_sim = torch.empty(N, N, dtype=SIM_DTYPE, device=DEVICE)

    n_chunks = (N + SIM_CHUNK - 1) // SIM_CHUNK
    for i in tqdm(range(0, N, SIM_CHUNK),
                  desc="  Sim matrix (GPU)", total=n_chunks, position=1, leave=False):
        end = min(i + SIM_CHUNK, N)
        text_sim[i:end]  = text_embs[i:end] @ text_embs.T
        image_sim[i:end] = img_embs[i:end]  @ img_embs.T
        log.debug(f"Sim chunk {i//SIM_CHUNK+1}/{n_chunks}"
                  f"  rows=[{i}:{end}]  VRAM={_vram_str()}")

    # Compute combined in-place to avoid allocating a 4th [NxN] matrix.
    # With N=54k and fp16, each matrix is ~5.5GB - a 4th would OOM on 24GB GPUs.
    # We reuse a pre-allocated tensor and compute chunk-by-chunk.
    combined = torch.empty(N, N, dtype=SIM_DTYPE, device=DEVICE)
    for i in range(0, N, SIM_CHUNK):
        end = min(i + SIM_CHUNK, N)
        combined[i:end] = ALPHA * text_sim[i:end] + BETA * image_sim[i:end]
    torch.cuda.synchronize()

    # Free the embedding tensors from GPU - no longer needed
    del text_embs, img_embs
    torch.cuda.empty_cache()

    log.info(f"Sim matrices done  |  VRAM={_vram_str()}")
    return combined, text_sim, image_sim


# OPTIONAL: Top-K Only Variant (N > ~40k / VRAM tight)
def compute_topk_only(
    text_embs: torch.Tensor,
    img_embs:  torch.Tensor,
    k_top: int = TOP_K_POOL,
    k_bot: int = BOTTOM_K_POOL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = text_embs.shape[0]
    text_embs = text_embs.to(DEVICE, dtype=SIM_DTYPE)
    img_embs  = img_embs.to(DEVICE,  dtype=SIM_DTYPE)

    topk_vals = torch.empty(N, k_top, dtype=SIM_DTYPE, device=DEVICE)
    topk_idx  = torch.empty(N, k_top, dtype=torch.int32, device=DEVICE)
    botk_vals = torch.empty(N, k_bot, dtype=SIM_DTYPE, device=DEVICE)
    botk_idx  = torch.empty(N, k_bot, dtype=torch.int32, device=DEVICE)

    log.info(f"TopK-only sim  |  N={N}  k_top={k_top}  k_bot={k_bot}")
    n_chunks = (N + SIM_CHUNK - 1) // SIM_CHUNK
    for i in tqdm(range(0, N, SIM_CHUNK),
                  desc="  TopK sim (GPU)", total=n_chunks, position=1, leave=False):
        end   = min(i + SIM_CHUNK, N)
        t_row = text_embs[i:end] @ text_embs.T
        v_row = img_embs[i:end]  @ img_embs.T
        c_row = ALPHA * t_row + BETA * v_row

        tk = torch.topk(c_row, k=k_top, dim=1)
        bk = torch.topk(c_row, k=k_bot, dim=1, largest=False)

        topk_vals[i:end] = tk.values.to(SIM_DTYPE)
        topk_idx[i:end]  = tk.indices.to(torch.int32)
        botk_vals[i:end] = bk.values.to(SIM_DTYPE)
        botk_idx[i:end]  = bk.indices.to(torch.int32)

    torch.cuda.synchronize()
    return topk_vals, topk_idx, botk_vals, botk_idx


# DIAGNOSIS
def diagnose_similarities(frame_ids, actions, text_sim, image_sim,
                           sample_size=DIAG_SAMPLE):
    N       = len(frame_ids)
    indices = random.sample(range(N), min(sample_size, N))

    text_same, text_diff = [], []
    img_same,  img_diff  = [], []

    log.info(f"Diagnosing  |  anchors={len(indices)}  peers=200 each")

    for i in tqdm(indices, desc="  Diagnosis", position=1, leave=False):
        peers = random.sample(range(N), min(200, N))
        peers = [j for j in peers if j != i]
        t_row = text_sim[i, peers].float().cpu().tolist()
        v_row = image_sim[i, peers].float().cpu().tolist()
        for k_idx, j in enumerate(peers):
            if actions[i] == actions[j]:
                text_same.append(t_row[k_idx]); img_same.append(v_row[k_idx])
            else:
                text_diff.append(t_row[k_idx]); img_diff.append(v_row[k_idx])

    def stats(arr, label):
        arr = np.array(arr)
        log.info(
            f"  {label:<12s}  mean={arr.mean():.3f}  "
            f"p10={np.percentile(arr,10):.3f}  p25={np.percentile(arr,25):.3f}  "
            f"p50={np.percentile(arr,50):.3f}  p75={np.percentile(arr,75):.3f}  "
            f"p90={np.percentile(arr,90):.3f}  max={arr.max():.3f}"
        )

    log.info("── Same-action pairs ─────────────────────────────────────────")
    stats(text_same, "text_sim")
    stats(img_same,  "image_sim")
    log.info("── Diff-action pairs ─────────────────────────────────────────")
    stats(text_diff, "text_sim")
    stats(img_diff,  "image_sim")

    gap_t = np.mean(text_same) - np.mean(text_diff)
    gap_v = np.mean(img_same)  - np.mean(img_diff)
    log.info(f"Separation gap  |  text={gap_t:+.4f}  image={gap_v:+.4f}")
    if gap_t < 0.01:
        log.warning("Text separation gap < 0.01 — text embeddings may not discriminate actions")
    if gap_v < 0.01:
        log.warning("Image separation gap < 0.01 — image embeddings may not discriminate actions")


# ADAPTIVE THRESHOLDS
def compute_adaptive_thresholds(
    text_sim:  torch.Tensor,
    image_sim: torch.Tensor,
    actions:   list[str],
) -> dict:
    N   = len(actions)
    idx = random.sample(range(N), min(ADAPTIVE_SAMPLE, N))

    same_text, same_img = [], []
    diff_text, diff_img = [], []

    log.info(f"Adaptive thresholds  |  sample={min(ADAPTIVE_SAMPLE,N)}  peers=100 each")

    for i in tqdm(idx, desc="  Threshold sampling", position=1, leave=False):
        peers = random.sample(range(N), min(100, N))
        peers = [j for j in peers if j != i]
        t_row = text_sim[i, peers].float().cpu().tolist()
        v_row = image_sim[i, peers].float().cpu().tolist()
        for k_idx, j in enumerate(peers):
            t = t_row[k_idx]; v = v_row[k_idx]
            if actions[i] == actions[j]:
                same_text.append(t); same_img.append(v)
            else:
                diff_text.append(t); diff_img.append(v)

    thresholds = {
        "POS_TEXT":  float(np.percentile(same_text, POS_PERCENTILE)),
        "POS_IMG":   float(np.percentile(same_img,  POS_PERCENTILE)),
        "HARD_TEXT": float(np.percentile(diff_text, HARD_PERCENTILE)),
        "HARD_IMG":  float(np.percentile(diff_img,  HARD_PERCENTILE)),
        "NEG_TEXT":  float(np.percentile(diff_text, NEG_PERCENTILE)),
        "NEG_IMG":   float(np.percentile(diff_img,  NEG_PERCENTILE)),
    }

    log.info(
        f"Thresholds  |  "
        f"POS text≥{thresholds['POS_TEXT']:.4f} img≥{thresholds['POS_IMG']:.4f}  |  "
        f"HARD text≥{thresholds['HARD_TEXT']:.4f} img≥{thresholds['HARD_IMG']:.4f}  |  "
        f"NEG text≤{thresholds['NEG_TEXT']:.4f} img≤{thresholds['NEG_IMG']:.4f}"
    )

    pos_ok  = sum(1 for t, v in zip(same_text, same_img)
                  if t >= thresholds["POS_TEXT"] and v >= thresholds["POS_IMG"])
    hard_ok = sum(1 for t, v in zip(diff_text, diff_img)
                  if t >= thresholds["HARD_TEXT"] and v >= thresholds["HARD_IMG"])
    easy_ok = sum(1 for t, v in zip(diff_text, diff_img)
                  if t <= thresholds["NEG_TEXT"] and v <= thresholds["NEG_IMG"])

    log.info(
        f"Pass rates  |  "
        f"pos={pos_ok}/{len(same_text)} ({pos_ok/len(same_text)*100:.1f}%)  "
        f"hard={hard_ok}/{len(diff_text)} ({hard_ok/len(diff_text)*100:.1f}%)  "
        f"easy={easy_ok}/{len(diff_text)} ({easy_ok/len(diff_text)*100:.1f}%)"
    )

    if pos_ok / len(same_text) < 0.1:
        log.warning("Positive pass rate < 10% — consider raising POS_PERCENTILE")
    if hard_ok / len(diff_text) < 0.1:
        log.warning("Hard-neg pass rate < 10% — consider raising HARD_PERCENTILE or TOP_K_POOL")
    if easy_ok / len(diff_text) < 0.1:
        log.warning("Easy-neg pass rate < 10% — consider lowering NEG_PERCENTILE or raising BOTTOM_K_POOL")

    return thresholds


# VECTORIZED TRIPLET MINING  (chunked)
def mine_triplets_vectorized(
    frame_ids:      list[str],
    actions:        list[str],
    vehicle_states: list[dict],
    image_paths:    list[str],
    text_sim:       torch.Tensor,
    image_sim:      torch.Tensor,
    combined_sim:   torch.Tensor,
    thresholds:     dict,
) -> list[dict]:
    N = len(frame_ids)

    POS_TEXT  = thresholds["POS_TEXT"];  POS_IMG  = thresholds["POS_IMG"]
    HARD_TEXT = thresholds["HARD_TEXT"]; HARD_IMG = thresholds["HARD_IMG"]
    NEG_TEXT  = thresholds["NEG_TEXT"];  NEG_IMG  = thresholds["NEG_IMG"]

    # Encode action labels once
    unique_actions = sorted(set(actions))
    action_to_id   = {a: idx for idx, a in enumerate(unique_actions)}
    action_ids_gpu = torch.tensor(
        [action_to_id[a] for a in actions], dtype=torch.int16, device=DEVICE,
    )

    # Optional: route IDs for same-route filtering
    routes_list = [parse_route(fid) for fid in frame_ids]
    frame_indices = [parse_frame_index(fid) for fid in frame_ids]

    if EXCLUDE_SAME_ROUTE_POSITIVES or MIN_TEMPORAL_GAP > 0:
        unique_routes = sorted(set(routes_list))
        route_to_id   = {r: idx for idx, r in enumerate(unique_routes)}
        route_ids_gpu = torch.tensor(
            [route_to_id[r] for r in routes_list], dtype=torch.int32, device=DEVICE,
        )
        frame_idx_gpu = torch.tensor(
            [fi if fi is not None else -1 for fi in frame_indices],
            dtype=torch.int32, device=DEVICE,
        )
        log.info(f"Route/temporal filtering enabled  |  "
                 f"EXCLUDE_SAME_ROUTE={EXCLUDE_SAME_ROUTE_POSITIVES}  "
                 f"MIN_TEMPORAL_GAP={MIN_TEMPORAL_GAP}")
    else:
        route_ids_gpu = None
        frame_idx_gpu = None

    n_chunks = (N + MINING_CHUNK - 1) // MINING_CHUNK
    log.info(
        f"Mining  |  N={N}  MINING_CHUNK={MINING_CHUNK}  n_chunks={n_chunks}  "
        f"|  VRAM={_vram_str()}"
    )
    log.info(
        f"  pos  : same-action  text≥{POS_TEXT:.4f}  img≥{POS_IMG:.4f}  top-{TOP_K_POOL} pool\n"
        f"  hard : diff-action  "
        + (f"img≥{POS_IMG:.4f}  text<{POS_TEXT:.4f}  (visual-only mode)"
           if USE_VISUAL_HARD_NEGATIVES else
           f"text≥{HARD_TEXT:.4f}  img≥{HARD_IMG:.4f}")
        + f"  top-{TOP_K_POOL} pool\n"
        f"  easy : diff-action  text≤{NEG_TEXT:.4f}  img≤{NEG_IMG:.4f}  bot-{BOTTOM_K_POOL} pool"
    )

    triplets        = []
    skipped_no_pos  = 0
    skipped_no_hard = 0
    skipped_no_easy = 0
    extra_skipped   = 0

    for chunk_start in tqdm(range(0, N, MINING_CHUNK),
                            desc="  Triplet mining", total=n_chunks,
                            position=1, leave=False):
        chunk_end = min(chunk_start + MINING_CHUNK, N)
        C = chunk_end - chunk_start

        # Slice the three sim matrices for this chunk of anchors
        c_chunk = combined_sim[chunk_start:chunk_end].float()
        t_chunk = text_sim[chunk_start:chunk_end].float()
        v_chunk = image_sim[chunk_start:chunk_end].float()

        # Action-equality mask for this chunk
        anchor_ids = action_ids_gpu[chunk_start:chunk_end].unsqueeze(1)
        same_chunk = (anchor_ids == action_ids_gpu.unsqueeze(0))
        diff_chunk = ~same_chunk

        # Self-exclusion mask
        eye_chunk = torch.ones(C, N, dtype=torch.bool, device=DEVICE)
        self_cols = torch.arange(chunk_start, chunk_end, device=DEVICE)
        eye_chunk[torch.arange(C, device=DEVICE), self_cols] = False

        # top-K / bottom-K via torch.topk
        # Mask self with -inf so a frame never enters its own top-K / bottom-K
        c_chunk_masked = c_chunk.clone()
        c_chunk_masked[torch.arange(C, device=DEVICE), self_cols] = float('-inf')

        topk_idx = torch.topk(c_chunk_masked, k=min(TOP_K_POOL,    N - 1), dim=1).indices
        botk_idx = torch.topk(c_chunk_masked, k=min(BOTTOM_K_POOL, N - 1), dim=1,
                               largest=False).indices

        top_mask = torch.zeros(C, N, dtype=torch.bool, device=DEVICE)
        bot_mask = torch.zeros(C, N, dtype=torch.bool, device=DEVICE)
        top_mask.scatter_(1, topk_idx, True)
        bot_mask.scatter_(1, botk_idx, True)

        top_mask &= eye_chunk
        bot_mask &= eye_chunk

        # Optional: route/temporal filtering for positives
        pos_filter = same_chunk  # base: same action
        if EXCLUDE_SAME_ROUTE_POSITIVES and route_ids_gpu is not None:
            anchor_routes = route_ids_gpu[chunk_start:chunk_end].unsqueeze(1)
            diff_route    = (anchor_routes != route_ids_gpu.unsqueeze(0))
            pos_filter    = pos_filter & diff_route
        elif MIN_TEMPORAL_GAP > 0 and route_ids_gpu is not None and frame_idx_gpu is not None:
            anchor_routes = route_ids_gpu[chunk_start:chunk_end].unsqueeze(1)
            same_route    = (anchor_routes == route_ids_gpu.unsqueeze(0))
            anchor_fidx   = frame_idx_gpu[chunk_start:chunk_end].unsqueeze(1)
            temporal_gap  = (anchor_fidx - frame_idx_gpu.unsqueeze(0)).abs()
            too_close     = same_route & (temporal_gap < MIN_TEMPORAL_GAP)
            pos_filter    = pos_filter & ~too_close

        # Threshold + pool masks
        pos_mask  = top_mask & pos_filter & (t_chunk >= POS_TEXT) & (v_chunk >= POS_IMG)

        if USE_VISUAL_HARD_NEGATIVES:
            # Visual-only hard negatives: visually similar but textually different
            hard_mask = top_mask & diff_chunk & (v_chunk >= POS_IMG) & (t_chunk < POS_TEXT)
        else:
            # Default: dual-threshold hard negatives
            hard_mask = top_mask & diff_chunk & (t_chunk >= HARD_TEXT) & (v_chunk >= HARD_IMG)

        easy_mask = bot_mask & diff_chunk & (t_chunk <= NEG_TEXT) & (v_chunk <= NEG_IMG)

        log.debug(
            f"chunk [{chunk_start}:{chunk_end}]  "
            f"pos_nnz={pos_mask.sum().item()}  "
            f"hard_nnz={hard_mask.sum().item()}  "
            f"easy_nnz={easy_mask.sum().item()}  "
            f"VRAM={_vram_str()}"
        )

        # Assemble triplets for each anchor in this chunk
        for local_i in range(C):
            global_i = chunk_start + local_i

            pos_idx_t  = pos_mask[local_i].nonzero(as_tuple=True)[0]
            hard_idx_t = hard_mask[local_i].nonzero(as_tuple=True)[0]
            easy_idx_t = easy_mask[local_i].nonzero(as_tuple=True)[0]

            if len(pos_idx_t) == 0:
                skipped_no_pos += 1;  continue
            if len(hard_idx_t) == 0:
                skipped_no_hard += 1; continue
            if len(easy_idx_t) == 0:
                skipped_no_easy += 1; continue

            pos_scores     = c_chunk[local_i][pos_idx_t]
            pos_idx_sorted = pos_idx_t[
                torch.argsort(pos_scores, descending=True)[:NUM_POS_PER_ANCHOR]
            ].tolist()

            hard_scores     = c_chunk[local_i][hard_idx_t]
            hard_idx_sorted = hard_idx_t[
                torch.argsort(hard_scores, descending=True)[:NUM_HARD_NEG_PER_ANCHOR]
            ].tolist()

            hard_fids      = {frame_ids[j] for j in hard_idx_sorted}
            easy_list      = [j for j in easy_idx_t.tolist() if frame_ids[j] not in hard_fids]
            random.shuffle(easy_list)
            easy_idx_final = easy_list[:NUM_NEG_PER_ANCHOR]

            if not easy_idx_final:
                extra_skipped += 1
                log.debug(f"Anchor {global_i}: all easy-neg candidates were hard-neg frames")
                continue

            def pack(j: int, _li: int = local_i) -> dict:
                return {
                    "frame_id":     frame_ids[j],
                    "frame_path":   image_paths[j],
                    "state":        vehicle_states[j],
                    "text_sim":     round(t_chunk[_li, j].item(), 4),
                    "image_sim":    round(v_chunk[_li, j].item(), 4),
                    "combined_sim": round(c_chunk[_li, j].item(), 4),
                }

            triplets.append({
                "anchor": {
                    "frame_id":   frame_ids[global_i],
                    "frame_path": image_paths[global_i],
                    "state":      vehicle_states[global_i],
                    "action":     actions[global_i],
                },
                "positives":      [pack(j) for j in pos_idx_sorted],
                "hard_negatives": [
                    {**pack(j), "neg_type": "hard_negative", "neg_action": actions[j]}
                    for j in hard_idx_sorted
                ],
                "negatives":      [
                    {**pack(j), "neg_type": "negative", "neg_action": actions[j]}
                    for j in easy_idx_final
                ],
            })

    log.info(
        f"Mining done  |  triplets={len(triplets)}  "
        f"skipped: no_pos={skipped_no_pos}  no_hard={skipped_no_hard}  "
        f"no_easy={skipped_no_easy}  post_exclusion={extra_skipped}"
    )
    if extra_skipped:
        log.warning(f"{extra_skipped} anchors dropped: no easy-neg candidates "
                    "remained after hard-neg frame exclusion")
    return triplets


# MAIN
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="RAG embedding + triplet mining pipeline")
    p.add_argument("--dataset-json",  type=str, default=None,
                   help="JSON file or directory of JSON files (default: DATASET_JSON global)")
    p.add_argument("--emb-dir",       type=str, default=None,
                   help="Output dir for embeddings (default: EMB_DIR global)")
    p.add_argument("--triplets-dir",  type=str, default=None,
                   help="Output dir for triplets (default: TRIPLETS_DIR global)")
    p.add_argument("--data-root",     type=str, default=None,
                   help="Root dir for resolving image paths (default: DATA_ROOT global)")
    p.add_argument("--strip-prefix",  type=str, default=None,
                   help="Prefix to strip from frame_path before joining with data-root")
    # Mining knobs. Defaults are the module constants above (the values used
    # for the fast/slow profiles); the `default` profile was mined with
    # --top-k-pool 200 --bottom-k-pool 200 --hard-percentile 50.
    p.add_argument("--top-k-pool",      type=int, default=None, help=f"default {TOP_K_POOL}")
    p.add_argument("--bottom-k-pool",   type=int, default=None, help=f"default {BOTTOM_K_POOL}")
    p.add_argument("--pos-percentile",  type=int, default=None, help=f"default {POS_PERCENTILE}")
    p.add_argument("--hard-percentile", type=int, default=None, help=f"default {HARD_PERCENTILE}")
    p.add_argument("--neg-percentile",  type=int, default=None, help=f"default {NEG_PERCENTILE}")
    return p.parse_args()


def main():
    global SIM_CHUNK, MINING_CHUNK, DATASET_JSON, EMB_DIR, TRIPLETS_DIR, DATA_ROOT, STRIP_PREFIX
    global TOP_K_POOL, BOTTOM_K_POOL, POS_PERCENTILE, HARD_PERCENTILE, NEG_PERCENTILE
    args = _parse_args()
    if args.top_k_pool      is not None: TOP_K_POOL      = args.top_k_pool
    if args.bottom_k_pool   is not None: BOTTOM_K_POOL   = args.bottom_k_pool
    if args.pos_percentile  is not None: POS_PERCENTILE  = args.pos_percentile
    if args.hard_percentile is not None: HARD_PERCENTILE = args.hard_percentile
    if args.neg_percentile  is not None: NEG_PERCENTILE  = args.neg_percentile
    if args.dataset_json  is not None: DATASET_JSON  = Path(args.dataset_json)
    if args.emb_dir       is not None: EMB_DIR       = Path(args.emb_dir)
    if args.triplets_dir  is not None: TRIPLETS_DIR  = Path(args.triplets_dir)
    if args.data_root     is not None: DATA_ROOT     = Path(args.data_root)
    if args.strip_prefix  is not None: STRIP_PREFIX  = args.strip_prefix

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    TRIPLETS_DIR.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()

    log.info("=" * 70)
    log.info("RAG Contrastive Data Pipeline  —  OPTIMIZED")
    log.info(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Device    : {DEVICE}"
             + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    if DEVICE == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_gb  = (torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()) / 1e9
        vram_info = f"{free_gb:.1f}GB free / {total_gb:.1f}GB total"
        tuned = "manual" if (SIM_CHUNK_OVERRIDE or MINING_CHUNK_OVERRIDE) else "auto-tuned"
    else:
        vram_info = "N/A (CPU)"
        tuned = "auto-tuned"
    log.info(f"  VRAM      : {vram_info}")
    log.info(f"  SIM_CHUNK : {SIM_CHUNK}  ({tuned})")
    log.info(f"  MINING_CHUNK: {MINING_CHUNK}  ({tuned})")
    log.info(f"  SIM_DTYPE : {SIM_DTYPE}")
    log.info(f"  Features  : VISUAL_HARD_NEG={USE_VISUAL_HARD_NEGATIVES}  "
             f"EXCLUDE_SAME_ROUTE={EXCLUDE_SAME_ROUTE_POSITIVES}  "
             f"MIN_TEMPORAL_GAP={MIN_TEMPORAL_GAP}")
    log.info(f"  Log file  : {LOG_FILE.resolve()}")
    log.info("=" * 70)

    pbar = make_pipeline_bar()

    # 1. Load
    #
    # Two modes:
    #   A) Pre-merged embeddings exist (from SLURM pipeline):
    #      -> Load metadata from EMB_DIR/frame_ids.json to stay in sync
    #        with the embedding row order.  Do NOT re-parse the JSON dir
    #        (which would shuffle into a different order).
    #   B) Fresh run (no cached embeddings):
    #      -> Parse JSON dir, write frame_ids.json, generate embeddings.
    #
    index_path   = EMB_DIR / "frame_ids.json"
    text_emb_path_check = EMB_DIR / "text_embeddings.pt"
    img_emb_path_check  = EMB_DIR / "image_embeddings.pt"
    use_premerged = (
        index_path.exists()
        and text_emb_path_check.exists()
        and img_emb_path_check.exists()
    )

    with stage("Load dataset", pbar):
        if use_premerged:
            log.info(f"Pre-merged embeddings detected — loading metadata from {index_path}")
            log.info("  (skipping JSON re-parse to preserve embedding ↔ metadata alignment)")
            with open(index_path) as f:
                meta = json.load(f)
            frame_ids      = meta["frame_ids"]
            image_paths    = meta["image_paths"]
            actions        = meta["actions"]
            vehicle_states = meta["vehicle_states"]
            # sentences not needed when embeddings are cached
            sentences      = [""] * len(frame_ids)
            log.info(f"Loaded {len(frame_ids)} entries from pre-merged index")

            action_counts = Counter(actions)
            log.info(f"  unique actions: {len(action_counts)}")
            for action, count in action_counts.most_common():
                log.info(f"    {action:<30s}  {count:>6d}  ({count/len(frame_ids)*100:.1f}%)")
        else:
            log.info("No pre-merged embeddings — parsing JSON directory")
            frame_ids, image_paths, sentences, actions, vehicle_states = \
                load_dataset(DATASET_JSON)
            with open(index_path, "w") as f:
                json.dump({
                    "frame_ids":      frame_ids,
                    "image_paths":    image_paths,
                    "actions":        actions,
                    "vehicle_states": vehicle_states,
                }, f, indent=2)
            log.info(f"Frame index  →  {index_path}")

    # 2. Text embeddings
    text_emb_path = EMB_DIR / "text_embeddings.pt"
    with stage("Text embeddings (BGE-M3)", pbar):
        if text_emb_path.exists():
            log.info(f"Cache found  →  {text_emb_path}")
            text_embs = torch.load(text_emb_path, map_location=DEVICE)
            # The cache may have been saved from another device
            text_embs = text_embs.to(DEVICE)
            if text_embs.shape[0] != len(sentences):
                log.warning(f"Size mismatch ({text_embs.shape[0]} vs {len(sentences)}) — regenerating")
                text_embs = extract_text_embeddings(sentences, text_emb_path)
            else:
                log.info(f"Cache hit  |  shape={tuple(text_embs.shape)}")
        else:
            text_embs = extract_text_embeddings(sentences, text_emb_path)

    # 3. Image embeddings
    img_emb_path = EMB_DIR / "image_embeddings.pt"
    with stage("Image embeddings (SigLIP)", pbar):
        if img_emb_path.exists():
            log.info(f"Cache found  →  {img_emb_path}")
            img_embs = torch.load(img_emb_path, map_location=DEVICE)
            # The cache may have been saved from another device
            img_embs = img_embs.to(DEVICE)
            if img_embs.shape[0] != len(image_paths):
                log.warning(f"Size mismatch ({img_embs.shape[0]} vs {len(image_paths)}) — regenerating")
                img_embs = extract_image_embeddings(image_paths, img_emb_path)
            else:
                log.info(f"Cache hit  |  shape={tuple(img_embs.shape)}")
        else:
            img_embs = extract_image_embeddings(image_paths, img_emb_path)

    # 4. Similarity matrices
    SIM_CHUNK, MINING_CHUNK = _resolve_chunks()
    log.info(f"Chunk sizes re-resolved after embedding  |  "
             f"SIM_CHUNK={SIM_CHUNK}  MINING_CHUNK={MINING_CHUNK}")

    with stage("Similarity matrices (GPU)", pbar):
        combined, text_sim, image_sim = compute_similarity_matrices_gpu(
            text_embs, img_embs
        )

    # 5. Diagnose
    with stage("Diagnose similarities", pbar):
        diagnose_similarities(frame_ids, actions, text_sim, image_sim)

    # 6. Adaptive thresholds
    with stage("Adaptive thresholds", pbar):
        thresholds = compute_adaptive_thresholds(text_sim, image_sim, actions)

    # 7. Mine triplets
    with stage("Triplet mining", pbar):
        triplets = mine_triplets_vectorized(
            frame_ids, actions, vehicle_states, image_paths,
            text_sim, image_sim, combined, thresholds,
        )

    # 8. Save
    with stage("Save outputs", pbar):
        out_path = TRIPLETS_DIR / "triplets.jsonl"
        with open(out_path, "w") as f:
            for t in triplets:
                f.write(json.dumps(t) + "\n")
        log.info(f"Triplets  →  {out_path}  ({len(triplets)} lines)")

        thresh_path = EMB_DIR / "thresholds.json"
        with open(thresh_path, "w") as f:
            json.dump({
                "thresholds": thresholds,
                "percentiles": {
                    "POS_PERCENTILE":  POS_PERCENTILE,
                    "HARD_PERCENTILE": HARD_PERCENTILE,
                    "NEG_PERCENTILE":  NEG_PERCENTILE,
                },
                "config": {
                    "TOP_K_POOL":              TOP_K_POOL,
                    "BOTTOM_K_POOL":           BOTTOM_K_POOL,
                    "NUM_POS_PER_ANCHOR":      NUM_POS_PER_ANCHOR,
                    "NUM_NEG_PER_ANCHOR":      NUM_NEG_PER_ANCHOR,
                    "NUM_HARD_NEG_PER_ANCHOR": NUM_HARD_NEG_PER_ANCHOR,
                    "ALPHA": ALPHA, "BETA": BETA,
                    "SIM_CHUNK": SIM_CHUNK,
                    "MINING_CHUNK": MINING_CHUNK,
                    "SIM_DTYPE": str(SIM_DTYPE),
                    "SIM_CHUNK_OVERRIDE": SIM_CHUNK_OVERRIDE,
                    "MINING_CHUNK_OVERRIDE": MINING_CHUNK_OVERRIDE,
                    "USE_VISUAL_HARD_NEGATIVES": USE_VISUAL_HARD_NEGATIVES,
                    "EXCLUDE_SAME_ROUTE_POSITIVES": EXCLUDE_SAME_ROUTE_POSITIVES,
                    "MIN_TEMPORAL_GAP": MIN_TEMPORAL_GAP,
                },
            }, f, indent=2)
        log.info(f"Thresholds  →  {thresh_path}")

    pbar.close()

    # Final summary
    wall_elapsed = time.perf_counter() - wall_start
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Wall time          : {wall_elapsed/60:.1f} min  ({wall_elapsed:.0f}s)")
    log.info(f"  Total frames       : {len(frame_ids)}")
    log.info(f"  Triplets generated : {len(triplets)}")
    log.info(f"  Coverage           : {len(triplets)/len(frame_ids)*100:.1f}%")

    if triplets:
        avg_pos  = np.mean([len(t["positives"])      for t in triplets])
        avg_neg  = np.mean([len(t["negatives"])      for t in triplets])
        avg_hard = np.mean([len(t["hard_negatives"]) for t in triplets])
        hard_sat = sum(1 for t in triplets if len(t["hard_negatives"]) == NUM_HARD_NEG_PER_ANCHOR)
        easy_sat = sum(1 for t in triplets if len(t["negatives"])      == NUM_NEG_PER_ANCHOR)

        log.info(f"  Avg pos/anchor     : {avg_pos:.2f}")
        log.info(f"  Avg neg/anchor     : {avg_neg:.2f}  "
                 f"(saturation {easy_sat/len(triplets)*100:.1f}%)")
        log.info(f"  Avg hard/anchor    : {avg_hard:.2f}  "
                 f"(saturation {hard_sat/len(triplets)*100:.1f}%)")
        log.info("  Triplets per action:")
        for action, count in Counter(t["anchor"]["action"] for t in triplets).most_common():
            log.info(f"    {action:<30s}  {count:>6d}  ({count/len(triplets)*100:.1f}%)")

        if hard_sat / len(triplets) < 0.5:
            log.warning(
                f"Hard-neg saturation {hard_sat/len(triplets)*100:.1f}% < 50%  —  "
                f"consider lowering HARD_PERCENTILE (now {HARD_PERCENTILE}) "
                f"or raising TOP_K_POOL (now {TOP_K_POOL})"
            )

    log.info(f"  Log file           : {LOG_FILE.resolve()}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()