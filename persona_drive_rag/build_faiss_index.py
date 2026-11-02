"""
build_faiss_index.py - PersonaDrive Stage 2b: per-style FAISS index builder
===========================================================================
Generates embeddings for frames using the best trained checkpoint,
then builds a FAISS index for retrieval at inference time.

Architecture matches embed_training.py exactly:
  - Precomputed SigLIP embeddings  (768-d)
  - ControlEncoder                 (128-d)
  - Concat                         (896-d)
  - ProjectionMLP  896 -> 1024      (2 linear layers, no residual)
  - L2-normalised output           (1024-d)

Usage:
    # one or more profiles; JSONL=none indexes every frame in FRAME_IDS
    python build_faiss_index.py \
        --checkpoint    checkpoints/best_model.pt \
        --output_dir    faiss_index \
        --batch_size    256 \
        --device        cuda \
        --profile NAME JSONL|none EMB_PT FRAME_IDS [PROCESSED_JSON]

Outputs (in --output_dir/<NAME>/):
    driving_index.faiss       - FAISS flat-IP index (cosine on unit sphere)
    driving_index.meta.pkl    - metadata list (frame_id, frame_path, state)
    driving_index.info.json   - build summary
"""

import os
import gzip
import json
import pickle
import argparse
import time
import threading
from typing import List, Dict, Optional

# Path resolution
# Frame paths in FAISS metadata are absolute and were captured under whatever
# root the data lived under at index-build time. At runtime we remap them onto
# the persistent on-share database root. Set PERSONADRIVE_DATA_ROOT (colon-separated
# like PATH) to prepend explicit roots to the search order.

_PERSISTENT_DATA_ROOT = f"{os.environ.get('PERSONADRIVE_ROOT', '/path/to/PersonaDrive')}/database"

# Structural rewrites for paths captured before the dataset layout changed:
#   OLD: <root>/database/simlingo/<X>_dataset/<kind>/simlingo/lb1_split/...
#   NEW: <root>/database/simlingo/<kind>/simlingo/<X>/lb1_split/...
# Covers data (rgb), drivelm (vqa) and commentary trees.
_PATH_REWRITES = (
    ("/fast_dataset/data/simlingo/",       "/data/simlingo/fast/"),
    ("/slow_dataset/data/simlingo/",       "/data/simlingo/slow/"),
    ("/fast_dataset/drivelm/simlingo/",    "/drivelm/simlingo/fast/"),
    ("/slow_dataset/drivelm/simlingo/",    "/drivelm/simlingo/slow/"),
    ("/fast_dataset/commentary/simlingo/", "/commentary/simlingo/fast/"),
    ("/slow_dataset/commentary/simlingo/", "/commentary/simlingo/slow/"),
)

_DATA_ROOTS_CACHE = None
_PATH_CACHE: Dict[str, str] = {}


def _discover_data_roots() -> list:
    # PERSONADRIVE_DATA_ROOT (colon-separated) wins; otherwise use the persistent root.
    roots: list = []
    env = os.environ.get("PERSONADRIVE_DATA_ROOT", "")
    for r in env.split(":"):
        r = r.strip().rstrip("/")
        if r and os.path.isdir(r):
            roots.append(r)
    if os.path.isdir(_PERSISTENT_DATA_ROOT):
        roots.append(_PERSISTENT_DATA_ROOT)
    seen, out = set(), []
    for r in roots:
        rp = os.path.realpath(r)
        if rp not in seen:
            seen.add(rp); out.append(rp)
    return out


def _data_roots() -> list:
    global _DATA_ROOTS_CACHE
    if _DATA_ROOTS_CACHE is None:
        _DATA_ROOTS_CACHE = _discover_data_roots()
    return _DATA_ROOTS_CACHE


def _reset_data_roots() -> list:
    """Re-scan after PERSONADRIVE_DATA_ROOT changes (e.g. set by CLI flag)."""
    global _DATA_ROOTS_CACHE, _PATH_CACHE
    _DATA_ROOTS_CACHE = None
    _PATH_CACHE = {}
    return _data_roots()


def _normalize_path(path: str) -> str:
    """Map a (possibly-stale) absolute frame path to a real path on this node."""
    if not path:
        return path
    cached = _PATH_CACHE.get(path)
    if cached is not None:
        return cached
    if os.path.exists(path):
        _PATH_CACHE[path] = path
        return path

    # Build candidate path strings: original + each structural rewrite.
    candidates = [path]
    for old, new in _PATH_REWRITES:
        if old in path:
            candidates.append(path.replace(old, new))

    roots = _data_roots()
    for cp in candidates:
        parts = cp.lstrip("/").split("/")
        for root in roots:
            for n_strip in range(len(parts)):
                cand = os.path.join(root, *parts[n_strip:])
                if os.path.exists(cand):
                    _PATH_CACHE[path] = cand
                    return cand
    _PATH_CACHE[path] = path
    return path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

from torch.utils.data import Dataset, DataLoader, random_split


# 0. CONSTANTS  (must match embed_training.py exactly)

COMMANDS = [
    "Follow Lane",
    "Change Lane Left",
    "Change Lane Right",
    "Go Left",
    "Go Right",
    "Go Straight",
]
COMMAND2IDX  = {c: i for i, c in enumerate(COMMANDS)}
NUM_COMMANDS = len(COMMANDS)

# Mapping from CARLA RoadOption integers to command strings.
# This converts raw measurement file integers (1-6) to command names
# so they can be looked up in COMMAND2IDX.
CARLA_TO_COMMAND_STRING = {
    1: "Go Left",
    2: "Go Right",
    3: "Go Straight",
    4: "Follow Lane",
    5: "Change Lane Left",
    6: "Change Lane Right",
}

CONTROL_DIM  = 128
VISION_DIM   = 768
PROJ_IN_DIM  = VISION_DIM + CONTROL_DIM   # 896
PROJ_OUT_DIM = 1024


# 1. MODEL  (identical to embed_training.py)

class ControlEncoder(nn.Module):
    def __init__(
        self,
        num_commands:      int = NUM_COMMANDS,
        command_emb_dim:   int = 16,
        scalar_hidden_dim: int = 32,
        output_dim:        int = CONTROL_DIM,
    ):
        super().__init__()
        self.scalar_norm       = nn.LayerNorm(4)
        self.command_embedding = nn.Embedding(num_commands, command_emb_dim)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(4, scalar_hidden_dim), nn.GELU(),
            nn.Linear(scalar_hidden_dim, scalar_hidden_dim), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(scalar_hidden_dim + command_emb_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, speed, throttle, steer, brake, command_idx):
        scalars     = torch.stack([speed, throttle, steer, brake.float()], dim=1)
        scalars     = self.scalar_norm(scalars)
        scalar_feat = self.scalar_mlp(scalars)
        cmd_feat    = self.command_embedding(command_idx)
        return self.fusion(torch.cat([scalar_feat, cmd_feat], dim=1))


class ProjectionMLP(nn.Module):
    """2-layer MLP - matches embed_training.py exactly."""
    def __init__(self, in_dim: int = PROJ_IN_DIM, out_dim: int = PROJ_OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DrivingEmbeddingModel(nn.Module):
    def __init__(
        self,
        vision_dim:  int  = VISION_DIM,
        control_dim: int  = CONTROL_DIM,
        embed_dim:   int  = PROJ_OUT_DIM,
        normalize:   bool = True,
    ):
        super().__init__()
        self.control_encoder = ControlEncoder()
        self.normalize       = normalize
        self.projection      = ProjectionMLP(
            in_dim=vision_dim + control_dim, out_dim=embed_dim
        )

    def forward(self, image_emb: torch.Tensor, control: Dict) -> torch.Tensor:
        c_emb = self.control_encoder(
            speed=control["speed"], throttle=control["throttle"],
            steer=control["steer"], brake=control["brake"],
            command_idx=control["command_idx"],
        )
        fused = torch.cat([image_emb, c_emb], dim=1)
        proj  = self.projection(fused)
        if self.normalize:
            proj = F.normalize(proj, p=2, dim=-1)
        return proj


# 2. DATASET  (flat - one record per unique frame)

class FrameDataset(Dataset):
    """
    Flat dataset of unique frames.

    Two modes:
      - jsonl_path provided : load only anchor frames from triplet JSONL (original)
      - jsonl_path is None  : load ALL frames from frame_index_path (full coverage)
    """

    def __init__(self, image_emb_path: str, frame_index_path: str,
                 jsonl_path: str = None):
        self.image_embs = torch.load(image_emb_path, map_location='cpu', weights_only=False)
        with open(frame_index_path) as f:
            meta = json.load(f)
        self.id_to_index = {fid: i for i, fid in enumerate(meta["frame_ids"])}

        self.records = []

        if jsonl_path is not None:
            # Original mode: anchor frames from triplet JSONL only
            seen = set()
            with open(jsonl_path) as f:
                for line in f:
                    entry  = json.loads(line.strip())
                    anchor = entry["anchor"]
                    fid    = anchor["frame_id"]
                    if fid in seen or fid not in self.id_to_index:
                        continue
                    seen.add(fid)
                    self.records.append({
                        "frame_id":   fid,
                        "frame_path": anchor.get("frame_path", ""),
                        "state":      anchor["state"],
                    })
        else:
            # Full mode: ALL frames in the embedding index (O(n), uses positional arrays)
            frame_ids_list  = meta["frame_ids"]
            image_paths     = meta.get("image_paths",    [""] * len(frame_ids_list))
            vehicle_states  = meta.get("vehicle_states", [{}] * len(frame_ids_list))
            for i, fid in enumerate(frame_ids_list):
                self.records.append({
                    "frame_id":   fid,
                    "frame_path": image_paths[i],
                    "state":      vehicle_states[i] if isinstance(vehicle_states, list) else {},
                })

    def __len__(self):
        return len(self.records)

    def _state_to_control(self, state: dict) -> Dict:
        cmd_raw = state.get("command", "Follow Lane")

        # Convert CARLA RoadOption integer to command string if needed
        # (measurement files store integers; processed JSON may have strings)
        if isinstance(cmd_raw, int):
            cmd_str = CARLA_TO_COMMAND_STRING.get(cmd_raw, "Follow Lane")
        else:
            cmd_str = cmd_raw

        cmd_idx = COMMAND2IDX.get(cmd_str, 0)
        return {
            "speed":       torch.tensor(state["speed"],        dtype=torch.float32),
            "throttle":    torch.tensor(state["throttle"],     dtype=torch.float32),
            "steer":       torch.tensor(state["steer"],        dtype=torch.float32),
            "brake":       torch.tensor(float(state["brake"]), dtype=torch.float32),
            "command_idx": torch.tensor(cmd_idx,               dtype=torch.long),
        }

    def __getitem__(self, idx: int):
        r = self.records[idx]
        return {
            "frame_id":   r["frame_id"],
            "frame_path": r["frame_path"],
            "state":      r["state"],
            "image_emb":  self.image_embs[self.id_to_index[r["frame_id"]]],
            "control":    self._state_to_control(r["state"]),
        }


def collate_frames(batch):
    def stack_ctrl(key):
        return {k: torch.stack([b[key][k] for b in batch]) for k in batch[0][key]}
    return {
        "frame_id":   [b["frame_id"]   for b in batch],
        "frame_path": [b["frame_path"] for b in batch],
        "state":      [b["state"]      for b in batch],
        "image_emb":  torch.stack([b["image_emb"] for b in batch]),
        "control":    stack_ctrl("control"),
    }


# 3. EMBEDDING GENERATION

def to_device(ctrl: dict, device) -> dict:
    return {k: v.to(device) for k, v in ctrl.items()}


@torch.no_grad()
def generate_embeddings(model, loader, device):
    model.eval()
    all_embs    = []
    frame_ids   = []
    frame_paths = []
    states      = []

    total   = len(loader)
    t_start = time.perf_counter()

    for step, batch in enumerate(loader):
        img_emb = batch["image_emb"].to(device)
        ctrl    = to_device(batch["control"], device)
        emb     = model(img_emb, ctrl)
        all_embs.append(emb.cpu().float().numpy())
        frame_ids.extend(batch["frame_id"])
        frame_paths.extend(batch["frame_path"])
        states.extend(batch["state"])

        if (step + 1) % 20 == 0 or (step + 1) == total:
            elapsed = time.perf_counter() - t_start
            print(f"  [{step+1}/{total}] frames={len(frame_ids)} | {elapsed:.1f}s")

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)
    return embeddings, frame_ids, frame_paths, states


# 4. MAIN


# 3b. METADATA ENRICHMENT

def _derive_vqa_path(frame_path: str) -> str:
    return (frame_path
            .replace('/data/simlingo/', '/drivelm/simlingo/')
            .replace('/rgb/', '/vqa/')
            .replace('.jpg', '.json.gz'))


def _derive_commentary_path(frame_path: str) -> str:
    return (frame_path
            .replace('/data/simlingo/', '/commentary/simlingo/')
            .replace('/rgb/', '/commentary/')
            .replace('.jpg', '.json.gz'))


def _load_gz(path: str):
    """Load a .json.gz file, return None if missing. Resolves stale paths."""
    if not path:
        return None
    resolved = _normalize_path(path)
    if not os.path.exists(resolved):
        return None
    try:
        with gzip.open(resolved, 'rt', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _extract_qa(vqa_data: dict) -> dict:
    """Extract Q&A pairs from VQA file into a clean dict."""
    if not vqa_data:
        return {}
    qa = {}
    for section in ('perception', 'prediction', 'planning', 'behavior'):
        items = vqa_data.get('QA', {}).get(section, [])
        qa[section] = [{'Q': i.get('Q', ''), 'A': i.get('A', '')} for i in items]
    return qa


def _parse_sentence(sentence: str) -> dict:
    """
    Parse the structured sentence field into vqa and commentary dicts.
    Handles sections: PERCEPTION, PREDICTION, PLANNING, COMMENTARY.
    """
    if not sentence:
        return {}

    section_map = {
        '--- PERCEPTION ---':  'perception',
        '--- PREDICTION ---':  'prediction',
        '--- PLANNING ---':    'planning',
        '--- COMMENTARY ---':  'commentary',
    }

    result = {'vqa': {}, 'commentary': ''}
    current_section = None
    current_q = None
    qa_pairs = []

    for line in sentence.splitlines():
        line = line.strip()

        if line in section_map:
            # Save previous section's Q&A
            if current_section and current_section != 'commentary':
                result['vqa'][current_section] = qa_pairs
            qa_pairs = []
            current_section = section_map[line]
            current_q = None
            continue

        # Generic section-header sentinel: any "--- X ---" line ends whatever
        # section we were in. Without this, unknown headers (e.g.
        # "--- FUTURE WAYPOINTS ---") leak into the commentary because they
        # aren't in section_map and the commentary branch happily appends them.
        if line.startswith('--- ') and line.endswith(' ---'):
            if current_section and current_section != 'commentary':
                result['vqa'][current_section] = qa_pairs
            qa_pairs = []
            current_section = None
            current_q = None
            continue

        if current_section == 'commentary':
            if line:
                result['commentary'] += line + ' '
        elif current_section:
            if line.startswith('Q:'):
                current_q = line[2:].strip()
            elif line.startswith('A:') and current_q:
                qa_pairs.append({'Q': current_q, 'A': line[2:].strip()})
                current_q = None

    # Save last section
    if current_section and current_section != 'commentary' and qa_pairs:
        result['vqa'][current_section] = qa_pairs

    result['commentary'] = result['commentary'].strip()
    return result


def _build_processed_lookup(processed_json_path: str) -> dict:
    """Build frame_path -> record lookup from processed JSON."""
    if not processed_json_path or not os.path.exists(processed_json_path):
        return {}
    print(f"  Loading processed JSON: {processed_json_path} ...")
    with open(processed_json_path) as f:
        records = json.load(f)
    lookup = {}
    for rec in records:
        fp = rec.get('frame_path', '')
        if fp:
            lookup[fp] = rec
    print(f"  Processed lookup: {len(lookup)} entries")
    return lookup


def _load_measurement_state(frame_path: str, frame_offset: int) -> dict:
    """
    Load state from a measurement file at frame_number + frame_offset.
    frame_offset=-1 -> one frame back, -2 -> two frames back.
    Falls back to aggregated measurements.json if per-frame .json.gz is missing.
    Returns {} (empty, no warning) if target frame is before route start.
    """
    from pathlib import Path
    p = Path(_normalize_path(frame_path))
    frame_num = int(p.stem)
    target_num = frame_num + frame_offset
    if target_num < 0:
        # Early frames don't have historical data - return empty without warning
        return {}

    route_dir = p.parent.parent

    # Try per-frame gzip file first
    data = _load_gz(str(route_dir / 'measurements' / f'{target_num:04d}.json.gz'))

    # Fall back to aggregated measurements.json (cached per route)
    if not data:
        agg_path = route_dir / 'measurements.json'
        if agg_path.exists():
            all_meas = _load_agg_measurements(str(agg_path))
            if all_meas and target_num < len(all_meas):
                data = all_meas[target_num]

    # Return empty dict (not error dict) for missing historical states
    # This is expected behavior for early frames or end-of-route frames
    if not data:
        return {}

    return {
        'speed':    data.get('speed'),
        'throttle': data.get('throttle'),
        'steer':    data.get('steer'),
        'brake':    data.get('brake'),
        'command':  data.get('command'),
    }


# Measurement file cache - avoids redundant gzip reads for overlapping frames
# Dict reads/writes are GIL-protected in CPython, no explicit lock needed
_meas_cache: Dict[str, object] = {}
_agg_meas_cache: Dict[str, list] = {}  # cache for aggregated measurements.json per route

def _load_measurement_cached(path: str):
    if path in _meas_cache:
        return _meas_cache[path]
    data = _load_gz(path)
    _meas_cache[path] = data
    return data

def _load_agg_measurements(agg_path: str):
    if agg_path in _agg_meas_cache:
        return _agg_meas_cache[agg_path]
    try:
        with open(agg_path) as f:
            data = json.load(f)
        _agg_meas_cache[agg_path] = data if isinstance(data, list) else None
    except Exception:
        _agg_meas_cache[agg_path] = None
    return _agg_meas_cache[agg_path]


def _compute_future_waypoints(frame_path: str, n_future: int = 10) -> list:
    """
    Compute future waypoints in ego-local coordinates using the same method
    as rag_data_processor.get_waypoints_from_measurements().

    Reads ego_matrix from the current frame and n_future subsequent frames,
    transforms each future ego position into the current frame's local coords:
        local_xy = R.T @ (future_translation - current_translation)
    where R and t come from ego_matrix[:3, :3] and ego_matrix[:3, 3].

    Returns a list of [x, y] pairs (float32, rounded to 4 dp).
    """
    from pathlib import Path
    p = Path(_normalize_path(frame_path))
    frame_num = int(p.stem)
    meas_dir = p.parent.parent / 'measurements'

    curr = _load_measurement_cached(str(meas_dir / f'{frame_num:04d}.json.gz'))
    if not curr or 'ego_matrix' not in curr:
        return []

    origin = np.array(curr['ego_matrix'], dtype=np.float64)
    origin_R = origin[:3, :3]    # (3, 3) rotation
    origin_t = origin[:3, 3:4]   # (3, 1) translation

    waypoints = []
    for i in range(1, n_future + 1):
        fut = _load_measurement_cached(str(meas_dir / f'{frame_num + i:04d}.json.gz'))
        if not fut or 'ego_matrix' not in fut:
            break
        wp = np.array(fut['ego_matrix'], dtype=np.float64)[:3, 3:4]  # (3, 1)
        ego_local = origin_R.T @ (wp - origin_t)                      # (3, 1)
        waypoints.append([round(float(ego_local[0, 0]), 4),
                          round(float(ego_local[1, 0]), 4)])

    return waypoints


def _enrich_record(frame_path: str, processed_lookup: dict) -> dict:
    """Load VQA, commentary, processed JSON fields and historical states."""
    enriched = {}

    # VQA - try file first, fall back to parsing sentence from processed JSON
    vqa_data = _load_gz(_derive_vqa_path(frame_path))
    enriched['vqa'] = _extract_qa(vqa_data)

    # Commentary - try file first
    com_data = _load_gz(_derive_commentary_path(frame_path))
    if com_data:
        enriched['commentary']    = com_data.get('commentary', '')
        enriched['scenario_name'] = com_data.get('scenario_name', '')
        enriched['cause_object']  = com_data.get('cause_object_string', '')

    # Processed JSON fields (optional enrichment)
    rec = processed_lookup.get(frame_path, {})
    if rec:
        enriched['sentence']          = rec.get('sentence', '')
        enriched['scenario']          = rec.get('scenario', '')

        # Fall back to parsing sentence when VQA/commentary files don't exist
        if not enriched.get('vqa') or not enriched.get('commentary'):
            parsed = _parse_sentence(rec.get('sentence', ''))
            if not enriched.get('vqa'):
                enriched['vqa'] = parsed.get('vqa', {})
            if not enriched.get('commentary'):
                enriched['commentary'] = parsed.get('commentary', '')
        enriched['target_point']      = rec.get('target_point')
        enriched['target_point_next'] = rec.get('target_point_next')
        enriched['command']           = rec.get('command')
        enriched['next_command']      = rec.get('next_command')

    # future_waypoints: use processed JSON if available, otherwise compute from measurements
    enriched['future_waypoints'] = (
        rec.get('future_waypoints') if rec
        else None
    ) or _compute_future_waypoints(frame_path)

    # Historical states: 1 and 2 frames back
    enriched['state_1_back'] = _load_measurement_state(frame_path, -1)
    enriched['state_2_back'] = _load_measurement_state(frame_path, -2)

    return enriched

def build_index_for_profile(name, jsonl_path, image_emb, frame_ids,
                             model, device, output_dir, args,
                             processed_json=None):
    """Build and save a FAISS index for a single profile."""
    print(f"\n{'='*60}")
    print(f"  Profile: {name}")
    print(f"{'='*60}")

    profile_out = os.path.join(output_dir, name)
    os.makedirs(profile_out, exist_ok=True)

    full_ds = FrameDataset(
        image_emb_path   = image_emb,
        frame_index_path = frame_ids,
        jsonl_path       = jsonl_path if jsonl_path and jsonl_path.lower() != "none" else None,
    )
    print(f"  Unique frames: {len(full_ds)}")

    loader = DataLoader(
        full_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_frames, num_workers=0, pin_memory=True,
    )

    print(f"  Generating embeddings ...")
    t0 = time.perf_counter()
    embeddings, frame_ids_list, frame_paths, states = generate_embeddings(model, loader, device)
    elapsed = time.perf_counter() - t0
    print(f"  {len(frame_ids_list)} embeddings in {elapsed:.1f}s  shape={embeddings.shape}")

    # Build FAISS index
    index = faiss.IndexFlatIP(PROJ_OUT_DIM)
    index.add(embeddings)

    base = os.path.join(profile_out, "driving_index")
    faiss.write_index(index, base + ".faiss")

    # Enrich metadata with VQA, commentary and processed JSON fields
    processed_lookup = _build_processed_lookup(processed_json)
    n_frames = len(frame_ids_list)
    print(f"  Enriching metadata for {n_frames} frames (parallel I/O) ...")
    t_enrich = time.perf_counter()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    num_workers = min(16, os.cpu_count() or 4)

    def _enrich_one(i):
        fp = frame_paths[i]
        record = {"frame_id": frame_ids_list[i], "frame_path": fp, "state": states[i]}
        record.update(_enrich_record(fp, processed_lookup))
        return i, record

    metadata = [None] * n_frames
    vqa_loaded = com_loaded = proc_loaded = 0

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_enrich_one, i): i for i in range(n_frames)}
        from tqdm import tqdm
        with tqdm(total=n_frames, desc=f"  Enriching ({num_workers} threads)",
                  unit="frame", ncols=90) as pbar:
            for fut in as_completed(futures):
                i, record = fut.result()
                metadata[i] = record
                if record.get('vqa'):        vqa_loaded  += 1
                if record.get('commentary'): com_loaded  += 1
                if record.get('sentence'):   proc_loaded += 1
                pbar.update(1)

    print(f"  Enriched in {time.perf_counter()-t_enrich:.1f}s  "
          f"vqa={vqa_loaded} | commentary={com_loaded} | processed={proc_loaded}")
    with open(base + ".meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    info = {
        "profile":          name,
        "num_frames":       len(frame_ids_list),
        "embedding_dim":    PROJ_OUT_DIM,
        "index_type":       "IndexFlatIP",
        "checkpoint":       args.checkpoint,
        "built_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(base + ".info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Sanity check
    D, I = index.search(embeddings[:1], k=3)
    print(f"  Sanity — top-3 for first frame:")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        st = states[idx]
        marker = " <- self" if idx == 0 else ""
        print(f"    [{rank+1}] score={dist:.4f}  {frame_ids_list[idx]}"
              f"  cmd={st.get('command','?')}{marker}")

    print(f"  Saved → {profile_out}/")
    return len(frame_ids_list)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from trained model")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output_dir",  type=str, default="faiss_index")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--device",      type=str, default="cuda")
    # Multi-profile: --profile name jsonl emb frame_ids (repeatable)
    parser.add_argument("--profile", action="append", nargs="+",
                        metavar="ARG",
                        help="Add a profile: --profile NAME JSONL EMB FRAME_IDS [PROCESSED_JSON]\n"
                             "  Pass 'none' for JSONL to index all frames.\n"
                             "  PROCESSED_JSON is optional — enables full metadata enrichment.")
    # Legacy single-profile args (backward compat)
    parser.add_argument("--jsonl_path",  type=str, default=None)
    parser.add_argument("--image_emb",   type=str, default="embeddings/image_embeddings.pt")
    parser.add_argument("--frame_ids",   type=str, default="embeddings/frame_ids.json")
    parser.add_argument("--data-root", dest="data_root", type=str, default=None,
                        help="Colon-separated list of database roots to prepend "
                             "to auto-discovered search order.")
    args = parser.parse_args()

    if args.data_root:
        existing = os.environ.get("PERSONADRIVE_DATA_ROOT", "")
        os.environ["PERSONADRIVE_DATA_ROOT"] = (
            args.data_root if not existing else f"{args.data_root}:{existing}"
        )
        _reset_data_roots()
    print(f"  data roots   : {_data_roots() or '(none discovered)'}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model (shared across all profiles)
    print(f"\nLoading model from {args.checkpoint} ...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = DrivingEmbeddingModel().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}  |  device={device}")

    # Determine profiles
    if args.profile:
        profiles = []
        for p in args.profile:
            if len(p) < 4:
                raise ValueError(f"--profile requires at least 4 args: NAME JSONL EMB FRAME_IDS, got {p}")
            profiles.append({
                "name":           p[0],
                "jsonl":          p[1],
                "emb":            p[2],
                "frame_ids":      p[3],
                "processed_json": p[4] if len(p) > 4 else None,
            })
    else:
        # Legacy single-profile
        if args.jsonl_path is None:
            raise ValueError("Provide --profile or --jsonl_path")
        profiles = [{"name": "default",
                     "jsonl":     args.jsonl_path,
                     "emb":       args.image_emb,
                     "frame_ids": args.frame_ids}]

    print(f"\nBuilding {len(profiles)} profile index(es) → {args.output_dir}/")

    total_frames = 0
    for cfg in profiles:
        n = build_index_for_profile(
            name           = cfg["name"],
            jsonl_path     = cfg["jsonl"],
            image_emb      = cfg["emb"],
            frame_ids      = cfg["frame_ids"],
            model          = model,
            device         = device,
            output_dir     = args.output_dir,
            args           = args,
            processed_json = cfg.get("processed_json"),
        )
        total_frames += n

    print(f"\n{'='*60}")
    print(f"  ALL DONE — {total_frames} frames indexed across {len(profiles)} profiles")
    print(f"  Output structure:")
    for cfg in profiles:
        print(f"    {args.output_dir}/{cfg['name']}/driving_index.{{faiss,meta.pkl,info.json}}")
    print(f"  At retrieval: load the index for your chosen profile")
    print(f"    index = faiss.read_index('{args.output_dir}/<profile>/driving_index.faiss')")
    print(f"    meta  = pickle.load(open('{args.output_dir}/<profile>/driving_index.meta.pkl','rb'))")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()