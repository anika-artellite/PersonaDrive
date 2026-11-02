#!/usr/bin/env python3
"""
Precompute in-context learning (ICL) examples for every frame in the database.

For each route, starting at frame SKIP_FRAMES+1:
  - Encode the frame (reusing pre-computed SigLIP embeddings when available)
  - Batch-query FAISS for the whole route in one call
  - Exclude same-route frames, keep top-2
  - Save per-route icl_examples.json.gz

Accelerations vs naive approach:
  1. Pre-computed embeddings   - reuse image_embeddings.pt, skip SigLIP (~50x)
  2. Batch FAISS query         - one search() per route instead of per-frame
  3. Batch control encoding    - run MLP on all route frames at once on GPU

Usage:
  python scripts/training/generate_icl_examples.py \\
    --checkpoint  /path/to/best_model.pt \\
    --faiss-index /path/to/fast/driving_index.faiss \\
    --faiss-meta  /path/to/fast/driving_index.meta.pkl \\
    --data-root   /path/to/database/simlingo/data/fast_dataset \\
    --emb-pt      /path/to/embeddings/fast/image_embeddings.pt \\
    --emb-index   /path/to/embeddings/fast/frame_ids.json \\
    --device cuda
"""

import os, gzip, json, pickle, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm


# Path resolution
# Frame paths in FAISS metadata are absolute and were captured under whatever
# root the data lived under at index-build time. At runtime we remap them
# onto the persistent on-share database root.

_PERSISTENT_DATA_ROOT = (
    f"{os.environ.get('PERSONADRIVE_ROOT', '/path/to/PersonaDrive')}/database"
)
_PATH_REWRITES = (
    ("/fast_dataset/data/simlingo/",       "/data/simlingo/fast/"),
    ("/slow_dataset/data/simlingo/",       "/data/simlingo/slow/"),
    ("/fast_dataset/drivelm/simlingo/",    "/drivelm/simlingo/fast/"),
    ("/slow_dataset/drivelm/simlingo/",    "/drivelm/simlingo/slow/"),
    ("/fast_dataset/commentary/simlingo/", "/commentary/simlingo/fast/"),
    ("/slow_dataset/commentary/simlingo/", "/commentary/simlingo/slow/"),
)

_DATA_ROOTS_CACHE = None
_PATH_CACHE: dict = {}


def _discover_data_roots():
    # PERSONADRIVE_DATA_ROOT (colon-separated) wins; otherwise use the persistent root.
    roots = []
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


def _data_roots():
    global _DATA_ROOTS_CACHE
    if _DATA_ROOTS_CACHE is None:
        _DATA_ROOTS_CACHE = _discover_data_roots()
    return _DATA_ROOTS_CACHE


def _normalize_path(path: str) -> str:
    """Map a (possibly-stale) absolute path to a real path on this node."""
    if not path:
        return path
    cached = _PATH_CACHE.get(path)
    if cached is not None:
        return cached
    if os.path.exists(path):
        _PATH_CACHE[path] = path
        return path
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


# Constants
COMMANDS     = ["Follow Lane","Change Lane Left","Change Lane Right",
                "Go Left","Go Right","Go Straight"]
COMMAND2IDX  = {c: i for i, c in enumerate(COMMANDS)}
NUM_COMMANDS = len(COMMANDS)

# CARLA RoadOption int -> command string. Source of truth:
# persona_drive_rag/rag_data_processor.py
ROADOPTION_TO_STR = {
    -1: "Follow Lane",   # VOID - fall back to lane follow
     1: "Go Left",
     2: "Go Right",
     3: "Go Straight",
     4: "Follow Lane",
     5: "Change Lane Left",
     6: "Change Lane Right",
}

def _command_to_str(cmd):
    """Normalize a command field to the canonical string form."""
    if cmd is None:
        return "Follow Lane"
    if isinstance(cmd, str):
        return cmd
    try:
        return ROADOPTION_TO_STR.get(int(cmd), "Follow Lane")
    except (TypeError, ValueError):
        return "Follow Lane"
CONTROL_DIM  = 128
VISION_DIM   = 768
PROJ_IN_DIM  = VISION_DIM + CONTROL_DIM
PROJ_OUT_DIM = 1024
SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_CACHE = os.environ.get("PERSONADRIVE_HF_CACHE") or None   # None: Hugging Face's default cache
SKIP_FRAMES  = 10
TOP_K_QUERY  = 10
NUM_ICL      = 2
# Same-route candidates are allowed only when their frame index is at least
# this far away from the query frame. The triplet/FAISS index was built with
# frame_step=5, so neighbors at ±5, ±10 are essentially the same scene; ±40
# frames ~ 10 s at 4 Hz is enough scene drift to be a useful exemplar while
# staying inside the route. Set to 0 to fully exclude same-route hits (legacy
# behavior); set to a very large number to require strictly cross-route hits.
MIN_SAME_ROUTE_FRAME_DIST = 40


# Model
class ControlEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_norm       = nn.LayerNorm(4)
        self.command_embedding = nn.Embedding(NUM_COMMANDS, 16)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(48, CONTROL_DIM), nn.GELU(), nn.LayerNorm(CONTROL_DIM))

    def forward(self, speed, throttle, steer, brake, command_idx):
        s = torch.stack([speed, throttle, steer, brake.float()], dim=1)
        return self.fusion(torch.cat([self.scalar_mlp(self.scalar_norm(s)),
                                      self.command_embedding(command_idx)], dim=1))


class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PROJ_IN_DIM, PROJ_OUT_DIM), nn.GELU(),
            nn.LayerNorm(PROJ_OUT_DIM),
            nn.Linear(PROJ_OUT_DIM, PROJ_OUT_DIM), nn.LayerNorm(PROJ_OUT_DIM))

    def forward(self, x): return self.net(x)


class DrivingEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.control_encoder = ControlEncoder()
        self.projection      = ProjectionMLP()

    def forward(self, image_emb, speed, throttle, steer, brake, command_idx):
        c = self.control_encoder(speed, throttle, steer, brake, command_idx)
        return F.normalize(self.projection(torch.cat([image_emb, c], dim=1)), p=2, dim=-1)


# Pre-computed embedding cache
class EmbeddingCache:
    """
    Lookup: image_path -> pre-computed 768-d SigLIP embedding.
    Falls back to None if path not found (caller must re-encode with SigLIP).
    """
    # Suffix marker used to normalize paths across different absolute prefixes.
    # The cache mixes paths from at least three locations:
    #   /path/to/scratch/.../database/simlingo/data/simlingo/...
    #   /path/to/PersonaDrive/database/simlingo/data/simlingo/...
    #   /path/to/other-mount/simlingo/data/simlingo/...
    # All share "/data/simlingo/<profile>/.../rgb/<frame>.jpg" - that tail
    # uniquely identifies the file regardless of how the data was mounted.
    _SUFFIX_MARKER = "/data/simlingo/"

    # Layout-rewrite pairs (cached path -> runtime path). Mirrors the structural
    # shift between the Apr-28 fast/slow rebuild and the new on-node layout.
    _LAYOUT_REWRITES = (
        ("/fast_dataset/data/simlingo/", "/data/simlingo/fast/"),
        ("/slow_dataset/data/simlingo/", "/data/simlingo/slow/"),
    )

    @classmethod
    def _suffix_key(cls, path: str) -> str:
        i = path.rfind(cls._SUFFIX_MARKER)
        return path[i:] if i >= 0 else path

    @classmethod
    def _layout_variants(cls, path: str):
        """Yield path plus each layout-rewritten variant (forward + reverse)."""
        yield path
        for old, new in cls._LAYOUT_REWRITES:
            if old in path:
                yield path.replace(old, new)
            if new in path:
                yield path.replace(new, old)

    def __init__(self, emb_pt: str, frame_ids_json: str):
        print("Loading pre-computed embeddings ...")
        embs = torch.load(emb_pt, weights_only=True)   # (N, 768) float32
        with open(frame_ids_json) as f:
            meta = json.load(f)
        image_paths = meta.get("image_paths", [])
        self.embs = embs
        self.path_to_idx = {p: i for i, p in enumerate(image_paths)}
        # Suffix-keyed fallback for paths that come in via a different absolute
        # prefix (different cwd / symlink chain) but resolve to the same file.
        self._suffix_to_idx: dict[str, int] = {}
        for p, i in self.path_to_idx.items():
            self._suffix_to_idx.setdefault(self._suffix_key(p), i)
        print(f"  {len(self.path_to_idx)} pre-computed embeddings loaded "
              f"({len(self._suffix_to_idx)} unique tmp_link suffixes)")

    def get(self, image_path: str):
        idx = self.path_to_idx.get(image_path)
        if idx is None:
            # Try suffix match across all layout variants of the path
            for variant in self._layout_variants(image_path):
                idx = self._suffix_to_idx.get(self._suffix_key(variant))
                if idx is not None:
                    break
            if idx is None:
                return None
            # Memoize so subsequent lookups for the same caller-path are O(1)
            self.path_to_idx[image_path] = idx
        return self.embs[idx]   # (768,) tensor


# SigLIP fallback encoder
class SigLIPEncoder:
    """Encodes images with SigLIP when pre-computed embedding is unavailable."""
    def __init__(self, device):
        self.device = device
        self.processor    = None
        self.vision_model = None   # lazy-load

    def _ensure_loaded(self):
        if self.vision_model is not None:
            return
        print("  Loading SigLIP (fallback for missing pre-computed embeddings) ...")
        self.processor    = AutoProcessor.from_pretrained(
            SIGLIP_MODEL, cache_dir=SIGLIP_CACHE)
        self.vision_model = AutoModel.from_pretrained(
            SIGLIP_MODEL, cache_dir=SIGLIP_CACHE).to(self.device).eval()
        for p in self.vision_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_batch(self, rgb_paths):
        self._ensure_loaded()
        imgs = [Image.open(p).convert("RGB") for p in rgb_paths]
        pv   = self.processor(images=imgs, return_tensors="pt",
                               padding=True)["pixel_values"].to(self.device)
        out  = self.vision_model.get_image_features(pixel_values=pv)
        feat = (out.pooler_output if hasattr(out, "pooler_output") and
                out.pooler_output is not None else out).float()
        return feat.cpu()   # (B, 768)


# Helpers
def _load_gz(path):
    if not os.path.exists(path): return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f: return json.load(f)
    except Exception: return None


def _route_id(frame_path):
    p = Path(frame_path)
    try:
        parts  = p.parts
        rgb_i  = next(i for i, x in enumerate(parts) if x == "rgb")
        return "/".join(parts[rgb_i - 3:rgb_i])
    except StopIteration:
        return str(p.parent.parent)


def _frame_idx_from_path(frame_path):
    """Extract the integer frame index from a /.../rgb/NNNN.jpg path.

    Returns None when the path doesn't match the expected layout - caller
    treats that as "unknown distance" and falls back to the strict same-route
    rule for safety.
    """
    try:
        return int(Path(frame_path).stem)
    except (ValueError, TypeError):
        return None


def _build_icl_example(meta):
    state = meta.get("state", {}) or {}
    s1    = meta.get("state_1_back", {}) or {}
    s2    = meta.get("state_2_back", {}) or {}
    def spd(d): return float(d.get("speed",    state.get("speed",    0.0)))
    def thr(d): return float(d.get("throttle", state.get("throttle", 0.0)))
    cmd = _command_to_str(state.get("command"))
    wps = meta.get("future_waypoints", [])
    wps = [[round(float(w[0]),4), round(float(w[1]),4)] for w in wps] if wps else []

    def _round_pt(p):
        if not p: return None
        try:
            return [round(float(p[0]), 4), round(float(p[1]), 4)]
        except (TypeError, ValueError, IndexError):
            return None

    example = {
        "rgb_path":          _normalize_path(meta.get("frame_path", "")),
        "speed_history":     [spd(s2), spd(s1), spd(state)],
        "throttle_history":  [thr(s2), thr(s1), thr(state)],
        "command":           cmd or "Follow Lane",
        "waypoints":         wps,
    }
    # Optional enrichment - only included when present in the metadata so old
    # indices without enrichment don't get empty fields.
    commentary = meta.get("commentary")
    if commentary:
        example["commentary"] = commentary
    scenario_name = meta.get("scenario_name")
    if scenario_name:
        example["scenario_name"] = scenario_name
    tp      = _round_pt(meta.get("target_point"))
    tp_next = _round_pt(meta.get("target_point_next"))
    if tp is not None:
        example["target_point"] = tp
    if tp_next is not None:
        example["target_point_next"] = tp_next
    return example


def find_routes(data_root, exclude_substrings=()):
    """Walk data_root and return route dirs (those containing measurements).

    Any route whose path contains one of `exclude_substrings` is skipped.
    Excluded substrings also prune walking (so we don't descend into them).
    """
    routes = []
    excludes = tuple(s for s in exclude_substrings if s)
    for dirpath, dirnames, filenames in os.walk(data_root):
        # Prune excluded subtrees in-place so we don't even descend into them
        if excludes:
            dirnames[:] = [d for d in dirnames
                           if not any(e in os.path.join(dirpath, d) for e in excludes)]
        if "measurements" in dirnames or "measurements.json" in filenames:
            if not (excludes and any(e in dirpath for e in excludes)):
                routes.append(dirpath)
    return sorted(routes)


# Per-route batch processing
@torch.no_grad()
def process_route_batch(route_dir, model, emb_cache, siglip,
                        index, metadata, device,
                        skip_frames=SKIP_FRAMES, top_k=TOP_K_QUERY, num_icl=NUM_ICL,
                        overwrite=False):
    """
    Process one route:
      1. Collect all frames >= skip_frames+1
      2. Get SigLIP embeddings (pre-computed or live)
      3. Batch control encoding
      4. Single FAISS search for all frames
      5. Save icl_examples.json.gz
    """
    out_path = os.path.join(route_dir, "icl_examples.json.gz")
    if os.path.exists(out_path) and not overwrite:
        return 0

    meas_dir = os.path.join(route_dir, "measurements")
    rgb_dir  = os.path.join(route_dir, "rgb")
    has_meas = os.path.isdir(meas_dir) or os.path.exists(os.path.join(route_dir, "measurements.json"))
    if not has_meas:
        print(f"[SKIP] missing measurements: {route_dir}")
        return 0
    if not os.path.isdir(rgb_dir):
        print(f"[SKIP] missing rgb/: {route_dir}")
        return 0

    current_route = _route_id(os.path.join(rgb_dir, "0000.jpg"))

    # Load measurements: aggregated file (1 read) or per-frame fallback
    agg_meas_path = os.path.join(route_dir, "measurements.json")
    agg_meas_gz   = agg_meas_path + ".gz"
    if os.path.exists(agg_meas_path):
        with open(agg_meas_path) as f:
            all_measurements = json.load(f)  # list indexed by frame number
    elif os.path.exists(agg_meas_gz):
        all_measurements = _load_gz(agg_meas_gz)
    else:
        all_measurements = None   # fall back to per-frame files

    # Collect frames
    frame_nums, rgb_paths, states = [], [], []
    def _frame_num(fname):
        s = Path(fname).stem          # '0080.json' for .gz, '0080' for .json
        return int(s.replace('.json', ''))

    # Frame list: from measurements/ subdir or from length of aggregated measurements.json
    if os.path.isdir(meas_dir):
        available_frames = sorted(_frame_num(f) for f in os.listdir(meas_dir)
                                  if f.endswith(".json.gz") or f.endswith(".json"))
    elif all_measurements is not None:
        available_frames = list(range(len(all_measurements)))
    else:
        return 0

    for fn in available_frames:
        if fn <= skip_frames: continue
        rp = os.path.join(rgb_dir, f"{fn:04d}.jpg")
        if not os.path.exists(rp): continue

        # Get measurement: aggregated list (fast) or per-frame file (fallback)
        if all_measurements is not None and fn < len(all_measurements):
            m = all_measurements[fn]
        else:
            mp = os.path.join(meas_dir, f"{fn:04d}.json.gz")
            if not os.path.exists(mp): mp = mp[:-3]
            if not os.path.exists(mp): continue
            m = _load_gz(mp) if mp.endswith(".gz") else json.load(open(mp))
        if m is None: continue

        cmd = _command_to_str(m.get("command", 4))
        states.append({
            "speed":    float(m.get("speed",    0.0)),
            "throttle": float(m.get("throttle", 0.0)),
            "steer":    float(m.get("steer",    0.0)),
            "brake":    float(bool(m.get("brake", False))),
            "command":  cmd,
        })
        frame_nums.append(fn)
        rgb_paths.append(rp)

    if not frame_nums:
        return 0

    # Step 1: get image embeddings (pre-computed or SigLIP)
    img_embs_list = []
    fallback_idxs = []
    for i, rp in enumerate(rgb_paths):
        cached = emb_cache.get(rp) if emb_cache else None
        if cached is not None:
            img_embs_list.append(cached)
        else:
            img_embs_list.append(None)
            fallback_idxs.append(i)

    # Batch-encode missing frames with SigLIP
    if fallback_idxs:
        missing_paths = [rgb_paths[i] for i in fallback_idxs]
        batch_size    = 32
        encoded       = []
        for b in range(0, len(missing_paths), batch_size):
            encoded.append(siglip.encode_batch(missing_paths[b:b+batch_size]))
        encoded = torch.cat(encoded, dim=0)   # (M, 768)
        for k, orig_i in enumerate(fallback_idxs):
            img_embs_list[orig_i] = encoded[k]

    img_embs = torch.stack(img_embs_list, dim=0).to(device)  # (N, 768)

    # Step 2: batch control encoding
    B = len(states)
    speed_t    = torch.tensor([s["speed"]    for s in states], dtype=torch.float32, device=device)
    throttle_t = torch.tensor([s["throttle"] for s in states], dtype=torch.float32, device=device)
    steer_t    = torch.tensor([s["steer"]    for s in states], dtype=torch.float32, device=device)
    brake_t    = torch.tensor([s["brake"]    for s in states], dtype=torch.float32, device=device)
    cmd_t      = torch.tensor([COMMAND2IDX.get(s["command"], 0) for s in states],
                               dtype=torch.long, device=device)

    embeddings = model(img_embs, speed_t, throttle_t, steer_t, brake_t, cmd_t)
    embs_np    = embeddings.cpu().float().numpy()  # (N, 1024)

    # Step 3: single batch FAISS query
    scores_all, idxs_all = index.search(embs_np, top_k + 1)  # (N, top_k+1)

    # Step 4: build ICL examples
    # Same-route hits within MIN_SAME_ROUTE_FRAME_DIST of the query frame are
    # rejected (too visually similar). Hits in the same route but far enough
    # away in time ARE allowed - the FAISS index is sparse enough (built with
    # frame_step=5) that long routes like Town03_lr_9 can otherwise have ALL
    # 11 nearest neighbors come from inside the same route, which dropped
    # ~15.5k frames from the file.
    aggregated = {}
    for i, fn in enumerate(frame_nums):
        examples = []
        for score, ret_idx in zip(scores_all[i], idxs_all[i]):
            if ret_idx < 0 or ret_idx >= len(metadata): continue
            meta = metadata[ret_idx]
            raw_fp = meta.get("frame_path", "")
            if not raw_fp: continue
            if _route_id(raw_fp) == current_route:
                cand_idx = _frame_idx_from_path(raw_fp)
                # Reject when we can't parse the candidate's frame index OR
                # when it's too close in time to the query frame.
                if cand_idx is None or abs(cand_idx - fn) < MIN_SAME_ROUTE_FRAME_DIST:
                    continue
            resolved_fp = _normalize_path(raw_fp)
            if not os.path.exists(resolved_fp): continue
            examples.append(_build_icl_example(meta))
            if len(examples) >= num_icl: break
        if examples:
            aggregated[str(fn)] = examples

    if aggregated:
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(aggregated, f)

    return len(aggregated)


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--faiss-index", required=True)
    parser.add_argument("--faiss-meta",  required=True)
    parser.add_argument("--data-root",   required=True)
    parser.add_argument("--emb-pt",        default=None,
                        help="Pre-computed image_embeddings.pt (strongly recommended)")
    parser.add_argument("--emb-index",     default=None,
                        help="frame_ids.json matching emb-pt")
    parser.add_argument("--save-emb-dir",  default=None,
                        help="If set, save the expanded full embedding cache here for reuse")
    parser.add_argument("--skip-frames", type=int, default=SKIP_FRAMES)
    parser.add_argument("--num-icl",     type=int, default=NUM_ICL)
    parser.add_argument("--top-k",       type=int, default=TOP_K_QUERY)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Re-generate icl_examples.json.gz even if it exists.")
    parser.add_argument("--exclude",     default="",
                        help="Comma-separated path substrings to skip when "
                             "walking --data-root (e.g. '/fast/,/slow/' for "
                             "the normal profile).")
    args = parser.parse_args()
    exclude_subs = [s for s in (args.exclude or "").split(",") if s.strip()]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  ICL Precomputation  (device={device})")
    print(f"{'='*60}")

    # Load FAISS
    print("Loading FAISS ...")
    index = faiss.read_index(args.faiss_index)
    with open(args.faiss_meta, "rb") as f:
        metadata = pickle.load(f)
    print(f"  {index.ntotal} vectors | {len(metadata)} metadata entries")

    # Load model
    print("Loading DrivingEmbeddingModel ...")
    model = DrivingEmbeddingModel().to(device).eval()
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    # Pre-computed embeddings cache
    emb_cache = None
    if args.emb_pt and args.emb_index:
        emb_cache = EmbeddingCache(args.emb_pt, args.emb_index)
        # Check cache coverage and expand if needed
        routes_preview = find_routes(args.data_root, exclude_substrings=exclude_subs)
        all_rgb = []
        for rd in routes_preview:
            rgb_dir = os.path.join(rd, "rgb")
            if os.path.isdir(rgb_dir):
                for f in os.listdir(rgb_dir):
                    if f.endswith(".jpg"):
                        all_rgb.append(os.path.join(rgb_dir, f))
        total_frames = len(all_rgb)
        # Use .get() so the realpath fallback is consulted (and memoized).
        cached_frames = sum(1 for p in all_rgb if emb_cache.get(p) is not None)
        cache_hit_rate = cached_frames / total_frames if total_frames else 0
        print(f"  Cache hit rate: {cached_frames}/{total_frames} = {cache_hit_rate*100:.1f}%")

        if cache_hit_rate < 0.5:
            print(f"\n  Low cache hit rate — pre-computing embeddings for ALL frames ...")
            siglip_tmp = SigLIPEncoder(device)
            siglip_tmp._ensure_loaded()
            batch_size  = 128
            new_paths, new_embs = [], []
            with tqdm(range(0, len(all_rgb), batch_size),
                      desc="  Pre-computing SigLIP", unit="batch") as pb:
                for b in pb:
                    batch = [p for p in all_rgb[b:b+batch_size]
                             if emb_cache.get(p) is None]
                    if not batch: continue
                    embs = siglip_tmp.encode_batch(batch)
                    new_paths.extend(batch)
                    new_embs.append(embs)
            if new_embs:
                new_tensor = torch.cat(new_embs, dim=0)
                # Merge into cache
                start = len(emb_cache.embs)
                emb_cache.embs = torch.cat([emb_cache.embs, new_tensor], dim=0)
                for i, p in enumerate(new_paths):
                    emb_cache.path_to_idx[p] = start + i
                print(f"  Cache expanded: {len(emb_cache.path_to_idx)} total entries")

                # Save expanded cache to disk for reuse
                if args.save_emb_dir:
                    os.makedirs(args.save_emb_dir, exist_ok=True)
                    pt_out   = os.path.join(args.save_emb_dir, "image_embeddings_full.pt")
                    idx_out  = os.path.join(args.save_emb_dir, "frame_ids_full.json")
                    torch.save(emb_cache.embs, pt_out)
                    idx2path = {v: k for k, v in emb_cache.path_to_idx.items()}
                    ordered_paths = [idx2path[i] for i in range(len(emb_cache.embs))]
                    with open(idx_out, "w") as f:
                        json.dump({"image_paths": ordered_paths,
                                   "frame_ids":   ordered_paths}, f)
                    print(f"  Saved → {pt_out}")
                    print(f"  Saved → {idx_out}")
                    print(f"  Next run: pass --emb-pt {pt_out} --emb-index {idx_out}")
    else:
        print("  WARNING: --emb-pt/--emb-index not provided — will use SigLIP for all frames (slow)")

    # SigLIP fallback (for any remaining cache misses)
    siglip = SigLIPEncoder(device)

    # Discover routes
    print(f"\nScanning: {args.data_root}")
    if exclude_subs:
        print(f"  Excluding paths containing: {exclude_subs}")
    routes = find_routes(args.data_root, exclude_substrings=exclude_subs)
    print(f"  {len(routes)} routes found")

    total, t0 = 0, time.perf_counter()
    with tqdm(routes, desc="Routes", unit="route") as pbar:
        for rd in pbar:
            n = process_route_batch(
                rd, model, emb_cache, siglip, index, metadata, device,
                skip_frames=args.skip_frames, top_k=args.top_k, num_icl=args.num_icl,
                overwrite=args.overwrite,
            )
            total += n
            pbar.set_postfix(frames=total)

    elapsed = time.perf_counter() - t0
    print(f"\nDone — {total} frames  |  {elapsed:.1f}s  |  {total/elapsed:.0f} frames/s")


if __name__ == "__main__":
    main()
