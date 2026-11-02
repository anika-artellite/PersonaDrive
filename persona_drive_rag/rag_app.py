"""
rag_app.py - PersonaDrive retrieval inspector (local dev tool)
=============================================================
Flask backend: upload image + vehicle state -> top-5 retrieved frames

Architecture matches embed_training.py:
  - SigLIP base (768-d) for query-time image encoding
  - ControlEncoder (128-d)
  - ProjectionMLP 896 -> 1024 (2-layer, no residual)
  - L2-normalised output

Usage:
    # Auto-discovery (recommended): load every subprofile under faiss_index/
    python rag_app.py
    # Or point at a different base dir:
    python rag_app.py --faiss-index-dir faiss_index --port 7861
    # Or pin exact profiles manually:
    python rag_app.py \
        --profile default faiss_index/default/driving_index.faiss faiss_index/default/driving_index.meta.pkl \
        --profile fast    faiss_index/fast/driving_index.faiss    faiss_index/fast/driving_index.meta.pkl
    # Legacy single-profile path also still works:
    python rag_app.py \
        --faiss_index faiss_index/driving_index.faiss \
        --metadata    faiss_index/driving_index.meta.pkl
"""

import os
import io
import gzip
import glob
import json
import pickle
import base64
import argparse
import traceback
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoProcessor, AutoModel


# 0. CONSTANTS  (must match embed_training.py)

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
CONTROL_DIM  = 128
VISION_DIM   = 768
PROJ_IN_DIM  = VISION_DIM + CONTROL_DIM   # 896
PROJ_OUT_DIM = 1024

SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_CACHE = os.environ.get("PERSONADRIVE_HF_CACHE") or None   # None: Hugging Face's default cache


# Path resolution  (must match build_faiss_index.py)
# Frame paths in FAISS metadata are absolute and were captured under whatever
# root the data lived under at index-build time. At runtime we remap them onto
# the persistent on-share database root. Set PERSONADRIVE_DATA_ROOT (colon-separated
# like PATH) or pass --data-root to prepend explicit roots to the search order.

_PERSISTENT_DATA_ROOT = f"{os.environ.get('PERSONADRIVE_ROOT', '/path/to/PersonaDrive')}/database"

# Structural rewrites for paths captured before the dataset layout changed:
#   OLD: <root>/database/simlingo/<X>_dataset/data/simlingo/lb1_split/...
#   NEW: <root>/database/simlingo/data/simlingo/<X>/lb1_split/...
_PATH_REWRITES = (
    ("/fast_dataset/data/simlingo/", "/data/simlingo/fast/"),
    ("/slow_dataset/data/simlingo/", "/data/simlingo/slow/"),
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
    """Re-scan after PERSONADRIVE_DATA_ROOT changes (e.g. set by --data-root)."""
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
    def __init__(self):
        super().__init__()
        self.control_encoder = ControlEncoder()
        self.projection      = ProjectionMLP()

    def forward(self, image_emb, control):
        c = self.control_encoder(
            speed=control["speed"], throttle=control["throttle"],
            steer=control["steer"], brake=control["brake"],
            command_idx=control["command_idx"],
        )
        return F.normalize(self.projection(torch.cat([image_emb, c], dim=1)), p=2, dim=-1)


# 2. INFERENCE ENGINE

class RAGEngine:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RAGEngine] device={self.device}")

        # SigLIP for encoding query images at runtime
        print("[RAGEngine] Loading SigLIP ...")
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL, cache_dir=SIGLIP_CACHE)
        self.vision_encoder = AutoModel.from_pretrained(SIGLIP_MODEL, cache_dir=SIGLIP_CACHE)
        self.vision_encoder.eval().to(self.device)
        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)

        # Driving embedding model (shared across all profiles)
        print("[RAGEngine] Loading DrivingEmbeddingModel ...")
        self.model = DrivingEmbeddingModel().to(self.device)
        ckpt = torch.load(args.checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[RAGEngine] Checkpoint epoch={ckpt.get('epoch', '?')}")

        # Load per-profile FAISS indices + metadata
        self.profiles = {}   # {name: {"index": faiss_index, "metadata": list}}
        self.default_profile = None

        if hasattr(args, "profile") and args.profile:
            for name, faiss_path, meta_path in args.profile:
                print(f"[RAGEngine] Loading profile '{name}' ...")
                index = faiss.read_index(faiss_path)
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)
                self.profiles[name] = {"index": index, "metadata": metadata}
                print(f"  {name}: {index.ntotal} vectors")
                if self.default_profile is None:
                    self.default_profile = name
        else:
            # Legacy single-profile fallback
            if not args.faiss_index or not args.metadata:
                raise ValueError(
                    "No FAISS index specified. Use --faiss_index <path> --metadata <path>, "
                    "or use --profile NAME FAISS_PATH META_PATH (repeatable for multiple profiles)."
                )
            print("[RAGEngine] Loading FAISS index (single profile) ...")
            index = faiss.read_index(args.faiss_index)
            with open(args.metadata, "rb") as f:
                metadata = pickle.load(f)
            self.profiles["default"] = {"index": index, "metadata": metadata}
            self.default_profile = "default"
            print(f"  default: {index.ntotal} vectors")

        print(f"[RAGEngine] Profiles loaded: {list(self.profiles.keys())}")

    @torch.no_grad()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL image -> SigLIP embedding (1, 768)"""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.vision_encoder.get_image_features(pixel_values=pixel_values)

        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state.mean(dim=1)
        else:
            raise TypeError(f"Unexpected SigLIP output type: {type(out)}")

        return F.normalize(feats.float(), p=2, dim=-1)

    @torch.no_grad()
    def query(self, pil_image: Image.Image, state: dict, top_k: int = 5,
              profile: str = None):
        profile = profile or self.default_profile
        if profile not in self.profiles:
            raise ValueError(f"Unknown profile '{profile}'. Available: {list(self.profiles.keys())}")

        index    = self.profiles[profile]["index"]
        metadata = self.profiles[profile]["metadata"]

        img_emb = self.encode_image(pil_image)   # (1, 768)

        cmd_idx = COMMAND2IDX.get(state.get("command", "Follow Lane"), 0)
        ctrl = {
            "speed":       torch.tensor([state["speed"]],        dtype=torch.float32, device=self.device),
            "throttle":    torch.tensor([state["throttle"]],     dtype=torch.float32, device=self.device),
            "steer":       torch.tensor([state["steer"]],        dtype=torch.float32, device=self.device),
            "brake":       torch.tensor([float(state["brake"])], dtype=torch.float32, device=self.device),
            "command_idx": torch.tensor([cmd_idx],               dtype=torch.long,    device=self.device),
        }

        emb = self.model(img_emb, ctrl)
        emb_np = emb.cpu().float().numpy()

        scores, indices = index.search(emb_np, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            meta = metadata[idx]
            img_b64 = None
            fp = meta.get("frame_path", "")
            resolved_fp = _normalize_path(fp) if fp else ""
            if resolved_fp and os.path.exists(resolved_fp):
                try:
                    with Image.open(resolved_fp) as im:
                        im_small = im.resize((320, 180))
                        buf = io.BytesIO()
                        im_small.save(buf, format="JPEG", quality=75)
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    pass

            results.append({
                "rank":            len(results) + 1,
                "score":           round(float(score), 4),
                "frame_id":        meta["frame_id"],
                "frame_path":      meta.get("frame_path", ""),
                "state":           meta.get("state", {}),
                "future_waypoints": meta.get("future_waypoints", []),
                "image_b64":       img_b64,
            })

        return results

    def get_stats(self):
        """Return per-profile statistics."""
        stats = {}
        for name, p in self.profiles.items():
            meta = p["metadata"]
            scenarios, commands = {}, {}
            for m in meta:
                s = m.get("scenario_name") or m.get("scenario") or "unknown"
                c = (m.get("state") or {}).get("command", "unknown")
                scenarios[s] = scenarios.get(s, 0) + 1
                commands[c]  = commands.get(c, 0) + 1
            stats[name] = {
                "total":     len(meta),
                "scenarios": dict(sorted(scenarios.items(), key=lambda x: -x[1])),
                "commands":  dict(sorted(commands.items(),  key=lambda x: -x[1])),
            }
        return stats

    def _resolve_frame_path(self, fp: str) -> str:
        """Resolve a (possibly-stale) absolute frame path against current data roots."""
        return _normalize_path(fp)

    def browse(self, profile: str, page: int = 0, page_size: int = 20,
               scenario_filter: str = None, command_filter: str = None):
        """Return a page of frames from the index with optional filters."""
        profile = profile or self.default_profile
        if profile not in self.profiles:
            raise ValueError(f"Unknown profile '{profile}'")
        meta = self.profiles[profile]["metadata"]

        filtered = []
        for m in meta:
            s = m.get("scenario_name") or m.get("scenario") or ""
            c = (m.get("state") or {}).get("command", "")
            if scenario_filter and scenario_filter.lower() not in s.lower():
                continue
            if command_filter and command_filter != c:
                continue
            filtered.append(m)

        total = len(filtered)
        page_items = filtered[page * page_size:(page + 1) * page_size]

        results = []
        for m in page_items:
            fp = m.get("frame_path", "")
            img_b64 = None
            if fp:
                resolved_fp = self._resolve_frame_path(fp)
                if os.path.exists(resolved_fp):
                    try:
                        with Image.open(resolved_fp) as im:
                            im_small = im.resize((320, 180))
                            buf = io.BytesIO()
                            im_small.save(buf, format="JPEG", quality=70)
                            img_b64 = base64.b64encode(buf.getvalue()).decode()
                    except Exception as e:
                        pass
            results.append({
                "frame_id":    m.get("frame_id", ""),
                "frame_path":  fp,
                "state":       m.get("state", {}),
                "scenario":    m.get("scenario_name") or m.get("scenario", ""),
                "commentary":  m.get("commentary", ""),
                "vqa_count":   sum(len(v) for v in (m.get("vqa") or {}).values()),
                "has_history": bool(m.get("state_1_back")),
                "image_b64":   img_b64,
            })
        return {"total": total, "page": page, "page_size": page_size,
                "pages": (total + page_size - 1) // page_size,
                "items": results}

    def get_frame_detail(self, profile: str, frame_id: str):
        """Return full metadata for a single frame."""
        profile = profile or self.default_profile
        meta = self.profiles[profile]["metadata"]
        for m in meta:
            if m.get("frame_id") == frame_id:
                result = dict(m)
                fp = m.get("frame_path", "")
                if fp:
                    resolved_fp = self._resolve_frame_path(fp)
                    if os.path.exists(resolved_fp):
                        try:
                            with Image.open(resolved_fp) as im:
                                buf = io.BytesIO()
                                im.save(buf, format="JPEG", quality=80)
                                result["image_b64"] = base64.b64encode(buf.getvalue()).decode()
                        except Exception:
                            pass
                return result
        return None


# 2b. ICL VIEWER ENGINE  (browses pre-generated icl_examples.json.gz)

# Subpath under each discovered data root to scan, per profile, plus path
# substrings to exclude (so 'normal' doesn't pick up fast/slow routes).
# `_data_roots()` returns paths ending in /database, so subpaths start with
# `simlingo/...` to land inside the simlingo subtree.
_ICL_PROFILE_SCAN = (
    ("normal", "simlingo/data/simlingo",       ("/data/simlingo/fast/", "/data/simlingo/slow/")),
    ("fast",   "simlingo/data/simlingo/fast",  ()),
    ("slow",   "simlingo/data/simlingo/slow",  ()),
)


def _encode_image_b64(path: str, size: Tuple[int, int] = (320, 180), quality: int = 75) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as im:
            im_small = im.resize(size)
            buf = io.BytesIO()
            im_small.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


class IclEngine:
    """Indexes icl_examples.json.gz files across profiles and serves them.

    For each profile, walks the discovered data roots' /data/simlingo/<profile>/
    subtree (excluding fast/slow for 'normal'), peeks each gz file to learn
    its frame count, and stores (route_dir, frame_nums) for fast lookup.
    """

    def __init__(self):
        # profile -> [(rel_id, route_dir, frame_nums)]  sorted by rel_id
        self.routes: Dict[str, List[Tuple[str, str, List[int]]]] = {}
        # profile -> {rel_id: (route_dir, frame_nums)}
        self.route_map: Dict[str, Dict[str, Tuple[str, List[int]]]] = {}
        self._scanned = False

    def scan(self, verbose: bool = True):
        """Walk the data roots once, populating self.routes / self.route_map."""
        data_roots = _data_roots()
        if verbose:
            print(f"[ICL] scanning {len(data_roots)} data roots ...")
        for profile, subpath, excludes in _ICL_PROFILE_SCAN:
            routes: List[Tuple[str, str, List[int]]] = []
            seen: set = set()
            for root in data_roots:
                base = os.path.join(root, subpath)
                if not os.path.isdir(base):
                    continue
                # glob is cheaper than os.walk and matches our flat naming
                pattern = os.path.join(base, "**", "icl_examples.json.gz")
                for icl_path in glob.iglob(pattern, recursive=True):
                    if any(e in icl_path for e in excludes):
                        continue
                    route_dir = os.path.dirname(icl_path)
                    rel = os.path.relpath(route_dir, root)
                    if rel in seen:
                        continue
                    seen.add(rel)
                    try:
                        with gzip.open(icl_path, "rt", encoding="utf-8") as f:
                            d = json.load(f)
                        frame_nums = sorted(int(k) for k in d.keys())
                    except Exception:
                        continue
                    if not frame_nums:
                        continue
                    routes.append((rel, route_dir, frame_nums))
            routes.sort(key=lambda x: x[0])
            self.routes[profile] = routes
            self.route_map[profile] = {r[0]: (r[1], r[2]) for r in routes}
            if verbose:
                print(f"[ICL]   {profile:>6s}: {len(routes)} routes with ICL data")
        self._scanned = True

    def list_profiles(self) -> Dict[str, int]:
        """Profiles that have at least one route with ICL data."""
        return {p: len(rs) for p, rs in self.routes.items() if rs}

    def list_routes(self, profile: str, page: int = 0, page_size: int = 200,
                    filter_str: str = "") -> dict:
        rs = self.routes.get(profile, [])
        if filter_str:
            f = filter_str.lower()
            rs = [r for r in rs if f in r[0].lower()]
        total = len(rs)
        start = page * page_size
        slice_ = rs[start:start + page_size]
        return {
            "profile":   profile,
            "total":     total,
            "page":      page,
            "page_size": page_size,
            "pages":     (total + page_size - 1) // page_size,
            "routes":    [{"id": r[0], "frames": len(r[2])} for r in slice_],
        }

    def list_frames(self, profile: str, route_id: str) -> Optional[dict]:
        entry = self.route_map.get(profile, {}).get(route_id)
        if entry is None:
            return None
        _, frame_nums = entry
        return {"profile": profile, "route_id": route_id, "frames": frame_nums}

    def get_frame_icl(self, profile: str, route_id: str, frame: int) -> Optional[dict]:
        entry = self.route_map.get(profile, {}).get(route_id)
        if entry is None:
            return None
        route_dir, _ = entry
        icl_path = os.path.join(route_dir, "icl_examples.json.gz")
        try:
            with gzip.open(icl_path, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            return {"error": f"failed to load {icl_path}: {e}"}
        examples = d.get(str(frame), [])

        # Query frame: route_dir/rgb/<frame>.jpg
        query_rgb = os.path.join(route_dir, "rgb", f"{int(frame):04d}.jpg")
        query_resolved = _normalize_path(query_rgb)
        query_b64 = _encode_image_b64(query_resolved, size=(960, 540), quality=82)

        # Resolve and base64 each example's rgb_path
        out_examples = []
        for ex in examples:
            ex2 = dict(ex)
            rp = ex2.get("rgb_path", "")
            resolved = _normalize_path(rp) if rp else ""
            ex2["rgb_path_resolved"] = resolved
            ex2["image_b64"] = _encode_image_b64(resolved, size=(960, 540), quality=80) if resolved else None
            out_examples.append(ex2)

        return {
            "profile":         profile,
            "route_id":        route_id,
            "frame":           int(frame),
            "query_rgb_path":  query_resolved,
            "query_image_b64": query_b64,
            "n_examples":      len(out_examples),
            "examples":        out_examples,
        }


# 3. FLASK APP

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PMD-LLM RAG Retrieval</title>
<style>
  :root {
    --bg: #0a0c0f; --panel: #111318; --border: #1e2330;
    --accent: #00e5ff; --accent2: #7c3aed; --text: #e2e8f0; --muted: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; min-height: 100vh; }
  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
  header { padding: 24px 0; border-bottom: 1px solid var(--border); margin-bottom: 32px; }
  h1 { font-size: 28px; color: #fff; }
  h1 span { color: var(--accent); }
  .subtitle { font-size: 13px; color: var(--muted); margin-top: 4px; }
  .main-grid { display: grid; grid-template-columns: 380px 1fr; gap: 24px; align-items: start; }
  @media (max-width: 800px) { .main-grid { grid-template-columns: 1fr; } }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }
  .panel-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); margin-bottom: 16px; }
  .upload-zone { border: 1.5px dashed var(--border); border-radius: 8px; padding: 24px; text-align: center; cursor: pointer; margin-bottom: 20px; min-height: 160px; display: flex; align-items: center; justify-content: center; }
  .upload-zone:hover { border-color: var(--accent); }
  .upload-zone.has-image { border-style: solid; border-color: var(--accent); padding: 0; overflow: hidden; }
  .upload-zone img { width: 100%; border-radius: 7px; display: block; }
  .upload-text { font-size: 13px; color: var(--muted); }
  #file-input { display: none; }
  .field { margin-bottom: 14px; }
  label { display: block; font-size: 11px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; letter-spacing: 0.08em; }
  input, select { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 9px 12px; color: var(--text); font-size: 13px; outline: none; }
  input:focus, select:focus { border-color: var(--accent); }
  .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .btn-query { width: 100%; padding: 13px; background: var(--accent); color: #000; font-size: 13px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; border: none; border-radius: 8px; cursor: pointer; margin-top: 8px; }
  .btn-query:hover { opacity: 0.85; }
  .btn-query:disabled { opacity: 0.4; cursor: not-allowed; }
  .result-card { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 12px; display: grid; grid-template-columns: 200px 1fr; }
  .result-card:hover { border-color: var(--accent); }
  .result-thumb { background: #070809; display: flex; align-items: center; justify-content: center; position: relative; min-height: 112px; }
  .result-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .rank-badge { position: absolute; top: 8px; left: 8px; background: var(--accent); color: #000; font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 4px; }
  .result-info { padding: 14px 16px; }
  .score-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .score-val { color: var(--accent); }
  .score-bar { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 10px; }
  .score-fill { height: 100%; background: linear-gradient(90deg, var(--accent2), var(--accent)); border-radius: 2px; }
  .frame-id { font-size: 11px; color: var(--muted); word-break: break-all; margin-bottom: 10px; }
  .state-chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip { background: var(--border); border-radius: 4px; padding: 3px 8px; font-size: 11px; color: var(--text); }
  .chip.cmd { background: rgba(124,58,237,0.25); color: #c4b5fd; }
  .empty-state { text-align: center; padding: 64px 24px; color: var(--muted); }
  .error-msg { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); color: #fca5a5; border-radius: 8px; padding: 12px; font-size: 13px; margin-bottom: 16px; }
  .results-count { font-size: 11px; color: var(--muted); }
  @media (max-width: 500px) { .result-card { grid-template-columns: 1fr; } }
  .tab-btn { background:none; border:1px solid var(--border); color:var(--muted); padding:8px 18px; border-radius:6px; cursor:pointer; font-size:13px; }
  .tab-btn.active { border-color:var(--accent); color:var(--accent); background:rgba(0,229,255,0.07); }
  .tab-content { display:none; }
  .tab-content.active { display:block; }
  .browse-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; }
  .browse-card { background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; cursor:pointer; }
  .browse-card:hover { border-color:var(--accent); }
  .browse-card img { width:100%; display:block; }
  .browse-card-info { padding:10px 12px; }
  .browse-card-id { font-size:10px; color:var(--muted); word-break:break-all; margin-bottom:6px; }
  .filter-bar { display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; align-items:flex-end; }
  .filter-bar input, .filter-bar select { width:auto; flex:1; min-width:140px; }
  .pagination { display:flex; gap:8px; align-items:center; margin-top:20px; justify-content:center; }
  .page-btn { background:var(--panel); border:1px solid var(--border); color:var(--text); padding:6px 14px; border-radius:6px; cursor:pointer; }
  .page-btn:disabled { opacity:0.3; cursor:not-allowed; }
  .stat-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:20px; }
  .stat-card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:20px; }
  .stat-title { font-size:13px; color:var(--accent); margin-bottom:14px; text-transform:uppercase; letter-spacing:.08em; }
  .stat-row { display:flex; justify-content:space-between; font-size:12px; padding:4px 0; border-bottom:1px solid var(--border); }
  .modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,.8); z-index:100; overflow-y:auto; padding:24px; }
  .modal.open { display:flex; align-items:flex-start; justify-content:center; }
  .modal-box { background:var(--panel); border:1px solid var(--border); border-radius:12px; max-width:900px; width:100%; padding:24px; position:relative; }
  .modal-close { position:absolute; top:16px; right:16px; background:none; border:none; color:var(--muted); font-size:20px; cursor:pointer; }
  .meta-section { margin-top:16px; }
  .meta-label { font-size:11px; color:var(--accent); text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
  .meta-text { font-size:12px; color:var(--text); background:var(--bg); padding:10px; border-radius:6px; white-space:pre-wrap; max-height:200px; overflow-y:auto; }
  .qa-item { margin-bottom:8px; font-size:12px; }
  .qa-q { color:var(--muted); }
  .qa-a { color:var(--text); margin-top:2px; }
  /* ── ICL viewer ───────────────────────────────────────────────────── */
  .icl-layout { display:grid; grid-template-columns: minmax(380px, 0.9fr) minmax(0, 1.4fr); gap:16px; align-items:start; }
  @media (max-width: 980px) { .icl-layout { grid-template-columns: 1fr; } }
  .icl-left { position:sticky; top:18px; }
  .icl-right { display:flex; flex-direction:column; gap:14px; }
  .icl-card { background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.005)); border:1px solid var(--border); border-radius:10px; overflow:hidden; transition: border-color .12s ease; }
  .icl-card:hover { border-color: rgba(0,229,255,0.35); }
  .icl-card.icl-query { border-color: rgba(0,229,255,0.35); box-shadow: 0 0 0 1px rgba(0,229,255,0.10), 0 8px 24px rgba(0,0,0,0.25); }
  .icl-card-head { display:flex; align-items:center; gap:10px; padding:10px 12px; border-bottom:1px solid var(--border); background:rgba(255,255,255,0.015); }
  .icl-card-title { font-size:13px; color:var(--text); font-weight:600; flex:0 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .icl-badge { font-size:10px; font-weight:700; letter-spacing:.10em; padding:3px 8px; border-radius:4px; background:rgba(0,229,255,0.13); color:var(--accent); border:1px solid rgba(0,229,255,0.30); }
  .icl-badge-ex { background:rgba(255,255,255,0.04); color:#cfd8dc; border-color:var(--border); }
  .icl-card-img { background:#000; }
  .icl-img { display:block; width:100%; cursor:zoom-in; transition: transform .15s ease, filter .15s ease; }
  .icl-img:hover { filter: brightness(1.06); }
  .icl-img-empty { display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:12px; background:#070809; }
  .icl-meta-grid { display:grid; grid-template-columns: 110px 1fr; column-gap:14px; row-gap:6px; padding:12px 14px 14px; font-size:12px; }
  .icl-meta-k { color:var(--muted); text-transform:uppercase; font-size:10.5px; letter-spacing:.06em; padding-top:1px; }
  .icl-meta-v { color:var(--text); min-width:0; }
  .icl-lightbox { position:fixed; inset:0; background:rgba(2,4,6,0.92); display:none; align-items:center; justify-content:center; z-index:9999; padding:32px; cursor:zoom-out; flex-direction:column; }
  .icl-lightbox.open { display:flex; }
  .icl-lightbox img { max-width: 96vw; max-height: 88vh; border-radius:6px; box-shadow: 0 20px 60px rgba(0,0,0,0.6); cursor:default; }
  .icl-lightbox-caption { position:absolute; bottom:18px; left:0; right:0; text-align:center; color:#aab; font-size:11px; font-family:monospace; padding:0 24px; word-break:break-all; }
  .icl-lightbox-close { position:absolute; top:18px; right:18px; background:rgba(255,255,255,0.08); color:#fff; border:1px solid var(--border); width:36px; height:36px; border-radius:50%; cursor:pointer; font-size:16px; display:flex; align-items:center; justify-content:center; }
  .icl-lightbox-close:hover { background:rgba(255,255,255,0.16); }
</style>
</head>
<body onload="initProfiles()">
<div class="container">
  <header>
    <h1>Persona-Drive <span>RAG</span></h1>
    <p class="subtitle">Query by image · Browse index · Inspect full metadata</p>
    <div style="display:flex;gap:16px;margin-top:16px">
      <button class="tab-btn active" onclick="switchTab('query')">&#x1F50D; Query</button>
      <button class="tab-btn" onclick="switchTab('browse')">&#x1F4C2; Browse Index</button>
      <button class="tab-btn" onclick="switchTab('icl')" id="tab-btn-icl" style="display:none">&#x1F517; ICL</button>
      <button class="tab-btn" onclick="switchTab('stats')">&#x1F4CA; Stats</button>
    </div>
  </header>
  <div class="tab-content active" id="tab-query">
  <div class="main-grid">
    <div>
      <div class="panel">
        <div class="panel-label">Query Input</div>
        <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()">
          <div id="upload-placeholder"><div style="font-size:32px;margin-bottom:8px">&#x1f5bc;</div><div class="upload-text"><strong>Click to upload</strong> a driving frame</div></div>
          <img id="preview-img" style="display:none" alt="preview">
        </div>
        <input type="file" id="file-input" accept="image/*">
        <div class="field-row">
          <div class="field"><label>Speed (m/s)</label><input type="number" id="speed" value="12.0" step="0.1"></div>
          <div class="field"><label>Throttle</label><input type="number" id="throttle" value="0.5" step="0.01"></div>
        </div>
        <div class="field-row">
          <div class="field"><label>Steer</label><input type="number" id="steer" value="0.0" step="0.01"></div>
          <div class="field"><label>Brake</label><select id="brake"><option value="0">Not pressed</option><option value="1">Pressed</option></select></div>
        </div>
        <div class="field">
          <label>Command</label>
          <select id="command">
            <option>Follow Lane</option>
            <option>Change Lane Left</option>
            <option>Change Lane Right</option>
            <option>Go Left</option>
            <option>Go Right</option>
            <option>Go Straight</option>
          </select>
        </div>
        <div class="field">
          <label>Driving Profile</label>
          <select id="profile"></select>
        </div>
        <button class="btn-query" id="query-btn" onclick="runQuery()">Retrieve Top-5</button>
      </div>
    </div>
    <div>
      <div class="panel">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
          <div class="panel-label" style="margin-bottom:0">Results</div>
          <div class="results-count" id="results-count"></div>
        </div>
        <div id="error-box" style="display:none"></div>
        <div id="results-container">
          <div class="empty-state"><div style="font-size:48px;opacity:0.3;margin-bottom:16px">&#x1f697;</div><div>Upload a frame and submit to retrieve similar scenarios</div></div>
        </div>
      </div>
    </div>
  </div>
  </div><!-- end tab-query -->

  <!-- BROWSE TAB -->
  <div class="tab-content" id="tab-browse">
    <div class="panel">
      <div class="panel-label">Browse Index</div>
      <div class="filter-bar">
        <div class="field" style="margin:0"><label>Profile</label><select id="br-profile" onchange="browseLoad(0)"></select></div>
        <div class="field" style="margin:0"><label>Scenario filter</label><input id="br-scenario" placeholder="e.g. ControlLoss" oninput="browseLoad(0)"></div>
        <div class="field" style="margin:0"><label>Command</label><select id="br-command" onchange="browseLoad(0)"><option value="">All</option><option>Follow Lane</option><option>Change Lane Left</option><option>Change Lane Right</option><option>Go Left</option><option>Go Right</option><option>Go Straight</option></select></div>
        <div style="font-size:12px;color:var(--muted);align-self:flex-end;padding-bottom:4px" id="br-count"></div>
      </div>
      <div class="browse-grid" id="browse-grid"><div class="empty-state">Loading...</div></div>
      <div class="pagination">
        <button class="page-btn" id="br-prev" onclick="browsePage(-1)">← Prev</button>
        <span id="br-pageinfo" style="font-size:12px;color:var(--muted)"></span>
        <button class="page-btn" id="br-next" onclick="browsePage(1)">Next →</button>
      </div>
    </div>
  </div>

  <!-- ICL TAB -->
  <div class="tab-content" id="tab-icl">
    <div class="panel">
      <div class="panel-label">In-Context Learning Examples</div>
      <div class="filter-bar">
        <div class="field" style="margin:0"><label>Profile</label><select id="icl-profile" onchange="iclProfileChanged()"></select></div>
        <div class="field" style="margin:0;flex:1"><label>Route filter</label><input id="icl-route-filter" placeholder="e.g. ControlLoss/Town01" oninput="iclLoadRoutes(0)"></div>
        <div style="font-size:12px;color:var(--muted);align-self:flex-end;padding-bottom:4px" id="icl-counts"></div>
      </div>
      <div class="field-row">
        <div class="field" style="flex:2"><label>Route</label><select id="icl-route" onchange="iclRouteChanged()"></select></div>
        <div class="field" style="flex:1"><label>Frame</label><select id="icl-frame" onchange="iclFrameChanged()"></select></div>
      </div>
      <div class="pagination" style="margin-top:8px">
        <button class="page-btn" id="icl-prev" onclick="iclRoutePage(-1)">← Prev routes</button>
        <span id="icl-pageinfo" style="font-size:12px;color:var(--muted)"></span>
        <button class="page-btn" id="icl-next" onclick="iclRoutePage(1)">Next routes →</button>
      </div>
      <div id="icl-display" style="margin-top:18px"><div class="empty-state">Pick a profile, route, and frame to see ICL examples.</div></div>
    </div>
  </div>

  <!-- STATS TAB -->
  <div class="tab-content" id="tab-stats">
    <div class="panel">
      <div class="panel-label">Index Statistics</div>
      <div class="stat-grid" id="stat-grid"><div class="empty-state">Loading...</div></div>
    </div>
  </div>

  <!-- DETAIL MODAL -->
  <div class="modal" id="detail-modal">
    <div class="modal-box">
      <button class="modal-close" onclick="closeModal()">✕</button>
      <div id="modal-content"></div>
    </div>
  </div>

</div>
<script>
let _browseState = {page:0, profile:'normal', scenario:'', command:''};
let _iclState = {profile:'', page:0, pages:0, total:0, route:'', frames:[], loaded:false};

async function initProfiles() {
  const stats = await fetch('/api/stats').then(r=>r.json()).catch(e=>({}));
  const profiles = Object.keys(stats || {});
  if (profiles.length === 0) return;

  const selects = ['profile', 'br-profile'];
  for (const id of selects) {
    const select = document.getElementById(id);
    if (!select) continue;
    select.innerHTML = '';
    for (const p of profiles) {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p.charAt(0).toUpperCase() + p.slice(1);
      select.appendChild(opt);
    }
    select.value = profiles[0];
  }
  _browseState.profile = profiles[0];

  // Discover ICL profiles. The ICL tab only appears if at least one profile
  // has scanned ICL data on disk.
  try {
    const iclMeta = await fetch('/api/icl/profiles').then(r=>r.json());
    const iclProfiles = Object.entries(iclMeta || {}).filter(([k,v]) => v > 0);
    if (iclProfiles.length) {
      const sel = document.getElementById('icl-profile');
      sel.innerHTML = '';
      for (const [name, count] of iclProfiles) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = `${name.charAt(0).toUpperCase()+name.slice(1)} (${count} routes)`;
        sel.appendChild(opt);
      }
      sel.value = iclProfiles[0][0];
      document.getElementById('tab-btn-icl').style.display = '';
      _iclState.profile = iclProfiles[0][0];
    }
  } catch (e) { /* ICL endpoint unreachable — leave tab hidden */ }
}

function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  event.target.classList.add('active');
  if (name==='browse' && document.getElementById('browse-grid').children[0]?.classList.contains('empty-state')) browseLoad(0);
  if (name==='stats') loadStats();
  if (name==='icl' && !_iclState.loaded) iclLoadRoutes(0);
}

async function browseLoad(page) {
  _browseState.page = page;
  _browseState.profile  = document.getElementById('br-profile').value;
  _browseState.scenario = document.getElementById('br-scenario').value;
  _browseState.command  = document.getElementById('br-command').value;
  const params = new URLSearchParams({profile:_browseState.profile, page, scenario:_browseState.scenario, command:_browseState.command});
  const data = await fetch('/api/browse?'+params).then(r=>r.json());
  document.getElementById('br-count').textContent = data.total + ' frames';
  document.getElementById('br-pageinfo').textContent = `Page ${data.page+1} / ${data.pages||1}`;
  document.getElementById('br-prev').disabled = data.page === 0;
  document.getElementById('br-next').disabled = data.page >= (data.pages||1)-1;
  const g = document.getElementById('browse-grid');
  if (!data.items.length) { g.innerHTML='<div class="empty-state">No frames match filter.</div>'; return; }
  g.innerHTML = data.items.map(m => {
    const img = m.image_b64 ? `<img src="data:image/jpeg;base64,${m.image_b64}">` : '<div style="height:112px;background:#070809;display:flex;align-items:center;justify-content:center;font-size:24px;opacity:.2">🖼</div>';
    const st = m.state||{};
    return `<div class="browse-card" onclick="showDetail('${m.frame_id.replace(/'/g,"\'")}','${_browseState.profile}')">
      ${img}
      <div class="browse-card-info">
        <div class="browse-card-id">${m.frame_id}</div>
        <div class="state-chips">
          ${m.scenario?`<span class="chip cmd">${m.scenario}</span>`:''}
          ${st.command?`<span class="chip">${st.command}</span>`:''}
          ${st.speed!=null?`<span class="chip">spd ${parseFloat(st.speed).toFixed(1)}</span>`:''}
          ${m.vqa_count?`<span class="chip">${m.vqa_count} QAs</span>`:''}
          ${m.has_history?`<span class="chip">+history</span>`:''}
        </div>
      </div>
    </div>`;
  }).join('');
}

function browsePage(dir) { browseLoad(_browseState.page + dir); }

async function loadStats() {
  const data = await fetch('/api/stats').then(r=>r.json());
  const g = document.getElementById('stat-grid');
  g.innerHTML = Object.entries(data).map(([profile, s]) => `
    <div class="stat-card">
      <div class="stat-title">${profile} — ${s.total.toLocaleString()} frames</div>
      <div class="meta-label">By Scenario</div>
      ${Object.entries(s.scenarios).slice(0,10).map(([k,v])=>`<div class="stat-row"><span>${k}</span><span>${v.toLocaleString()}</span></div>`).join('')}
      <div class="meta-label" style="margin-top:12px">By Command</div>
      ${Object.entries(s.commands).map(([k,v])=>`<div class="stat-row"><span>${k}</span><span>${v.toLocaleString()}</span></div>`).join('')}
    </div>`).join('');
}

async function showDetail(frameId, profile) {
  document.getElementById('detail-modal').classList.add('open');
  document.getElementById('modal-content').innerHTML = '<div style="text-align:center;padding:40px;color:var(--muted)">Loading...</div>';
  const data = await fetch(`/api/frame?profile=${profile}&frame_id=${encodeURIComponent(frameId)}`).then(r=>r.json());
  if (!data || data.error) { document.getElementById('modal-content').innerHTML = '<p style="color:#fca5a5">Not found</p>'; return; }
  const st = data.state||{}, s1=data.state_1_back||{}, s2=data.state_2_back||{};
  const vqa = data.vqa||{};
  const qaHtml = ['perception','prediction','planning','behavior'].map(sec => {
    const items = vqa[sec]||[];
    if (!items.length) return '';
    return `<div class="meta-label">${sec.toUpperCase()} (${items.length})</div>`+
      items.map(q=>`<div class="qa-item"><div class="qa-q">Q: ${q.Q}</div><div class="qa-a">A: ${q.A}</div></div>`).join('');
  }).join('');
  document.getElementById('modal-content').innerHTML = `
    ${data.image_b64?`<img src="data:image/jpeg;base64,${data.image_b64}" style="width:100%;border-radius:8px;margin-bottom:16px">` : ''}
    <div style="font-size:11px;color:var(--muted);word-break:break-all;margin-bottom:12px">${data.frame_id}</div>
    <div class="state-chips" style="margin-bottom:16px">
      ${data.scenario?`<span class="chip cmd">${data.scenario}</span>`:''}
      ${st.command?`<span class="chip">${st.command}</span>`:''}
      ${st.speed!=null?`<span class="chip">spd ${parseFloat(st.speed).toFixed(1)}</span>`:''}
      ${st.throttle!=null?`<span class="chip">thr ${parseFloat(st.throttle).toFixed(2)}</span>`:''}
      ${st.steer!=null?`<span class="chip">str ${parseFloat(st.steer).toFixed(2)}</span>`:''}
    </div>
    <div class="meta-section"><div class="meta-label">Historical State</div>
      <div class="meta-text">-1: spd=${s1.speed!=null?parseFloat(s1.speed).toFixed(2):'?'} thr=${s1.throttle!=null?parseFloat(s1.throttle).toFixed(2):'?'} str=${s1.steer!=null?parseFloat(s1.steer).toFixed(2):'?'}
-2: spd=${s2.speed!=null?parseFloat(s2.speed).toFixed(2):'?'} thr=${s2.throttle!=null?parseFloat(s2.throttle).toFixed(2):'?'} str=${s2.steer!=null?parseFloat(s2.steer).toFixed(2):'?'}</div></div>
    ${data.commentary?`<div class="meta-section"><div class="meta-label">Commentary</div><div class="meta-text">${data.commentary}</div></div>`:''}
    ${(data.future_waypoints && data.future_waypoints.length)?`<div class="meta-section"><div class="meta-label">Future Waypoints (ego-local, m)</div><div class="meta-text">${data.future_waypoints.map((wp,i)=>`+${i+1}: [${(+wp[0]).toFixed(2)}, ${(+wp[1]).toFixed(2)}]`).join('  ·  ')}</div></div>`:''}
    ${(data.target_point && data.target_point.length)?`<div class="meta-section"><div class="meta-label">Target Point</div><div class="meta-text">[${data.target_point.map(v=>(+v).toFixed(2)).join(', ')}]${data.target_point_next?`  →  next: [${data.target_point_next.map(v=>(+v).toFixed(2)).join(', ')}]`:''}</div></div>`:''}
    ${qaHtml?`<div class="meta-section"><div class="meta-label">VQA</div>${qaHtml}</div>`:''}
    ${data.sentence?`<div class="meta-section"><div class="meta-label">Full Sentence</div><div class="meta-text" style="white-space:pre-wrap;font-family:monospace;font-size:11px">${data.sentence}</div></div>`:''}
  `;
}
function closeModal() { document.getElementById('detail-modal').classList.remove('open'); }
document.getElementById('detail-modal').addEventListener('click', e => { if(e.target===e.currentTarget) closeModal(); });

// ── ICL viewer ─────────────────────────────────────────────────────────────
function iclProfileChanged() {
  _iclState.profile = document.getElementById('icl-profile').value;
  _iclState.page = 0;
  iclLoadRoutes(0);
}
async function iclLoadRoutes(page) {
  const profile = document.getElementById('icl-profile').value;
  if (!profile) return;
  const filter = document.getElementById('icl-route-filter').value || '';
  const params = new URLSearchParams({profile, page, filter});
  const data = await fetch('/api/icl/routes?'+params.toString()).then(r=>r.json());
  if (data.error) { document.getElementById('icl-display').innerHTML = `<div class="empty-state">${data.error}</div>`; return; }
  _iclState.profile = profile; _iclState.page = data.page; _iclState.pages = data.pages; _iclState.total = data.total; _iclState.loaded = true;
  document.getElementById('icl-counts').textContent = `${data.total} routes`;
  document.getElementById('icl-pageinfo').textContent = data.pages ? `page ${data.page+1} / ${data.pages}` : '';
  document.getElementById('icl-prev').disabled = data.page <= 0;
  document.getElementById('icl-next').disabled = data.page >= data.pages - 1;
  const sel = document.getElementById('icl-route');
  sel.innerHTML = '';
  if (!data.routes.length) {
    document.getElementById('icl-display').innerHTML = '<div class="empty-state">No routes match the filter.</div>';
    document.getElementById('icl-frame').innerHTML = '';
    return;
  }
  for (const r of data.routes) {
    const opt = document.createElement('option'); opt.value = r.id;
    opt.textContent = `${r.id}  (${r.frames} frames)`;
    sel.appendChild(opt);
  }
  sel.value = data.routes[0].id;
  iclRouteChanged();
}
function iclRoutePage(delta) {
  const p = (_iclState.page || 0) + delta;
  if (p < 0 || p >= _iclState.pages) return;
  iclLoadRoutes(p);
}
async function iclRouteChanged() {
  const profile = document.getElementById('icl-profile').value;
  const route = document.getElementById('icl-route').value;
  if (!profile || !route) return;
  const params = new URLSearchParams({profile, route});
  const data = await fetch('/api/icl/frames?'+params.toString()).then(r=>r.json());
  const fSel = document.getElementById('icl-frame');
  fSel.innerHTML = '';
  if (data.error || !data.frames || !data.frames.length) {
    document.getElementById('icl-display').innerHTML = '<div class="empty-state">No frames have ICL data for this route.</div>';
    return;
  }
  _iclState.route = route; _iclState.frames = data.frames;
  for (const fn of data.frames) {
    const opt = document.createElement('option'); opt.value = fn; opt.textContent = String(fn).padStart(4,'0');
    fSel.appendChild(opt);
  }
  fSel.value = data.frames[0];
  iclFrameChanged();
}
async function iclFrameChanged() {
  const profile = document.getElementById('icl-profile').value;
  const route = document.getElementById('icl-route').value;
  const frame = document.getElementById('icl-frame').value;
  if (!profile || !route || !frame) return;
  document.getElementById('icl-display').innerHTML = '<div class="empty-state">Loading...</div>';
  const params = new URLSearchParams({profile, route, frame});
  const data = await fetch('/api/icl/frame?'+params.toString()).then(r=>r.json());
  if (data.error) { document.getElementById('icl-display').innerHTML = `<div class="empty-state">Error: ${data.error}</div>`; return; }
  iclRender(data);
}
function iclRender(data) {
  const fmtPt = p => Array.isArray(p) ? `[${p.map(v=>(+v).toFixed(2)).join(', ')}]` : '—';
  const fmtHist = a => Array.isArray(a) ? a.map(v=>(+v).toFixed(2)).join(' / ') : '—';
  const fmtWps = ws => (Array.isArray(ws) && ws.length)
    ? ws.map((w,i)=>`+${i+1}: [${(+w[0]).toFixed(2)}, ${(+w[1]).toFixed(2)}]`).join(' · ')
    : '—';
  const lastSeg = p => { if (!p) return ''; const parts = String(p).split('/'); return parts.slice(-3).join('/'); };
  const fileName = p => { if (!p) return ''; const parts = String(p).split('/'); return parts[parts.length-1]; };
  const placeholder = (h='200px') => `<div class="icl-img-empty" style="height:${h}">image not found on disk</div>`;
  const escAttr = s => String(s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
  const imgHtml = (b64, path, h='auto') => b64
    ? `<img class="icl-img" src="data:image/jpeg;base64,${b64}" data-full-path="${escAttr(path)}" onclick="iclLightbox(this)" style="${h!=='auto'?`height:${h};width:100%;object-fit:cover`:'width:100%'}">`
    : placeholder();

  // ── Query frame block (left, sticky) ─────────────────────────────────────
  const queryBlock = `
    <div class="icl-card icl-query">
      <div class="icl-card-head">
        <div class="icl-badge">QUERY</div>
        <div class="icl-card-title">Frame ${String(data.frame).padStart(4,'0')}</div>
      </div>
      <div class="icl-card-img">${imgHtml(data.query_image_b64, data.query_rgb_path)}</div>
      <div class="icl-meta-grid">
        <div class="icl-meta-k">profile</div><div class="icl-meta-v"><span class="chip cmd">${data.profile}</span></div>
        <div class="icl-meta-k">route</div><div class="icl-meta-v" style="font-family:monospace;font-size:11px;word-break:break-all">${lastSeg(data.route_id)}</div>
        <div class="icl-meta-k">frame</div><div class="icl-meta-v">${String(data.frame).padStart(4,'0')}.jpg</div>
      </div>
    </div>`;

  // ── ICL example blocks (right column, stacked) ───────────────────────────
  const exBlocks = (data.examples || []).map((ex, i) => {
    const cmd = ex.command || '—';
    const sc  = ex.scenario_name || '—';
    return `
      <div class="icl-card icl-example">
        <div class="icl-card-head">
          <div class="icl-badge icl-badge-ex">ICL #${i+1}</div>
          <div class="icl-card-title">${sc}</div>
          <span class="chip" style="margin-left:auto">${cmd}</span>
        </div>
        <div class="icl-card-img">${imgHtml(ex.image_b64, ex.rgb_path_resolved||ex.rgb_path, '180px')}</div>
        <div class="icl-meta-grid">
          <div class="icl-meta-k">target</div>
            <div class="icl-meta-v" style="font-family:monospace;font-size:11px">
              ${fmtPt(ex.target_point)}${ex.target_point_next?` <span style="color:var(--muted)">→</span> ${fmtPt(ex.target_point_next)}`:''}
            </div>
          <div class="icl-meta-k">speed hist</div><div class="icl-meta-v" style="font-family:monospace;font-size:11px">${fmtHist(ex.speed_history)} <span style="color:var(--muted);font-size:10px">m/s (t-2 / t-1 / t)</span></div>
          <div class="icl-meta-k">throttle hist</div><div class="icl-meta-v" style="font-family:monospace;font-size:11px">${fmtHist(ex.throttle_history)}</div>
          <div class="icl-meta-k">commentary</div><div class="icl-meta-v" style="white-space:pre-wrap;line-height:1.45">${ex.commentary||'—'}</div>
          <div class="icl-meta-k">waypoints</div><div class="icl-meta-v" style="font-family:monospace;font-size:10.5px;line-height:1.5;color:#cfd8dc">${fmtWps(ex.waypoints)}</div>
          <div class="icl-meta-k">source</div><div class="icl-meta-v" style="font-family:monospace;font-size:10.5px;color:var(--muted);word-break:break-all">${lastSeg(ex.rgb_path_resolved||ex.rgb_path)}</div>
        </div>
      </div>`;
  }).join('');
  const rightCol = (data.examples && data.examples.length)
    ? exBlocks
    : '<div class="empty-state">No ICL examples were saved for this frame (within the first SKIP_FRAMES of the route, or retrieval skipped).</div>';

  document.getElementById('icl-display').innerHTML = `
    <div class="icl-layout">
      <div class="icl-left">${queryBlock}</div>
      <div class="icl-right">${rightCol}</div>
    </div>`;
}

// Lightbox: click any thumbnail to open it full-screen.
function iclLightbox(imgEl) {
  const src  = imgEl.getAttribute('src');
  const path = imgEl.getAttribute('data-full-path') || '';
  let lb = document.getElementById('icl-lightbox');
  if (!lb) {
    lb = document.createElement('div');
    lb.id = 'icl-lightbox';
    lb.className = 'icl-lightbox';
    lb.onclick = () => lb.classList.remove('open');
    document.body.appendChild(lb);
  }
  lb.innerHTML = `
    <button class="icl-lightbox-close" onclick="event.stopPropagation();document.getElementById('icl-lightbox').classList.remove('open')">✕</button>
    <img src="${src}" alt="" onclick="event.stopPropagation()">
    <div class="icl-lightbox-caption">${path}</div>`;
  lb.classList.add('open');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    const lb = document.getElementById('icl-lightbox');
    if (lb && lb.classList.contains('open')) lb.classList.remove('open');
  }
});
</script>
<script>
let uploadedFile = null;
document.getElementById('file-input').addEventListener('change', (e) => {
  const file = e.target.files[0]; if (!file) return; uploadedFile = file;
  const reader = new FileReader();
  reader.onload = (ev) => { const img = document.getElementById('preview-img'); img.src = ev.target.result; img.style.display = 'block'; document.getElementById('upload-placeholder').style.display = 'none'; document.getElementById('upload-zone').classList.add('has-image'); };
  reader.readAsDataURL(file);
});
async function runQuery() {
  const errBox = document.getElementById('error-box'); errBox.style.display = 'none';
  if (!uploadedFile) { showError('Please upload a driving frame first.'); return; }
  const btn = document.getElementById('query-btn'); btn.disabled = true; btn.textContent = 'Retrieving...';
  document.getElementById('results-count').textContent = '';
  const fd = new FormData();
  fd.append('image', uploadedFile);
  fd.append('speed', document.getElementById('speed').value);
  fd.append('throttle', document.getElementById('throttle').value);
  fd.append('steer', document.getElementById('steer').value);
  fd.append('brake', document.getElementById('brake').value);
  fd.append('command', document.getElementById('command').value);
  fd.append('profile', document.getElementById('profile').value);
  try {
    const resp = await fetch('/query', { method: 'POST', body: fd });
    const data = await resp.json();
    if (!resp.ok || data.error) { showError(data.error || 'Server error'); return; }
    renderResults(data.results);
    document.getElementById('results-count').textContent = data.results.length + ' results - ' + data.elapsed_ms + 'ms';
  } catch (err) { showError('Network error: ' + err.message); }
  finally { btn.disabled = false; btn.textContent = 'Retrieve Top-5'; }
}
function showError(msg) { const b = document.getElementById('error-box'); b.innerHTML = '<div class="error-msg">' + msg + '</div>'; b.style.display = 'block'; }
function renderResults(results) {
  const c = document.getElementById('results-container');
  if (!results || !results.length) { c.innerHTML = '<div class="empty-state">No results.</div>'; return; }
  const mx = results[0].score, mn = results[results.length-1].score, rng = mx - mn || 0.001;
  c.innerHTML = results.map(r => {
    const bw = 60 + ((r.score - mn) / rng) * 40, st = r.state || {};
    const img = r.image_b64 ? '<img src="data:image/jpeg;base64,' + r.image_b64 + '">' : '<div style="font-size:28px;opacity:0.2">&#x1f5bc;</div>';
    const chips = [
      st.command ? '<span class="chip cmd">' + st.command + '</span>' : '',
      st.speed != null ? '<span class="chip">spd ' + parseFloat(st.speed).toFixed(1) + '</span>' : '',
      st.throttle != null ? '<span class="chip">thr ' + parseFloat(st.throttle).toFixed(2) + '</span>' : '',
      st.steer != null ? '<span class="chip">str ' + parseFloat(st.steer).toFixed(2) + '</span>' : '',
    ].filter(Boolean).join('');
    return '<div class="result-card"><div class="result-thumb">' + img + '<div class="rank-badge">#' + r.rank + '</div></div><div class="result-info"><div class="score-label"><span>Similarity</span><span class="score-val">' + r.score.toFixed(4) + '</span></div><div class="score-bar"><div class="score-fill" style="width:' + bw + '%"></div></div><div class="frame-id">' + r.frame_id + '</div><div class="state-chips">' + chips + '</div></div></div>';
  }).join('');
}
</script>
</body>
</html>
"""

app        = Flask(__name__)
engine     = None
icl_engine = None

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/query", methods=["POST"])
def query():
    t0 = time.perf_counter()
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        f = request.files["image"]
        pil_image = Image.open(f.stream).convert("RGB")
        state = {
            "speed":    float(request.form.get("speed",    12.0)),
            "throttle": float(request.form.get("throttle",  0.5)),
            "steer":    float(request.form.get("steer",     0.0)),
            "brake":    int(request.form.get("brake",       0)),
            "command":  request.form.get("command", "Follow Lane"),
        }
        profile = request.form.get("profile", None)
        results = engine.query(pil_image, state, top_k=5, profile=profile)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return jsonify({"results": results, "elapsed_ms": elapsed_ms})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    try:
        return jsonify(engine.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browse")
def api_browse():
    try:
        profile  = request.args.get("profile", None)
        page     = int(request.args.get("page", 0))
        scenario = request.args.get("scenario", None) or None
        command  = request.args.get("command",  None) or None
        return jsonify(engine.browse(profile, page=page, scenario_filter=scenario,
                                     command_filter=command))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/frame")
def api_frame():
    try:
        profile  = request.args.get("profile", None)
        frame_id = request.args.get("frame_id", "")
        result   = engine.get_frame_detail(profile, frame_id)
        if result is None:
            return jsonify({"error": "Frame not found"}), 404
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ICL viewer endpoints
@app.route("/api/icl/profiles")
def api_icl_profiles():
    if icl_engine is None:
        return jsonify({})
    return jsonify(icl_engine.list_profiles())


@app.route("/api/icl/routes")
def api_icl_routes():
    try:
        if icl_engine is None:
            return jsonify({"routes": [], "total": 0, "page": 0, "pages": 0, "page_size": 0})
        profile    = request.args.get("profile", "")
        page       = int(request.args.get("page", 0))
        page_size  = int(request.args.get("page_size", 200))
        filter_str = request.args.get("filter", "") or ""
        return jsonify(icl_engine.list_routes(profile, page=page, page_size=page_size,
                                              filter_str=filter_str))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/icl/frames")
def api_icl_frames():
    try:
        if icl_engine is None:
            return jsonify({"error": "ICL engine not initialised"}), 404
        profile  = request.args.get("profile", "")
        route_id = request.args.get("route", "")
        result = icl_engine.list_frames(profile, route_id)
        if result is None:
            return jsonify({"error": "route not found"}), 404
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/icl/frame")
def api_icl_frame():
    try:
        if icl_engine is None:
            return jsonify({"error": "ICL engine not initialised"}), 404
        profile  = request.args.get("profile", "")
        route_id = request.args.get("route", "")
        frame    = int(request.args.get("frame", 0))
        result = icl_engine.get_frame_icl(profile, route_id, frame)
        if result is None:
            return jsonify({"error": "route not found"}), 404
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def main():
    global engine
    parser = argparse.ArgumentParser(description="PMD-LLM RAG Web App")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--host",        type=str, default="0.0.0.0")
    parser.add_argument("--port",        type=int, default=7860)
    # Multi-profile: --profile name faiss_path meta_path  (repeatable)
    parser.add_argument("--profile", action="append", nargs=3,
                        metavar=("NAME", "FAISS", "META"),
                        help="Add a profile: --profile normal faiss_index/normal/driving_index.faiss "
                             "faiss_index/normal/driving_index.meta.pkl")
    # Auto-discovery base dir. When neither --profile nor --faiss_index is
    # passed, every subdir <name>/ under this directory containing
    # driving_index.{faiss,meta.pkl} is loaded as profile <name>. Subdirs
    # ending in any of _old/_legacy/_bak/_backup/_archive are skipped so
    # archived / rolled-back indices don't auto-load.
    parser.add_argument("--faiss-index-dir", dest="faiss_index_dir",
                        type=str, default="faiss_index",
                        help="Base dir for auto-discovery (default: faiss_index)")
    parser.add_argument("--exclude-profile", action="append", default=[],
                        help="Exclude this profile name from auto-discovery "
                             "(repeatable). Useful to hide one without renaming it.")
    # Legacy single-profile fallback
    parser.add_argument("--faiss_index", type=str, default=None)
    parser.add_argument("--metadata",    type=str, default=None)
    # Override / extend data-root search (colon-separated, like PATH).
    # Anything passed here is prepended to the persistent root.
    parser.add_argument("--data-root", dest="data_root", type=str, default=None,
                        help="Colon-separated list of database roots to prepend "
                             "to the persistent root in the search order.")
    parser.add_argument("--no-icl", dest="no_icl", action="store_true",
                        help="Skip the ICL-viewer scan at startup. The ICL tab "
                             "will be hidden in the UI.")
    args = parser.parse_args()

    if args.data_root:
        existing = os.environ.get("PERSONADRIVE_DATA_ROOT", "")
        os.environ["PERSONADRIVE_DATA_ROOT"] = (
            args.data_root if not existing else f"{args.data_root}:{existing}"
        )
        _reset_data_roots()

    # Auto-discover profiles when no explicit --profile / --faiss_index given.
    # Skips subdirs whose name ends in any of _old/_legacy/_bak/_backup/_archive
    # (so backup indices don't get loaded silently) and any name listed in
    # --exclude-profile.
    if not args.profile and not args.faiss_index and args.faiss_index_dir:
        base = Path(args.faiss_index_dir)
        if base.is_dir():
            skip_suffixes = ("_old", "_legacy", "_bak", "_backup", "_archive")
            excluded      = set(args.exclude_profile or [])
            discovered, skipped = [], []
            for sub in sorted(base.iterdir()):
                if not sub.is_dir():
                    continue
                fp = sub / "driving_index.faiss"
                mp = sub / "driving_index.meta.pkl"
                if not (fp.exists() and mp.exists()):
                    continue
                if any(sub.name.endswith(s) for s in skip_suffixes):
                    skipped.append((sub.name, "archived"))
                    continue
                if sub.name in excluded:
                    skipped.append((sub.name, "excluded"))
                    continue
                discovered.append([sub.name, str(fp), str(mp)])
            if discovered:
                args.profile = discovered
                print(f"[auto] Discovered {len(discovered)} profile(s) under {base}/")
                for d in discovered:
                    print(f"  loaded : {d[0]}")
                for name, why in skipped:
                    print(f"  skipped: {name}  ({why})")
            else:
                print(f"[auto] No usable profiles found under {base}/")

    print(f"\n{'='*50}")
    print(f"  PMD-LLM RAG Web App")
    print(f"{'='*50}")
    for k, v in vars(args).items():
        print(f"  {k:<14}: {v}")
    print(f"  architecture : SigLIP({VISION_DIM})+Control({CONTROL_DIM})->MLP({PROJ_IN_DIM}->{PROJ_OUT_DIM})")
    print(f"  data roots   : {_data_roots() or '(none discovered)'}")
    print(f"{'='*50}\n")

    engine = RAGEngine(args)

    # Optional ICL-viewer engine. Scans for icl_examples.json.gz across the
    # discovered data roots; if no profile yields any routes the UI tab stays
    # hidden. Set --no-icl to skip the scan entirely.
    global icl_engine
    if not args.no_icl:
        icl_engine = IclEngine()
        try:
            icl_engine.scan(verbose=True)
            n_total = sum(icl_engine.list_profiles().values())
            if n_total == 0:
                print("[ICL] no icl_examples.json.gz found — ICL tab will be hidden")
        except Exception as e:
            print(f"[ICL] scan failed: {e}")
            icl_engine = None
    else:
        print("[ICL] disabled via --no-icl")

    print(f"\nServer ready -> http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()