"""
ICL retrieval via FAISS.

Given a query (image + ego state), returns the top-K nearest-neighbour frames
as ICL example dicts in the exact format
`simlingo_training.dataloader.dataset_driving.build_icl_prefix` consumes.

The embedding model architecture is mirrored from
`persona_drive_rag/rag_app.py` / `persona_drive_rag/embed_training.py`:
    SigLIP-base (768) + ControlEncoder (128)
    -> ProjectionMLP 896 -> 1024
    -> L2-normalised -> FAISS inner-product search.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


# Constants - must match persona_drive_rag/rag_app.py / embed_training.py
COMMANDS = [
    "Follow Lane",
    "Change Lane Left",
    "Change Lane Right",
    "Go Left",
    "Go Right",
    "Go Straight",
]
COMMAND2IDX: Dict[str, int] = {c: i for i, c in enumerate(COMMANDS)}
NUM_COMMANDS = len(COMMANDS)

# CARLA's RoadOption integer enum is a *different* ordering than COMMAND2IDX.
# Measurement files (and `state_1_back` / `state_2_back` entries in FAISS
# metadata) store these raw integers; they MUST be converted to strings
# before COMMAND2IDX lookup, otherwise the embedding sees the wrong category.
# This mapping must match `persona_drive_rag/build_faiss_index.py` and
# `embed_training.py` exactly - the FAISS index was built with this order.
CARLA_TO_COMMAND_STRING: Dict[int, str] = {
    1: "Go Left",
    2: "Go Right",
    3: "Go Straight",
    4: "Follow Lane",
    5: "Change Lane Left",
    6: "Change Lane Right",
}


def _command_to_idx(cmd: Any) -> int:
    """Resolve a command (string OR CARLA RoadOption int) to its embedding idx.

    Falls back to 0 ("Follow Lane") for unknown / missing values. This is the
    SAME conversion `build_faiss_index.py` and `embed_training.py` apply, so the FAISS
    index sees the same `command_idx` we send at query time.
    """
    if cmd is None:
        return 0
    if isinstance(cmd, str):
        return COMMAND2IDX.get(cmd, 0)
    if isinstance(cmd, (int, np.integer, bool)):
        cmd_str = CARLA_TO_COMMAND_STRING.get(int(cmd), "Follow Lane")
        return COMMAND2IDX.get(cmd_str, 0)
    return 0


def _command_to_str(cmd: Any) -> str:
    """Resolve a command (string OR CARLA RoadOption int) to its string label.

    Mirrors `_command_to_str` in `scripts/training/generate_icl_examples.py`: strings are
    passed through verbatim (so unknowns survive into the ICL prompt the way
    the source data had them), and only ints / None go through the
    CARLA RoadOption mapping. The strict 0-5 idx clamp lives in
    `_command_to_idx`, not here.
    """
    if cmd is None:
        return "Follow Lane"
    if isinstance(cmd, str):
        return cmd
    if isinstance(cmd, (int, np.integer, bool)):
        return CARLA_TO_COMMAND_STRING.get(int(cmd), "Follow Lane")
    return "Follow Lane"

CONTROL_DIM = 128
VISION_DIM = 768
PROJ_IN_DIM = VISION_DIM + CONTROL_DIM   # 896
PROJ_OUT_DIM = 1024

SIGLIP_MODEL = "google/siglip-base-patch16-224"
SIGLIP_CACHE = os.environ.get("PERSONADRIVE_HF_CACHE") or None   # None: Hugging Face's default cache


# Path normalisation - FAISS metadata frame_paths were captured under whatever
# root the data lived under at index-build time; remap them onto the persistent
# on-share database root.
_PERSISTENT_DATA_ROOT = (
    f"{os.environ.get('PERSONADRIVE_ROOT', '/path/to/PersonaDrive')}/database"
)
_PATH_REWRITES = (
    ("/fast_dataset/data/simlingo/", "/data/simlingo/fast/"),
    ("/slow_dataset/data/simlingo/", "/data/simlingo/slow/"),
)
_DATA_ROOTS_CACHE: Optional[List[str]] = None
_PATH_CACHE: Dict[str, str] = {}


def _discover_data_roots() -> List[str]:
    # PERSONADRIVE_DATA_ROOT (colon-separated) wins; otherwise use the persistent root.
    roots: List[str] = []
    for r in os.environ.get("PERSONADRIVE_DATA_ROOT", "").split(":"):
        r = r.strip().rstrip("/")
        if r and os.path.isdir(r):
            roots.append(r)
    if os.path.isdir(_PERSISTENT_DATA_ROOT):
        roots.append(_PERSISTENT_DATA_ROOT)
    seen, out = set(), []
    for r in roots:
        rp = os.path.realpath(r)
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _data_roots() -> List[str]:
    global _DATA_ROOTS_CACHE
    if _DATA_ROOTS_CACHE is None:
        _DATA_ROOTS_CACHE = _discover_data_roots()
    return _DATA_ROOTS_CACHE


_WARNED_OUTSIDE_ROOTS = False


def _normalize_path(path: str) -> str:
    """Map a (possibly stale) absolute frame path to a real path on this node.

    The configured data roots (PERSONADRIVE_DATA_ROOT, then $PERSONADRIVE_ROOT/database) are searched
    first; the frame's original absolute path is used only when no root holds the frame. (Checking the
    original path first meant that wherever it still existed, e.g. a scratch directory used when the
    index was built, every retrieval read from there even when a local copy was configured.)
    """
    global _WARNED_OUTSIDE_ROOTS
    if not path:
        return path
    cached = _PATH_CACHE.get(path)
    if cached is not None:
        return cached
    roots = _data_roots()
    if any(path.startswith(r + "/") for r in roots) and os.path.exists(path):
        _PATH_CACHE[path] = path
        return path

    candidates = [path]
    for old, new in _PATH_REWRITES:
        if old in path:
            candidates.append(path.replace(old, new))

    for cp in candidates:
        parts = cp.lstrip("/").split("/")
        for root in roots:
            for n_strip in range(len(parts)):
                cand = os.path.join(root, *parts[n_strip:])
                if os.path.exists(cand):
                    _PATH_CACHE[path] = cand
                    return cand
    if os.path.exists(path) and not _WARNED_OUTSIDE_ROOTS:
        _WARNED_OUTSIDE_ROOTS = True
        print(f"[ICL] reading index frames from their original paths outside the data roots "
              f"(e.g. {path}); set PERSONADRIVE_DATA_ROOT to a local copy of the database", flush=True)
    _PATH_CACHE[path] = path
    return path


# Embedding model (same architecture as persona_drive_rag/rag_app.py - kept inline so this
# module has no dependency on the Flask app file).
class ControlEncoder(nn.Module):
    def __init__(self,
                 num_commands: int = NUM_COMMANDS,
                 command_emb_dim: int = 16,
                 scalar_hidden_dim: int = 32,
                 output_dim: int = CONTROL_DIM):
        super().__init__()
        self.scalar_norm = nn.LayerNorm(4)
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
        scalars = torch.stack([speed, throttle, steer, brake.float()], dim=1)
        scalars = self.scalar_norm(scalars)
        scalar_feat = self.scalar_mlp(scalars)
        cmd_feat = self.command_embedding(command_idx)
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
        self.projection = ProjectionMLP()

    def forward(self, image_emb, control):
        c = self.control_encoder(
            speed=control["speed"], throttle=control["throttle"],
            steer=control["steer"], brake=control["brake"],
            command_idx=control["command_idx"],
        )
        return F.normalize(self.projection(torch.cat([image_emb, c], dim=1)),
                           p=2, dim=-1)


# Metadata -> ICL-example mapping (the format build_icl_prefix consumes)


def _safe_state_field(state: Optional[Dict[str, Any]],
                      field: str, default: float = 0.0) -> float:
    if not state:
        return default
    v = state.get(field, default)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def metadata_to_icl_example(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a FAISS metadata entry into an ICL example dict.

    The returned dict has exactly the keys
    `simlingo_training.dataloader.dataset_driving.build_icl_prefix` reads:
        rgb_path, speed_history, throttle_history, command, waypoints,
        target_point, target_point_next, commentary.
    """
    state = meta.get("state") or {}
    s1 = meta.get("state_1_back") or {}
    s2 = meta.get("state_2_back") or {}

    # Use state's own values as the t0 fallback if 1/2-back are missing
    s0_speed = _safe_state_field(state, "speed")
    s0_throttle = _safe_state_field(state, "throttle")

    speed_history = [
        _safe_state_field(s2, "speed", s0_speed),
        _safe_state_field(s1, "speed", s0_speed),
        s0_speed,
    ]
    throttle_history = [
        _safe_state_field(s2, "throttle", s0_throttle),
        _safe_state_field(s1, "throttle", s0_throttle),
        s0_throttle,
    ]

    # `state.command` is usually a string; fall back to top-level numeric
    # `command` field if state.command is missing.
    command = state.get("command")
    if not isinstance(command, str):
        # state.command may be missing or stored as a CARLA RoadOption integer.
        # Try state.command first (may be int), then fall back to top-level
        # meta.command - both are CARLA RoadOption ints when not strings.
        command = _command_to_str(command if command is not None else meta.get("command"))

    waypoints = meta.get("future_waypoints") or []

    # Resolve the rgb_path against currently-mounted data roots (FAISS
    # metadata paths are stale absolute paths from index-build time).
    rgb_path = _normalize_path(meta.get("frame_path", ""))

    out: Dict[str, Any] = {
        "rgb_path": rgb_path,
        "speed_history": [float(x) for x in speed_history],
        "throttle_history": [float(x) for x in throttle_history],
        "command": command,
        "waypoints": [list(map(float, p)) for p in waypoints],
    }
    tp = meta.get("target_point")
    tpn = meta.get("target_point_next")
    if tp is not None:
        out["target_point"] = list(map(float, tp))
    if tpn is not None:
        out["target_point_next"] = list(map(float, tpn))
    commentary = meta.get("commentary")
    if commentary:
        out["commentary"] = commentary
    return out


# IclRetriever - the public interface
PROFILE_NAMES = ("default", "normal", "fast", "slow")


class IclRetriever:
    """Online ICL retrieval: query -> top-K examples in build_icl_prefix format.

    One instance == one profile. The profile name (`normal` / `fast` / `slow`)
    selects which FAISS index + metadata is loaded - these correspond to the
    pre-built indices under `<root>/faiss_index/<profile>/`. To query
    multiple profiles, instantiate multiple `IclRetriever`s.
    """

    def __init__(self,
                 checkpoint: str,
                 profile: str = "normal",
                 root: Optional[str] = None,
                 faiss_index: Optional[str] = None,
                 metadata: Optional[str] = None,
                 index_subdir: str = "faiss_index",
                 index_filename: str = "driving_index.faiss",
                 metadata_filename: str = "driving_index.meta.pkl",
                 device: Optional[Union[str, torch.device]] = None,
                 verbose: bool = True):
        """
        Args:
            checkpoint: Path to the DrivingEmbeddingModel weights (e.g.
                `<root>/checkpoints/best_model.pt`).
            profile: Which pre-built profile to load: `'normal'`, `'fast'`,
                or `'slow'`. Used to resolve index/metadata paths under
                `<root>/<index_subdir>/<profile>/`.
            root: Directory containing the `<index_subdir>/<profile>/`
                layout. Required unless `faiss_index` and `metadata` are
                given explicitly.
            faiss_index, metadata: Explicit paths that override the
                `root`+`profile` resolution. Useful for a custom layout.
            device: torch device; defaults to cuda if available.
        """
        if profile not in PROFILE_NAMES:
            raise ValueError(
                f"profile must be one of {PROFILE_NAMES}, got {profile!r}")
        self.profile = profile

        # Resolve index + metadata paths: explicit args win, else use
        # `<root>/<index_subdir>/<profile>/`.
        if faiss_index is None or metadata is None:
            if root is None:
                raise ValueError(
                    "Provide either explicit `faiss_index` + `metadata` paths, "
                    "or `root` so the per-profile paths can be resolved.")
            faiss_index = faiss_index or os.path.join(
                root, index_subdir, profile, index_filename)
            metadata = metadata or os.path.join(
                root, index_subdir, profile, metadata_filename)

        if not os.path.exists(faiss_index):
            raise FileNotFoundError(f"FAISS index not found: {faiss_index}")
        if not os.path.exists(metadata):
            raise FileNotFoundError(f"FAISS metadata not found: {metadata}")

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if verbose:
            print(f"[IclRetriever] profile={profile}  device={self.device}")
            print("[IclRetriever] loading SigLIP …")
        # use_fast=True replaces the slow Python preprocessor with the
        # tensor-only fast path (~5x faster image preprocessing).
        self.processor = AutoProcessor.from_pretrained(
            SIGLIP_MODEL, cache_dir=SIGLIP_CACHE, use_fast=True)
        # Inference is read-only; bf16 halves VRAM and ~doubles forward speed
        # vs fp32 with negligible cosine-sim drift on this 768-d encoder.
        self.vision_encoder = AutoModel.from_pretrained(SIGLIP_MODEL, cache_dir=SIGLIP_CACHE)
        self.vision_encoder.eval().to(self.device).bfloat16()
        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)

        if verbose:
            print("[IclRetriever] loading DrivingEmbeddingModel …")
        self.model = DrivingEmbeddingModel().to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        if verbose:
            print(f"[IclRetriever] loading FAISS index ({profile}): {faiss_index}")
        self.index = faiss.read_index(faiss_index)
        with open(metadata, "rb") as f:
            self.metadata: List[Dict[str, Any]] = pickle.load(f)
        if verbose:
            print(f"[IclRetriever] {self.index.ntotal} vectors, "
                  f"{len(self.metadata)} metadata entries")

    @torch.no_grad()
    def _encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pil_image, return_tensors="pt")
        # Encoder runs in bf16 (see __init__); match the input dtype to avoid
        # silent fp32->bf16 conversion in the kernel.
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)
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
    def retrieve(self,
                 query_image: Union[str, Image.Image],
                 query_state: Dict[str, Any],
                 k: int = 2,
                 exclude_self_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return top-K ICL example dicts in build_icl_prefix format.

        Args:
            query_image: PIL.Image or path to an image file.
            query_state: dict with keys speed, throttle, steer, brake, command.
                Missing keys default to 0 / "Follow Lane".
            k: number of examples to return.
            exclude_self_path: if given, drop any retrieval whose metadata
                frame_path matches (useful when re-querying training frames
                so the example doesn't trivially equal the query).
        """
        if isinstance(query_image, str):
            pil = Image.open(query_image).convert("RGB")
        else:
            pil = query_image.convert("RGB") if query_image.mode != "RGB" else query_image

        img_emb = self._encode_image(pil)

        cmd_idx = _command_to_idx(query_state.get("command"))
        ctrl = {
            "speed":       torch.tensor([float(query_state.get("speed", 0.0))],    dtype=torch.float32, device=self.device),
            "throttle":    torch.tensor([float(query_state.get("throttle", 0.0))], dtype=torch.float32, device=self.device),
            "steer":       torch.tensor([float(query_state.get("steer", 0.0))],    dtype=torch.float32, device=self.device),
            "brake":       torch.tensor([float(query_state.get("brake", 0.0))],    dtype=torch.float32, device=self.device),
            "command_idx": torch.tensor([cmd_idx],                                 dtype=torch.long,    device=self.device),
        }
        emb = self.model(img_emb, ctrl).cpu().float().numpy()

        # Search a few extras so we can drop self-matches and still satisfy k.
        n_search = k + (4 if exclude_self_path else 0)
        scores, indices = self.index.search(emb, min(n_search, self.index.ntotal))

        out: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if exclude_self_path and meta.get("frame_path") == exclude_self_path:
                continue
            ex = metadata_to_icl_example(meta)
            ex["_score"] = float(score)
            ex["_profile"] = self.profile
            out.append(ex)
            if len(out) >= k:
                break
        return out
