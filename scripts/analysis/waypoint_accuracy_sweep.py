"""
Open-loop waypoint prediction accuracy sweep at K = 0..4 in-context examples.

Same caveat as scripts/analysis/qa_accuracy_sweep.py: validation_* routes aren't on this
node, so samples come from the *training* set. K=2 was the trained-for
context length, so absolute numbers at K=2 will be optimistic; the
*relative* shape across K is the useful signal.

Per-sample:
    1. Pick a frame `t` from a route's measurements.json that has at
       least `pred_horizon` future frames available.
    2. GT waypoints = ego_frame_t(measurements[t+1 : t+H+1]['ego_matrix'][:3, 3]).
       Mirror get_waypoints() in dataset_base.py - origin is the ego pose
       at t, future positions are rotated/translated into that frame.
    3. Build the driving prompt:
           "Current speed: {s} m/s. Command: {...}. Predict the waypoints."
       Prepend ICL prefix for K > 0 (built from icl_examples.json.gz[t]).
    4. Run DrivingModel.forward(...) -> speed_wps, route, _.
    5. ADE (mean L2 over horizon) and FDE (L2 at final step) per sample.

Outputs:
    waypoint_sweep_results.json
    Console: per-K ADE/FDE in metres + latency.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoConfig, AutoProcessor

import hydra
from hydra.utils import to_absolute_path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
from simlingo_training.utils.hf_revision import hf_revision
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
from simlingo_training.utils.icl_retrieval import IclRetriever
from simlingo_training.dataloader.dataset_driving import build_icl_prefix
from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics


# Token-level timing hooks
#
# Wraps the language_model so we can split a `model(di)` call into:
#   prefill   - adaptors + image-feature injection (before greedy_sample)
#   first_tok - first LLM forward inside greedy_sample
#   decode    - subsequent LLM forwards inside greedy_sample (autoregressive)
#   traj_head - adaptors.driving + LLM forward + adaptors.driving.get_predictions
#               (after greedy_sample, the "project to other NN" stage)
#
# Reported metrics:
#   TTFT  = prefill + first_tok    (input ready -> first language token out)
#   TTLT  = decode                 (autoregressive token dump, no projection)
#   total = TTFT + TTLT + traj_head (end-to-end to actual trajectory waypoints)
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _TimingHooks:
    def __init__(self, model):
        self.model = model
        self._patched = False
        self.reset()
        self._patch()

    def reset(self):
        # populated on every model(di) call
        self.gs_start = None
        self.gs_end = None
        self.lm_fwd_durations_in_gs: List[float] = []
        self._in_gs = False

    def _patch(self):
        if self._patched:
            return
        lm = self.model.language_model
        orig_gs = lm.greedy_sample
        orig_lm_fwd = lm.forward
        # Patched greedy_sample bypasses lm.forward and calls lm.model
        # directly. Wrap both so we always capture the per-iteration cost.
        orig_inner_fwd = lm.model.forward

        def timed_lm_fwd(*a, **k):
            _sync()
            t0 = time.perf_counter()
            r = orig_lm_fwd(*a, **k)
            _sync()
            dt = time.perf_counter() - t0
            if self._in_gs:
                self.lm_fwd_durations_in_gs.append(dt)
            return r

        def timed_inner_fwd(*a, **k):
            _sync()
            t0 = time.perf_counter()
            r = orig_inner_fwd(*a, **k)
            _sync()
            dt = time.perf_counter() - t0
            if self._in_gs:
                self.lm_fwd_durations_in_gs.append(dt)
            return r

        def timed_gs(*a, **k):
            self.lm_fwd_durations_in_gs = []
            self._in_gs = True
            _sync()
            self.gs_start = time.perf_counter()
            try:
                r = orig_gs(*a, **k)
            finally:
                _sync()
                self.gs_end = time.perf_counter()
                self._in_gs = False
            return r

        lm.greedy_sample = timed_gs
        lm.forward = timed_lm_fwd
        lm.model.forward = timed_inner_fwd
        self._patched = True

    def collect(self, total_start: float, total_end: float) -> Dict[str, float]:
        """Compose per-call breakdown from the captured timestamps.

        Returns ms-valued dict suitable for JSON dump.
        """
        total = (total_end - total_start) * 1000.0
        if self.gs_start is None or self.gs_end is None or not self.lm_fwd_durations_in_gs:
            # predict_language path didn't fire - single-shot forward only
            return {
                "prefill_ms": 0.0,
                "first_tok_ms": 0.0,
                "ttft_ms": 0.0,
                "ttlt_ms": 0.0,
                "traj_head_ms": total,
                "total_ms": total,
                "n_tokens": 0,
            }
        prefill_s = self.gs_start - total_start
        first_tok_s = self.lm_fwd_durations_in_gs[0]
        decode_s = sum(self.lm_fwd_durations_in_gs[1:])
        # The greedy_sample wall time includes the LLM forwards plus a small
        # Python-side overhead (sample + cat + mask update). Attribute that to
        # the decoding bucket so TTFT + TTLT add up to the gs wall time.
        gs_wall_s = self.gs_end - self.gs_start
        decode_wall_s = max(gs_wall_s - first_tok_s, decode_s)
        traj_head_s = total_end - self.gs_end
        return {
            "prefill_ms":   prefill_s * 1000.0,
            "first_tok_ms": first_tok_s * 1000.0,
            "ttft_ms":      (prefill_s + first_tok_s) * 1000.0,
            "ttlt_ms":      decode_wall_s * 1000.0,
            "traj_head_ms": traj_head_s * 1000.0,
            "total_ms":     total,
            "n_tokens":     len(self.lm_fwd_durations_in_gs),
        }


COMMAND_TEXT = {
    1: "go left at the next intersection",
    2: "go right at the next intersection",
    3: "go straight at the next intersection",
    4: "follow the road",
    5: "do a lane change to the left",
    6: "do a lane change to the right",
}


# GT waypoint extraction - mirror dataset_base.get_waypoints()
def ego_frame_waypoints(measurements: List[Dict[str, Any]],
                        t: int,
                        horizon: int) -> np.ndarray:
    """Return [horizon, 2] future ego-frame (x, y) waypoints starting at t+1.

    Origin is measurements[t]'s ego pose; subsequent ego positions are
    rotated/translated into that frame and projected to BEV (drop z).
    """
    origin = np.asarray(measurements[t]["ego_matrix"])[:3]
    origin_t = origin[:, 3:4]
    origin_R = origin[:, :3]
    wps = []
    for i in range(1, horizon + 1):
        idx = min(t + i, len(measurements) - 1)
        pos = np.asarray(measurements[idx]["ego_matrix"])[:3, 3:4]
        ego = origin_R.T @ (pos - origin_t)
        wps.append(ego[:2, 0])
    return np.asarray(wps, dtype=np.float32)


# Sample discovery
def discover_samples(database_root: str, n: int, horizon: int,
                     seed: int, skip_first_n: int = 10) -> List[Dict[str, Any]]:
    data_root = Path(database_root) / "data" / "simlingo"
    rng = random.Random(seed)
    # sorted: rglob order depends on the filesystem, which made the drawn
    # frames differ between machines for the same seed
    routes = sorted(data_root.rglob("measurements.json"))
    rng.shuffle(routes)
    out: List[Dict[str, Any]] = []
    for meas_path in routes:
        route_dir = meas_path.parent
        icl_path = route_dir / "icl_examples.json.gz"
        rgb_dir = route_dir / "rgb"
        if not (icl_path.exists() and rgb_dir.exists()):
            continue
        try:
            with open(meas_path) as f:
                measurements = json.load(f)
            with gzip.open(icl_path, "rt") as f:
                icl_agg = json.load(f)
        except Exception:
            continue
        icl_frames = set(int(k) for k in icl_agg.keys())
        n_meas = len(measurements)

        # Need at least horizon future frames, ICL available, RGB on disk.
        valid_frames = [
            t for t in range(skip_first_n, n_meas - horizon - 1)
            if t in icl_frames and (rgb_dir / f"{t:04d}.jpg").exists()
        ]
        rng.shuffle(valid_frames)
        for t in valid_frames[:3]:
            out.append({
                "route_dir": str(route_dir),
                "meas_path": str(meas_path),
                "icl_path": str(icl_path),
                "rgb_path": str(rgb_dir / f"{t:04d}.jpg"),
                "frame_idx": t,
            })
            if len(out) >= n:
                return out
    return out


def load_samples(samples_json: str, database_root: str) -> List[Dict[str, Any]]:
    """Frames of an earlier sweep: its out-json, or a list of
    {"route": <path relative to <database>/data/simlingo>, "frame_idx": t}."""
    with open(samples_json) as f:
        doc = json.load(f)
    if isinstance(doc, dict):  # an earlier sweep's out-json
        first = doc["predictions"][sorted(doc["predictions"], key=int)[0]]
        doc = [{"route": x["rgb_path"].split("/data/simlingo/", 1)[1].rsplit("/rgb/", 1)[0],
                "frame_idx": x["frame_idx"]} for x in first]
    data_root = Path(database_root) / "data" / "simlingo"
    out: List[Dict[str, Any]] = []
    for s in doc:
        route_dir = data_root / s["route"]
        t = int(s["frame_idx"])
        out.append({
            "route_dir": str(route_dir),
            "meas_path": str(route_dir / "measurements.json"),
            "icl_path": str(route_dir / "icl_examples.json.gz"),
            "rgb_path": str(route_dir / "rgb" / f"{t:04d}.jpg"),
            "frame_idx": t,
        })
    missing = [s["meas_path"] for s in out if not os.path.exists(s["meas_path"])]
    if missing:
        raise SystemExit(f"{len(missing)} sample routes not found under {data_root}, e.g. {missing[0]}")
    return out


# VLM loader - same as scripts/analysis/benchmark_icl_timing.py
def load_vlm(hydra_config: str, ckpt_path: str, device: torch.device):
    cfg = OmegaConf.load(hydra_config)
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
    cache_root = os.environ.get("PERSONADRIVE_HF_CACHE") or None   # None: Hugging Face's default cache
    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant, trust_remote_code=True, revision=hf_revision(cfg.model.vision_model.variant),
        cache_dir=cache_root,
    )
    tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>",
        "<ORG_WAYPOINTS>", "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>",
        "<TARGET_POINT>",
    ]})
    tokenizer.padding_side = "left"

    vlm_cache_dir = f"pretrained/{cfg.model.vision_model.variant.split('/')[1]}"
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = hydra.utils.instantiate(
        cfg.model, cfg_data_module=cfg.data_module, processor=processor,
        cache_dir=vlm_cache_dir, _recursive_=False,
    ).to(device)
    torch.set_default_dtype(default_dtype)

    if os.path.isdir(ckpt_path):
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )
        print(f"[wp-sweep] consolidating DeepSpeed ckpt dir: {ckpt_path}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tmp_config = AutoConfig.from_pretrained(
        cfg.model.vision_model.variant, trust_remote_code=True, revision=hf_revision(cfg.model.vision_model.variant),
        cache_dir=cache_root,
    )
    image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
    patch_size = tmp_config.vision_config.patch_size
    num_image_token = int(
        (image_size // patch_size) ** 2 * (tmp_config.downsample_ratio ** 2)
    )

    conv_path = to_absolute_path(f"{vlm_cache_dir}/conversation.py")
    if not os.path.exists(conv_path):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=cfg.model.vision_model.variant,
                          local_dir=vlm_cache_dir)
    spec = importlib.util.spec_from_file_location("get_conv_template", conv_path)
    conv_module = importlib.util.module_from_spec(spec)
    sys.modules["get_conv_template"] = conv_module
    spec.loader.exec_module(conv_module)

    return cfg, model, tokenizer, conv_module, num_image_token


# Per-sample input construction
def build_inputs(sample: Dict[str, Any], K: int,
                 cfg, tokenizer, conv_module, num_image_token: int,
                 device: torch.device,
                 retriever: "IclRetriever | None" = None,
                 use_cot: bool = False):
    """Build a DrivingInput for one sample.

    Retrieval source:
      - retriever != None: live FAISS top-K via IclRetriever (any K).
      - retriever is None: fallback to per-route icl_examples.json.gz
        (capped at the file's stored neighbor count - typically 2).
    """
    with open(sample["meas_path"]) as f:
        measurements = json.load(f)

    t = sample["frame_idx"]
    meas = measurements[t]
    speed = round(float(meas.get("speed", 0.0)), 1)
    cmd = int(meas.get("command", 4))
    next_cmd = int(meas.get("next_command", 4))
    target_point = meas.get("target_point", [0.0, 0.0])
    dist = int(np.linalg.norm(np.asarray(target_point, dtype=np.float32)))

    cmd_text = COMMAND_TEXT.get(cmd, "follow the road")
    next_text = COMMAND_TEXT.get(next_cmd, cmd_text)
    next_suffix = f" then {next_text}" if next_text != cmd_text else ""
    if cmd == 4:
        cmd_str = f"Command: {cmd_text}{next_suffix}."
    else:
        cmd_str = f"Command: {cmd_text} in {dist} meter{next_suffix}."

    # Match agent_simlingo.py: use_cot=True asks for commentary before the
    # waypoint head fires; this is the GlobalConfig default. The waypoint-only
    # suffix is what the agent uses when use_cot=False.
    if use_cot:
        prompt = f"Current speed: {speed} m/s. {cmd_str} What should the ego do next?"
    else:
        prompt = f"Current speed: {speed} m/s. {cmd_str} Predict the waypoints."

    if K == 0:
        icl_prefix, icl_imgs = None, None
    elif retriever is not None:
        # Live FAISS retrieval - matches agent_simlingo.py's online path.
        query_state = {
            "speed":    speed,
            "throttle": float(meas.get("throttle", 0.0)),
            "steer":    float(meas.get("steer", 0.0)),
            "brake":    float(meas.get("brake", 0.0)),
            "command":  cmd,
        }
        examples = retriever.retrieve(
            query_image=sample["rgb_path"],
            query_state=query_state,
            k=K,
            exclude_self_path=sample["rgb_path"],
        )
        if len(examples) < K:
            # Not enough on-disk frames to fill K - skip this sample.
            raise RuntimeError(
                f"only {len(examples)}/{K} live retrievals usable"
            )
        icl_prefix, icl_imgs = build_icl_prefix(
            examples, num_icl_examples=K, cut_bottom_quarter=True,
        )
    else:
        with gzip.open(sample["icl_path"], "rt") as f:
            icl_agg = json.load(f)
        icl_examples = icl_agg.get(str(t), [])
        if not icl_examples:
            icl_prefix, icl_imgs = None, None
        else:
            K_eff = min(K, len(icl_examples))
            icl_prefix, icl_imgs = build_icl_prefix(
                icl_examples[:K_eff], num_icl_examples=K_eff,
                cut_bottom_quarter=True,
            )

    q_img = Image.open(sample["rgb_path"]).convert("RGB")
    q_arr = np.array(q_img)
    q_arr = q_arr[: int(q_arr.shape[0] - (q_arr.shape[0] * 4.8) // 16), :, :]
    q_arr = np.transpose(q_arr, (2, 0, 1))
    frame_arrays = ([im.squeeze(0) for im in icl_imgs] if icl_imgs else []) + [q_arr]
    T_actual = len(frame_arrays)

    if icl_prefix is not None:
        prompt = icl_prefix + "Current scene:\n<image>\n" + prompt

    transform = build_transform(input_size=448)
    images_processed_tmp = []
    for arr in frame_arrays:
        img_pil = Image.fromarray(arr.transpose(1, 2, 0))
        tiles = dynamic_preprocess(
            img_pil, image_size=448,
            use_thumbnail=cfg.model.vision_model.use_global_img, max_num=2,
        )
        pv = torch.stack([transform(t_) for t_ in tiles])
        images_processed_tmp.append(pv)
    stacked = torch.stack(images_processed_tmp)
    num_patches = stacked.shape[1]
    H448, W448 = stacked.shape[3], stacked.shape[4]
    camera_images = (
        stacked.view(1, T_actual, num_patches, 3, H448, W448)
        .to(device).bfloat16()
    )

    template = conv_module.get_conv_template("internlm2-chat")
    question = prompt if "<image>" in prompt else "<image>\n" + prompt
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    sys_p = (
        template.system_template.replace("{system_message}", template.system_message)
        + template.sep
    )
    query = query.replace(sys_p, "")

    image_tokens = (
        "<img>" + "<IMG_CONTEXT>" * num_image_token * 2 + "</img>"
    )
    n_img = query.count("<image>")
    if n_img != T_actual:
        raise RuntimeError(
            f"image-token mismatch K={K}: prompt has {n_img}, T={T_actual}"
        )
    query = query.replace("<image>", image_tokens)

    tok = tokenizer([query], padding=True, return_tensors="pt",
                    return_offsets_mapping=True, add_special_tokens=False)
    ids = tok["input_ids"].to(device)
    valid = (tok["input_ids"] != tokenizer.pad_token_id).to(device)
    ll = LanguageLabel(
        phrase_ids=ids, phrase_valid=valid, phrase_mask=valid,
        placeholder_values=[{}], language_string=[query], loss_masking=None,
    )

    speed_t = torch.tensor([[speed]], device=device, dtype=torch.float32)
    tgt = torch.tensor(
        [[float(target_point[0]), float(target_point[1])]],
        device=device, dtype=torch.float32,
    )
    cam_intr = (
        get_camera_intrinsics(W448, H448, 110)
        .unsqueeze(0).view(1, 3, 3).float().to(device)
    )
    cam_extr = (
        get_camera_extrinsics().unsqueeze(0).view(1, 4, 4).float().to(device)
    )

    di = DrivingInput(
        camera_images=camera_images, image_sizes=None,
        camera_intrinsics=cam_intr, camera_extrinsics=cam_extr,
        vehicle_speed=speed_t, target_point=tgt,
        prompt=ll, prompt_inference=ll,
    )
    prompt_n_tokens = int(ids.shape[1])
    return di, measurements, prompt_n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hydra-config", default=None, help="default: <run>/.hydra/config.yaml next to --ckpt",
    )
    ap.add_argument(
        "--ckpt",
        default=os.environ.get("PERSONADRIVE_CKPT") or str(REPO_ROOT / "pretrained/personadrive_icl_sft/checkpoints/epoch=000.ckpt/pytorch_model.pt"),
    )
    ap.add_argument(
        "--database",
        default=str(REPO_ROOT / "database/simlingo"),
    )
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--samples-json", default=None,
                    help="evaluate exactly these frames instead of drawing --n-samples "
                         "(an earlier --out-json, or scripts/analysis/openloop_samples_v1.json)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-list", default="0,1,2,3,4")
    ap.add_argument("--horizon", type=int, default=11,
                    help="future waypoint horizon (matches cfg pred_len)")
    ap.add_argument(
        "--out-json",
        default=str(REPO_ROOT / "waypoint_sweep_results.json"),
    )
    ap.add_argument("--live-icl", action="store_true", default=True,
                    help="Use live FAISS retrieval (required for K>2; the "
                         "per-route icl_examples.json.gz only stores 2 neighbours).")
    ap.add_argument("--no-live-icl", dest="live_icl", action="store_false")
    ap.add_argument("--use-cot", action="store_true", default=False,
                    help="Use the CoT/commentary prompt suffix "
                         "(\"What should the ego do next?\") instead of "
                         "the waypoint-only suffix (\"Predict the waypoints.\"). "
                         "Matches the agent's GlobalConfig default (use_cot=True).")
    ap.add_argument(
        "--icl-root",
        default=os.environ.get(
            "PERSONADRIVE_ICL_ROOT",
            str(REPO_ROOT / "persona_drive_rag"),
        ),
    )
    ap.add_argument("--icl-ckpt", default=os.environ.get("PERSONADRIVE_ICL_CKPT"))
    ap.add_argument(
        "--icl-profile",
        default=os.environ.get("PERSONADRIVE_ICL_PROFILE", "normal"),
    )
    args = ap.parse_args()
    if args.hydra_config is None:
        # <run>/.hydra/config.yaml above the checkpoint: <run>/checkpoints/last.ckpt (DeepSpeed dir)
        # or <run>/checkpoints/epoch=000.ckpt/pytorch_model.pt (single file, as on Hugging Face)
        runs = [p for p in Path(args.ckpt).absolute().parents if (p / ".hydra" / "config.yaml").is_file()]
        args.hydra_config = str((runs[0] if runs else Path(args.ckpt).parents[1]) / ".hydra" / "config.yaml")
    if args.icl_ckpt is None:
        args.icl_ckpt = os.path.join(args.icl_root, "checkpoints/best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_list = [int(x) for x in args.k_list.split(",")]
    print(f"[wp-sweep] device={device}  K list={k_list}  "
          f"N={args.n_samples}  horizon={args.horizon}  seed={args.seed}")

    if args.samples_json:
        samples = load_samples(args.samples_json, args.database)
    else:
        samples = discover_samples(args.database, args.n_samples, args.horizon, args.seed)
    print(f"[wp-sweep] discovered {len(samples)} runnable samples")
    if not samples:
        print("[wp-sweep] no samples found — aborting")
        return

    cfg, model, tokenizer, conv_module, num_image_token = load_vlm(
        args.hydra_config, args.ckpt, device,
    )

    retriever = None
    if args.live_icl:
        print(f"[wp-sweep] live ICL via FAISS  root={args.icl_root}  "
              f"profile={args.icl_profile}")
        retriever = IclRetriever(
            checkpoint=args.icl_ckpt, profile=args.icl_profile,
            root=args.icl_root, device=device, verbose=True,
        )

    timing_hooks = _TimingHooks(model)

    results: Dict[int, Dict[str, Any]] = {}
    per_sample: Dict[int, List[Dict[str, Any]]] = {}
    pred_h = None  # model's output horizon - discovered on first forward

    for K in k_list:
        print(f"\n[wp-sweep] ── K={K} over {len(samples)} samples ──────────────")
        ade_list, fde_list = [], []
        prompt_toks_list: List[int] = []
        gen_toks_list: List[int] = []
        ttft_list: List[float] = []
        ttlt_list: List[float] = []
        total_list: List[float] = []
        traj_head_list: List[float] = []
        per_step_err: List[np.ndarray] = []
        rows = []
        for idx, sample in enumerate(samples):
            try:
                di, measurements, prompt_n_tokens = build_inputs(
                    sample, K, cfg, tokenizer, conv_module,
                    num_image_token, device, retriever=retriever,
                    use_cot=args.use_cot,
                )
            except Exception as e:
                print(f"  [skip {idx}] build failed: "
                      f"{type(e).__name__}: {e}")
                continue
            timing_hooks.reset()
            _sync()
            t_call_start = time.perf_counter()
            try:
                with torch.inference_mode():
                    speed_wps, route, language = model(di)
            except Exception as e:
                print(f"  [skip {idx}] forward failed: "
                      f"{type(e).__name__}: {e}")
                continue
            _sync()
            t_call_end = time.perf_counter()
            timing = timing_hooks.collect(t_call_start, t_call_end)
            lang_text = language[0] if language else ""

            pred = speed_wps.float().cpu().numpy()[0]  # [H_pred, 2]
            H_pred = pred.shape[0]
            if pred_h is None:
                pred_h = H_pred
                print(f"[wp-sweep]   model output horizon = {H_pred}")

            gt = ego_frame_waypoints(
                measurements, sample["frame_idx"], H_pred
            )
            err = np.linalg.norm(pred - gt, axis=1)
            ade = float(err.mean())
            fde = float(err[-1])
            ade_list.append(ade)
            fde_list.append(fde)
            per_step_err.append(err)
            prompt_toks_list.append(prompt_n_tokens)
            gen_toks_list.append(timing["n_tokens"])
            ttft_list.append(timing["ttft_ms"])
            ttlt_list.append(timing["ttlt_ms"])
            total_list.append(timing["total_ms"])
            traj_head_list.append(timing["traj_head_ms"])
            rows.append({
                "rgb_path": sample["rgb_path"],
                "frame_idx": sample["frame_idx"],
                "ADE_m": ade,
                "FDE_m": fde,
                "per_step_err_m": err.tolist(),
                "prompt_tokens": prompt_n_tokens,
                "n_generated_tokens": timing["n_tokens"],
                "language": lang_text,
                "ttft_ms": timing["ttft_ms"],
                "ttlt_ms": timing["ttlt_ms"],
                "traj_head_ms": timing["traj_head_ms"],
                "total_ms": timing["total_ms"],
            })
            if (idx + 1) % 25 == 0:
                print(f"  [{idx+1}/{len(samples)}] "
                      f"ADE={np.mean(ade_list):.3f} m  FDE="
                      f"{np.mean(fde_list):.3f} m  "
                      f"prompt={int(np.mean(prompt_toks_list))} tok  "
                      f"TTFT={np.mean(ttft_list):.0f}  "
                      f"TTLT={np.mean(ttlt_list):.0f}  "
                      f"total={np.mean(total_list):.0f} ms")

        if not ade_list:
            print(f"[wp-sweep] K={K}: no successful samples")
            continue
        ade_arr = np.asarray(ade_list)
        fde_arr = np.asarray(fde_list)
        per_step = np.stack(per_step_err)
        results[K] = {
            "n": len(ade_list),
            "ADE_mean_m":   float(ade_arr.mean()),
            "ADE_median_m": float(np.median(ade_arr)),
            "ADE_p95_m":    float(np.percentile(ade_arr, 95)),
            "FDE_mean_m":   float(fde_arr.mean()),
            "FDE_median_m": float(np.median(fde_arr)),
            "FDE_p95_m":    float(np.percentile(fde_arr, 95)),
            "per_step_mean_m": per_step.mean(axis=0).tolist(),
            "prompt_tokens_mean":  float(np.mean(prompt_toks_list)),
            "prompt_tokens_max":   int(np.max(prompt_toks_list)),
            "n_generated_tokens_mean": float(np.mean(gen_toks_list)),
            "ttft_mean_ms":  float(np.mean(ttft_list)),
            "ttft_p95_ms":   float(np.percentile(ttft_list, 95)),
            "ttlt_mean_ms":  float(np.mean(ttlt_list)),
            "ttlt_p95_ms":   float(np.percentile(ttlt_list, 95)),
            "traj_head_mean_ms": float(np.mean(traj_head_list)),
            "total_mean_ms": float(np.mean(total_list)),
            "total_p95_ms":  float(np.percentile(total_list, 95)),
        }
        per_sample[K] = rows
        r = results[K]
        print(f"[wp-sweep] K={K}  ADE={r['ADE_mean_m']:.3f} m  "
              f"FDE={r['FDE_mean_m']:.3f} m  n={r['n']}  "
              f"prompt={int(r['prompt_tokens_mean'])} tok  "
              f"TTFT={r['ttft_mean_ms']:.0f}  "
              f"TTLT={r['ttlt_mean_ms']:.0f}  "
              f"total={r['total_mean_ms']:.0f} ms")

    print("\n" + "=" * 110)
    print(f"{'K':>3} | {'n':>3} | {'prompt':>6} | {'gen':>3} | "
          f"{'ADE':>6} | {'FDE':>6} | "
          f"{'TTFT':>6} | {'TTLT':>6} | {'traj':>6} | {'TOTAL':>6}  (ms)")
    print("-" * 110)
    for K in k_list:
        if K not in results:
            continue
        r = results[K]
        print(f"{K:>3} | {r['n']:>3} | "
              f"{int(r['prompt_tokens_mean']):>6} | "
              f"{int(r['n_generated_tokens_mean']):>3} | "
              f"{r['ADE_mean_m']:>6.3f} | {r['FDE_mean_m']:>6.3f} | "
              f"{r['ttft_mean_ms']:>6.1f} | {r['ttlt_mean_ms']:>6.1f} | "
              f"{r['traj_head_mean_ms']:>6.1f} | {r['total_mean_ms']:>6.1f}")
    print("=" * 110)
    print("\nPer-timestep ADE (metres), one row per K:")
    for K in k_list:
        if K not in results:
            continue
        row = results[K]["per_step_mean_m"]
        cells = "  ".join(f"{v:5.2f}" for v in row)
        print(f"  K={K}: {cells}")

    with open(args.out_json, "w") as f:
        json.dump({
            "args": vars(args),
            "results": {str(k): v for k, v in results.items()},
            "predictions": {str(k): v for k, v in per_sample.items()},
        }, f, indent=2)
    print(f"\n[wp-sweep] wrote {args.out_json}")


if __name__ == "__main__":
    main()
