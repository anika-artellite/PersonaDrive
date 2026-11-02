"""
Benchmark the end-to-end ICL flow at K = 0, 1, 2, 3.

Stages timed (per K):
    1. embedding   - SigLIP image encoder + DrivingEmbeddingModel projection
                     (the 1024-d FAISS query vector). Skipped when K = 0.
    2. lookup      - IclRetriever.index.search(emb, K).
                     Skipped when K = 0.
    3. vlm_forward - DrivingModel(driving_input) -> (speed_wps, route, language).
                     For K = 0 the DrivingInput has T = 1 (just the query
                     frame), matching the agent's use_icl=False path.

Run:
    cd ${PERSONADRIVE_ROOT}
    python scripts/analysis/benchmark_icl_timing.py

Tunables via env / argparse:
    BENCH_HYDRA_CONFIG, BENCH_CKPT, PERSONADRIVE_ICL_ROOT, PERSONADRIVE_ICL_CKPT,
    PERSONADRIVE_ICL_PROFILE, --n-warmup, --n-iter, --k-list.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from simlingo_training.utils.icl_retrieval import (
    IclRetriever,
    _command_to_idx,
    _normalize_path,
)
from simlingo_training.dataloader.dataset_driving import build_icl_prefix
from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics


# Timing helpers
def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, n_warmup: int, n_iter: int) -> np.ndarray:
    for _ in range(n_warmup):
        fn()
    _sync()
    out = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        out[i] = (time.perf_counter() - t0) * 1000.0  # ms
    return out


def stats(ts: np.ndarray) -> Dict[str, float]:
    if ts.size == 0:
        return dict(mean=0.0, median=0.0, std=0.0, p95=0.0, min=0.0, max=0.0)
    return dict(
        mean=float(ts.mean()),
        median=float(np.median(ts)),
        std=float(ts.std()),
        p95=float(np.percentile(ts, 95)),
        min=float(ts.min()),
        max=float(ts.max()),
    )


# Build the VLM exactly the way agent_simlingo.LingoAgent.setup() does
def load_vlm(hydra_config: str, ckpt_path: str, device: torch.device):
    cfg = OmegaConf.load(hydra_config)
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

    cache_root = os.environ.get("PERSONADRIVE_HF_CACHE") or None   # None: Hugging Face's default cache
    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant,
        trust_remote_code=True,
        revision=hf_revision(cfg.model.vision_model.variant),
        cache_dir=cache_root,
    )
    tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>",
                "<ORG_WAYPOINTS>", "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>",
                "<TARGET_POINT>",
            ]
        }
    )
    tokenizer.padding_side = "left"

    vlm_cache_dir = f"pretrained/{cfg.model.vision_model.variant.split('/')[1]}"
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = hydra.utils.instantiate(
        cfg.model,
        cfg_data_module=cfg.data_module,
        processor=processor,
        cache_dir=vlm_cache_dir,
        _recursive_=False,
    ).to(device)
    torch.set_default_dtype(default_dtype)

    if os.path.isdir(ckpt_path):
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )
        print(f"[bench] consolidating DeepSpeed ckpt dir: {ckpt_path}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tmp_config = AutoConfig.from_pretrained(
        cfg.model.vision_model.variant,
        trust_remote_code=True,
        revision=hf_revision(cfg.model.vision_model.variant),
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


# Query frame: prefer a real, on-disk frame from FAISS metadata
def pick_query_frame(retriever: IclRetriever) -> Tuple[Image.Image, str]:
    for meta in retriever.metadata[:200]:
        p = _normalize_path(meta.get("frame_path", ""))
        if p and os.path.exists(p):
            return Image.open(p).convert("RGB"), p
    print("[bench] no real frame on-disk — falling back to synthetic 1024x512 RGB")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (512, 1024, 3), dtype=np.uint8)
    return Image.fromarray(arr), "<synthetic>"


# Build a DrivingInput at a given K. Mirrors agent_simlingo.tick().
# Returns DrivingInput plus (icl_prefix, icl_imgs) - None if K = 0.
def build_driving_input(
    K: int,
    query_pil: Image.Image,
    retrieved3: List[Dict[str, Any]],
    cfg,
    tokenizer,
    conv_module,
    num_image_token: int,
    device: torch.device,
):
    if K == 0:
        icl_prefix, icl_imgs = None, None
    else:
        icl_prefix, icl_imgs = build_icl_prefix(
            retrieved3[:K],
            num_icl_examples=K,
            cut_bottom_quarter=True,
        )
        if icl_prefix is None:
            # Real ICL images failed to load - synthesize stand-ins so timing
            # still reflects K extra frames on the T axis.
            print(f"[bench] WARNING: K={K} real ICL images missing; "
                  f"using synthetic stand-ins for timing only")
            rng = np.random.default_rng(K)
            icl_imgs = []
            for _ in range(K):
                arr = rng.integers(0, 255, (3, 768, 1024), dtype=np.uint8)
                icl_imgs.append(arr[None])
            icl_prefix = "Driving profile examples:\n\n" + (
                "<image>\nSpeed: t-2=0.0 t-1=0.0 t0=0.0 m/s. "
                "Throttle: t-2=0.00 t-1=0.00 t0=0.00. Command: Follow Lane.\n"
                "Waypoints: [0.0,0.0]\n\n"
            ) * K

    # Mirror agent's per-frame cut + dynamic_preprocess
    q_arr = np.array(query_pil)  # H, W, C
    q_arr = q_arr[: int(q_arr.shape[0] - (q_arr.shape[0] * 4.8) // 16), :, :]
    q_arr = np.transpose(q_arr, (2, 0, 1))  # C, H, W
    frame_arrays = ([im.squeeze(0) for im in icl_imgs] if icl_imgs else []) + [q_arr]
    T_actual = len(frame_arrays)

    transform = build_transform(input_size=448)
    images_processed_tmp, images_sizes_tmp = [], []
    for arr in frame_arrays:
        img_pil = Image.fromarray(arr.transpose(1, 2, 0))
        tiles = dynamic_preprocess(
            img_pil,
            image_size=448,
            use_thumbnail=cfg.model.vision_model.use_global_img,
            max_num=2,
        )
        pixel_values = torch.stack([transform(t) for t in tiles])
        images_processed_tmp.append(pixel_values)
        images_sizes_tmp.append([img_pil.size[1], img_pil.size[0]])
    stacked = torch.stack(images_processed_tmp)
    num_patches = stacked.shape[1]
    H448, W448 = stacked.shape[3], stacked.shape[4]
    camera_images = (
        stacked.view(1, T_actual, num_patches, 3, H448, W448)
        .to(device)
        .bfloat16()
    )

    # Prompt construction (mirror tick)
    prompt_tp = "Command: follow the road."
    prompt = f"Current speed: 10.0 m/s. {prompt_tp} Predict the waypoints."
    prompt = f"<SAFETY> {prompt}"
    if icl_prefix is not None:
        prompt = icl_prefix + "Current scene:\n<image>\n" + prompt

    template = conv_module.get_conv_template("internlm2-chat")
    question = prompt if "<image>" in prompt else "<image>\n" + prompt
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    system_prompt = (
        template.system_template.replace("{system_message}", template.system_message)
        + template.sep
    )
    query = query.replace(system_prompt, "")

    num_patches_all = 2  # matches agent
    image_tokens = (
        "<img>" + "<IMG_CONTEXT>" * num_image_token * num_patches_all + "</img>"
    )
    n_img = query.count("<image>")
    assert n_img == T_actual, (
        f"image-token mismatch at K={K}: prompt has {n_img} <image> tokens, "
        f"T={T_actual}"
    )
    query = query.replace("<image>", image_tokens)

    tok = tokenizer(
        [query],
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    ids = tok["input_ids"].to(device)
    valid = (tok["input_ids"] != tokenizer.pad_token_id).to(device)
    ll = LanguageLabel(
        phrase_ids=ids,
        phrase_valid=valid,
        phrase_mask=valid,
        placeholder_values=[{}],
        language_string=[query],
        loss_masking=None,
    )

    speed = torch.tensor([[10.0]], device=device, dtype=torch.float32)
    tgt = torch.tensor([[5.0, 0.0]], device=device, dtype=torch.float32)
    cam_intr = (
        get_camera_intrinsics(W448, H448, 110)
        .unsqueeze(0)
        .view(1, 3, 3)
        .float()
        .to(device)
    )
    cam_extr = (
        get_camera_extrinsics().unsqueeze(0).view(1, 4, 4).float().to(device)
    )

    di = DrivingInput(
        camera_images=camera_images,
        image_sizes=None,
        camera_intrinsics=cam_intr,
        camera_extrinsics=cam_extr,
        vehicle_speed=speed,
        target_point=tgt,
        prompt=ll,
        prompt_inference=ll,
    )
    return di, T_actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hydra-config",
        default=os.environ.get("BENCH_HYDRA_CONFIG"), help="default: <run>/.hydra/config.yaml next to --ckpt",
    )
    ap.add_argument(
        "--ckpt",
        default=os.environ.get("BENCH_CKPT") or os.environ.get("PERSONADRIVE_CKPT") or str(REPO_ROOT / "pretrained/personadrive_icl_sft/checkpoints/epoch=000.ckpt/pytorch_model.pt"),
    )
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
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--k-list", default="0,1,2,3")
    ap.add_argument(
        "--out-json",
        default=str(REPO_ROOT / "bench_icl_timing.json"),
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
    print(f"[bench] device={device}  K list={k_list}  "
          f"warmup={args.n_warmup} iter={args.n_iter}")
    print(f"[bench] hydra config: {args.hydra_config}")
    print(f"[bench] vlm ckpt    : {args.ckpt}")
    print(f"[bench] icl root    : {args.icl_root}")
    print(f"[bench] icl ckpt    : {args.icl_ckpt}")
    print(f"[bench] icl profile : {args.icl_profile}")

    cfg, model, tokenizer, conv_module, num_image_token = load_vlm(
        args.hydra_config, args.ckpt, device
    )
    retriever = IclRetriever(
        checkpoint=args.icl_ckpt,
        profile=args.icl_profile,
        root=args.icl_root,
        device=device,
        verbose=True,
    )

    query_pil, query_src = pick_query_frame(retriever)
    print(f"[bench] query frame: {query_src}  size={query_pil.size}")
    query_state = dict(speed=10.0, throttle=0.5, steer=0.0, brake=0.0, command=4)

    # One retrieval at K=3 - slice for K=1,2 so all three sizes share the
    # same neighbour list (only the prefix length / T-axis differs).
    max_k = max(k for k in k_list if k > 0) if any(k > 0 for k in k_list) else 0
    retrieved = (
        retriever.retrieve(
            query_image=query_pil, query_state=query_state, k=max_k
        )
        if max_k > 0
        else []
    )
    print(f"[bench] retrieved {len(retrieved)} neighbours for K=1..{max_k}")

    results: Dict[int, Dict[str, Any]] = {}
    for K in k_list:
        print(f"\n[bench] ── K={K} ─────────────────────────────────────")
        di, T_actual = build_driving_input(
            K, query_pil, retrieved, cfg, tokenizer, conv_module,
            num_image_token, device,
        )
        print(f"[bench]   T_actual={T_actual}  "
              f"camera_images={tuple(di.camera_images.shape)}")

        @torch.inference_mode()
        def run_vlm():
            return model(di)

        vlm_ts = time_fn(run_vlm, args.n_warmup, args.n_iter)

        if K == 0:
            emb_ts = np.zeros(args.n_iter)
            lookup_ts = np.zeros(args.n_iter)
        else:
            cmd_idx = _command_to_idx(query_state.get("command"))
            ctrl = {
                "speed": torch.tensor(
                    [float(query_state["speed"])], device=device
                ),
                "throttle": torch.tensor(
                    [float(query_state["throttle"])], device=device
                ),
                "steer": torch.tensor(
                    [float(query_state["steer"])], device=device
                ),
                "brake": torch.tensor(
                    [float(query_state["brake"])], device=device
                ),
                "command_idx": torch.tensor(
                    [cmd_idx], dtype=torch.long, device=device
                ),
            }

            @torch.inference_mode()
            def run_emb():
                img_emb = retriever._encode_image(query_pil)
                _ = retriever.model(img_emb, ctrl).cpu().float().numpy()

            # Cache one embedding for the lookup-only timer
            with torch.inference_mode():
                img_emb_once = retriever._encode_image(query_pil)
                emb_once = (
                    retriever.model(img_emb_once, ctrl).cpu().float().numpy()
                )

            def run_lookup():
                retriever.index.search(emb_once, K)

            emb_ts = time_fn(run_emb, args.n_warmup, args.n_iter)
            lookup_ts = time_fn(run_lookup, args.n_warmup, args.n_iter)

        results[K] = {
            "T_actual": T_actual,
            "embedding": stats(emb_ts),
            "lookup": stats(lookup_ts),
            "vlm_forward": stats(vlm_ts),
            "total_mean_ms": float(
                emb_ts.mean() + lookup_ts.mean() + vlm_ts.mean()
            ),
        }
        r = results[K]
        print(
            f"[bench]   embed   = {r['embedding']['mean']:8.2f} ms "
            f"(p95 {r['embedding']['p95']:.2f})"
        )
        print(
            f"[bench]   lookup  = {r['lookup']['mean']:8.3f} ms "
            f"(p95 {r['lookup']['p95']:.3f})"
        )
        print(
            f"[bench]   vlm fwd = {r['vlm_forward']['mean']:8.2f} ms "
            f"(p95 {r['vlm_forward']['p95']:.2f})"
        )
        print(f"[bench]   total  = {r['total_mean_ms']:8.2f} ms")

    print("\n" + "=" * 78)
    print(
        f"{'K':>3} | {'T':>2} | {'embed (ms)':>12} | "
        f"{'lookup (ms)':>12} | {'vlm (ms)':>10} | {'total (ms)':>12}"
    )
    print("-" * 78)
    for K in k_list:
        r = results[K]
        print(
            f"{K:>3} | {r['T_actual']:>2} | "
            f"{r['embedding']['mean']:>12.2f} | "
            f"{r['lookup']['mean']:>12.3f} | "
            f"{r['vlm_forward']['mean']:>10.2f} | "
            f"{r['total_mean_ms']:>12.2f}"
        )
    print("=" * 78)

    with open(args.out_json, "w") as f:
        json.dump(
            {
                "device": str(device),
                "n_warmup": args.n_warmup,
                "n_iter": args.n_iter,
                "hydra_config": args.hydra_config,
                "ckpt": args.ckpt,
                "icl_root": args.icl_root,
                "icl_profile": args.icl_profile,
                "results": {str(k): v for k, v in results.items()},
            },
            f,
            indent=2,
        )
    print(f"[bench] wrote {args.out_json}")


if __name__ == "__main__":
    main()
