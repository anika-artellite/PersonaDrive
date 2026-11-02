"""
Open-loop QA accuracy sweep at K = 0, 1, 2, 3, 4 in-context examples.

NOTE on caveat: the standard validation_* routes are not present on this
node, so this script samples Q/A pairs from the *training* set (the only
routes whose vqa.json / icl_examples.json.gz / rgb frames are mounted
here). K=2 was the trained-for context length, so absolute accuracy at
K=2 will be optimistic; the *relative* shape across K is the useful
signal. To run a true held-out eval, transfer the validation_* routes
and rerun against `evalset_vqa.json`.

Per-sample we mirror dataset_eval_qa_comm.__getitem__ in spirit:
    1. Pick one Q/A pair from the route's vqa.json[frame_idx].
    2. Build prompt: "Current speed: {s} m/s. Command: <command...>. Q: {Q}"
    3. For K > 0, prepend the ICL prefix built from icl_examples.json.gz[frame_idx].
    4. Tokenize, run DrivingModel.forward(return_language=True), decode.
    5. Exact-match against "A: {GT_answer}".

Outputs:
    qa_sweep_results.json   per-K accuracy + per-sample preds
    Console table           accuracy + mean latency per K
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
from typing import Any, Dict, List, Optional, Tuple

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
from simlingo_training.dataloader.dataset_driving import build_icl_prefix
from team_code.simlingo_utils import get_camera_extrinsics, get_camera_intrinsics


COMMAND_TEXT = {
    1: "go left at the next intersection",
    2: "go right at the next intersection",
    3: "go straight at the next intersection",
    4: "follow the road",
    5: "do a lane change to the left",
    6: "do a lane change to the right",
}


# Sample discovery
def discover_qa_samples(database_root: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Walk drivelm/ for vqa.json files; sample n (route_dir, frame_idx, Q, A) entries.

    Requires matching measurements.json, icl_examples.json.gz, and rgb/<frame>.jpg
    so every selected sample is fully runnable.
    """
    drivelm_root = Path(database_root) / "drivelm" / "simlingo"
    candidates: List[Tuple[Path, int, str, str]] = []
    rng = random.Random(seed)

    routes = list(drivelm_root.rglob("vqa.json"))
    rng.shuffle(routes)
    for vqa_path in routes:
        drivelm_route = vqa_path.parent
        data_route = Path(str(drivelm_route).replace("/drivelm/", "/data/"))
        meas_path = data_route / "measurements.json"
        icl_path = data_route / "icl_examples.json.gz"
        rgb_dir = data_route / "rgb"
        if not (meas_path.exists() and icl_path.exists() and rgb_dir.exists()):
            continue
        try:
            with open(vqa_path) as f:
                vqa = json.load(f)
        except Exception:
            continue

        # Look up which frames have an ICL entry - only those can be used at K>0
        try:
            with gzip.open(icl_path, "rt") as f:
                icl_agg = json.load(f)
        except Exception:
            continue
        icl_frames = set(int(k) for k in icl_agg.keys())

        # Sample a handful of frames per route to keep diversity
        frame_indices = list(range(min(len(vqa), 141)))
        rng.shuffle(frame_indices)
        for fi in frame_indices[:3]:
            if fi not in icl_frames:
                continue
            rgb_file = rgb_dir / f"{fi:04d}.jpg"
            if not rgb_file.exists():
                continue
            qa = vqa[fi]
            qa_dict = qa.get("QA", {})
            qa_list: List[Dict[str, str]] = []
            # Open-ended descriptive Qs aren't useful for exact-match; the
            # model paraphrases ground-truth and always misses. Skip them so
            # the K-sweep signal isn't drowned in noise.
            OPEN_ENDED = (
                "What are the important objects",
                "Which lanes are important",
                "What should the ego vehicle do",
                "Why",
            )
            for cat in ("perception", "prediction", "planning", "behavior"):
                for item in qa_dict.get(cat, []):
                    q = (item.get("Q") or "").strip()
                    a = (item.get("A") or "").strip()
                    if not q or not a:
                        continue
                    if "<OBJECT>" in q or "<LOCATION>" in q or "<DISTANCE>" in q:
                        continue
                    if any(o in q for o in OPEN_ENDED):
                        continue
                    # Prefer short, constrained answers (yes/no/number/category)
                    if len(a) > 120:
                        continue
                    qa_list.append({"Q": q, "A": a, "cat": cat})
            if not qa_list:
                continue
            qa_pick = qa_list[0]  # deterministic
            candidates.append({
                "data_route": str(data_route),
                "rgb_path": str(rgb_file),
                "frame_idx": fi,
                "meas_path": str(meas_path),
                "icl_path": str(icl_path),
                "Q": qa_pick["Q"],
                "A": qa_pick["A"],
                "cat": qa_pick["cat"],
            })
            if len(candidates) >= n:
                return candidates
    return candidates


# Build inputs for one (sample, K) and run the model
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
        print(f"[qa-sweep] consolidating DeepSpeed ckpt dir: {ckpt_path}")
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


def load_route_assets(sample: Dict[str, Any]):
    with open(sample["meas_path"]) as f:
        measurements = json.load(f)
    with gzip.open(sample["icl_path"], "rt") as f:
        icl_agg = json.load(f)
    meas = measurements[sample["frame_idx"]]
    icl_examples = icl_agg.get(str(sample["frame_idx"]), [])
    return meas, icl_examples


def build_prompt_and_input(
    sample: Dict[str, Any],
    K: int,
    cfg,
    tokenizer,
    conv_module,
    num_image_token: int,
    device: torch.device,
):
    meas, icl_examples_all = load_route_assets(sample)
    speed = round(float(meas.get("speed", 0.0)), 1)
    cmd = int(meas.get("command", 4))
    next_cmd = int(meas.get("next_command", 4))
    target_point = meas.get("target_point", [0.0, 0.0])
    dist_to_command = int(np.linalg.norm(np.asarray(target_point, dtype=np.float32)))

    command_text = COMMAND_TEXT.get(cmd, "follow the road")
    next_text = COMMAND_TEXT.get(next_cmd, command_text)
    next_suffix = f" then {next_text}" if next_text != command_text else ""
    if cmd == 4:
        cmd_str = f"Command: {command_text}{next_suffix}."
    else:
        cmd_str = f"Command: {command_text} in {dist_to_command} meter{next_suffix}."

    prompt = f"Current speed: {speed} m/s. {cmd_str} Q: {sample['Q']}"

    # Image: query frame + optional ICL
    if K == 0 or not icl_examples_all:
        icl_prefix, icl_imgs = None, None
    else:
        K_eff = min(K, len(icl_examples_all))
        icl_prefix, icl_imgs = build_icl_prefix(
            icl_examples_all[:K_eff], num_icl_examples=K_eff,
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
        pv = torch.stack([transform(t) for t in tiles])
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
    system_prompt = (
        template.system_template.replace("{system_message}", template.system_message)
        + template.sep
    )
    query = query.replace(system_prompt, "")

    num_patches_all = 2
    image_tokens = (
        "<img>" + "<IMG_CONTEXT>" * num_image_token * num_patches_all + "</img>"
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
    return di


def normalize_answer(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("a:"):
        t = t[2:].strip()
    t = t.rstrip(".").strip()
    t = " ".join(t.split())
    return t.lower()


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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-list", default="0,1,2,3,4")
    ap.add_argument(
        "--out-json", default=str(REPO_ROOT / "qa_sweep_results.json"),
    )
    args = ap.parse_args()
    if args.hydra_config is None:
        # <run>/.hydra/config.yaml above the checkpoint: <run>/checkpoints/last.ckpt (DeepSpeed dir)
        # or <run>/checkpoints/epoch=000.ckpt/pytorch_model.pt (single file, as on Hugging Face)
        runs = [p for p in Path(args.ckpt).absolute().parents if (p / ".hydra" / "config.yaml").is_file()]
        args.hydra_config = str((runs[0] if runs else Path(args.ckpt).parents[1]) / ".hydra" / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_list = [int(x) for x in args.k_list.split(",")]
    print(f"[qa-sweep] device={device}  K list={k_list}  "
          f"N={args.n_samples}  seed={args.seed}")

    samples = discover_qa_samples(args.database, args.n_samples, args.seed)
    print(f"[qa-sweep] discovered {len(samples)} runnable samples")
    if len(samples) == 0:
        print("[qa-sweep] no samples found — aborting")
        return

    cfg, model, tokenizer, conv_module, num_image_token = load_vlm(
        args.hydra_config, args.ckpt, device,
    )

    results: Dict[int, Dict[str, Any]] = {}
    per_sample: Dict[int, List[Dict[str, Any]]] = {}

    for K in k_list:
        print(f"\n[qa-sweep] ── K={K} over {len(samples)} samples ───────────────")
        n_correct = 0
        n_total = 0
        latencies = []
        rows = []
        for idx, sample in enumerate(samples):
            try:
                di = build_prompt_and_input(
                    sample, K, cfg, tokenizer, conv_module,
                    num_image_token, device,
                )
            except Exception as e:
                print(f"  [skip {idx}] build failed: "
                      f"{type(e).__name__}: {e}")
                continue
            t0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    _, _, language = model(di, return_language=True)
            except Exception as e:
                print(f"  [skip {idx}] forward failed: "
                      f"{type(e).__name__}: {e}")
                continue
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            pred = language[0] if language else ""
            pred_norm = normalize_answer(pred)
            gt_norm = normalize_answer(sample["A"])
            correct = int(pred_norm == gt_norm)
            # Soft-match: GT fully contained in prediction OR vice-versa
            # (cheap proxy for "approximately right" on short answers).
            soft = int(gt_norm in pred_norm or pred_norm in gt_norm)
            n_correct += correct
            n_total += 1
            rows.append({
                "rgb_path": sample["rgb_path"],
                "Q": sample["Q"],
                "A_gt": sample["A"],
                "A_pred": pred,
                "correct": correct,
                "soft": soft,
                "cat": sample["cat"],
            })
            if (idx + 1) % 25 == 0:
                acc = n_correct / max(n_total, 1)
                print(f"  [{idx+1}/{len(samples)}] running acc={acc:.3f} "
                      f"(latency mean={np.mean(latencies):.1f} ms)")

        acc = n_correct / max(n_total, 1)
        soft_acc = sum(r["soft"] for r in rows) / max(n_total, 1)
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        per_cat = {}
        for r in rows:
            c = r["cat"]
            per_cat.setdefault(c, {"n": 0, "correct": 0, "soft": 0})
            per_cat[c]["n"] += 1
            per_cat[c]["correct"] += r["correct"]
            per_cat[c]["soft"] += r["soft"]
        per_cat_acc = {c: v["correct"] / max(v["n"], 1) for c, v in per_cat.items()}

        results[K] = {
            "accuracy": acc,
            "soft_accuracy": soft_acc,
            "n_total": n_total,
            "n_correct": n_correct,
            "mean_latency_ms": mean_lat,
            "per_category_accuracy": per_cat_acc,
            "per_category_n": {c: v["n"] for c, v in per_cat.items()},
        }
        per_sample[K] = rows
        print(f"[qa-sweep] K={K}  acc={acc:.3f}  n={n_total}  "
              f"latency={mean_lat:.1f} ms")
        for c, v in per_cat_acc.items():
            print(f"           {c:<11} acc={v:.3f}  n={per_cat[c]['n']}")

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'K':>3} | {'n':>4} | {'exact':>7} | {'soft':>7} | {'lat (ms)':>9}")
    print("-" * 78)
    for K in k_list:
        r = results[K]
        print(f"{K:>3} | {r['n_total']:>4} | "
              f"{r['accuracy']:>7.3f} | "
              f"{r['soft_accuracy']:>7.3f} | "
              f"{r['mean_latency_ms']:>9.1f}")
    print("=" * 78)

    with open(args.out_json, "w") as f:
        json.dump({
            "args": vars(args),
            "results": {str(k): v for k, v in results.items()},
            "predictions": {str(k): v for k, v in per_sample.items()},
        }, f, indent=2)
    print(f"[qa-sweep] wrote {args.out_json}")


if __name__ == "__main__":
    main()
