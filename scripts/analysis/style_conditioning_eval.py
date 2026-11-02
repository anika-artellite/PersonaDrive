"""
Style-conditioning cross-matrix at K=2.

Question: when we condition retrieval on a driving-style profile, does the
model actually drive in that style?

ADE/FDE against a single ground truth cannot answer that on its own - if you
condition on `fast` while the GT is a `normal` log, a faithful style-follower
gets a WORSE ADE, not a better one. So we run a cross-conditioning matrix:

    rows    = the style whose routes the query frames (and GT) come from
    columns = the profile whose FAISS index supplies the K=2 examples

If style conditioning works, error is minimised on the DIAGONAL (retrieval
profile == the GT's own style). The falsification test is built in: if one
column wins in *every* row, that is a retrieval-pool artifact (the indices
differ 64x in size: normal 54040, slow 8960, fast 4559, default 848), not
style-following.

Second, GT-free evidence: on identical query frames we measure the implied
kinematics of the PREDICTED waypoints. Conditioning on `fast` should raise
implied speed relative to `slow` regardless of what the GT did. This is
immune to the "maybe those routes are just easier" confound.

Usage:
    python scripts/analysis/style_conditioning_eval.py --n-per-style 50
"""
from __future__ import annotations

import argparse
import json
import random
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from waypoint_accuracy_sweep import (   # noqa: E402
    build_inputs, ego_frame_waypoints, load_vlm,
)
from simlingo_training.utils.icl_retrieval import IclRetriever  # noqa: E402

# Query-style name -> subdirectory under database/simlingo/data/simlingo.
# `normal` has no dedicated dir; training_1_scenario is the normal-style pool
# that the `normal` FAISS index was built from.
STYLE_DIRS = {
    "default": "default",
    "fast":    "fast",
    "normal":  "training_1_scenario",
    "slow":    "slow",
}

# measurements are stored every data_save_freq(=5) sim frames at carla_fps(=20)
# -> 4 Hz -> 0.25 s between consecutive stored frames (team_code/config.py:339-341).
DT = 0.25


def discover_in(data_root: Path, n: int, horizon: int, seed: int,
                skip_first_n: int = 10, per_route: int = 3) -> List[Dict[str, Any]]:
    """Sample frames from one style's routes.

    Mirrors waypoint_accuracy_sweep.discover_samples but (a) takes the leaf
    data dir directly and (b) drops the icl_examples.json.gz requirement -
    live FAISS retrieval never reads it, and the `default` style has none.
    """
    rng = random.Random(seed)
    routes = sorted(data_root.rglob("measurements.json"))
    rng.shuffle(routes)
    out: List[Dict[str, Any]] = []
    for meas_path in routes:
        route_dir = meas_path.parent
        rgb_dir = route_dir / "rgb"
        if not rgb_dir.is_dir():
            continue
        try:
            with open(meas_path) as f:
                measurements = json.load(f)
        except Exception:
            continue
        valid = [t for t in range(skip_first_n, len(measurements) - horizon - 1)
                 if (rgb_dir / f"{t:04d}.jpg").exists()]
        if not valid:
            continue
        rng.shuffle(valid)
        for t in valid[:per_route]:
            out.append({
                "route_dir": str(route_dir),
                "meas_path": str(meas_path),
                "icl_path":  str(route_dir / "icl_examples.json.gz"),  # unused
                "rgb_path":  str(rgb_dir / f"{t:04d}.jpg"),
                "frame_idx": t,
            })
            if len(out) >= n:
                return out
    return out


def kinematics(wps: np.ndarray) -> Dict[str, float]:
    """Implied kinematics of an ego-frame waypoint sequence [H, 2]."""
    steps = np.linalg.norm(np.diff(np.vstack([[0.0, 0.0], wps]), axis=0), axis=1)
    seg_speed = steps / DT
    return {
        "speed_mean": float(seg_speed.mean()),      # mean implied speed over horizon
        "speed_first": float(seg_speed[0]),         # instantaneous implied speed
        "path_len": float(steps.sum()),             # distance covered in horizon
        "reach": float(np.linalg.norm(wps[-1])),    # net displacement
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hydra-config", default=None, help="default: <run>/.hydra/config.yaml next to --ckpt")
    ap.add_argument("--ckpt",
                    default=os.environ.get("PERSONADRIVE_CKPT") or str(REPO_ROOT / "pretrained/personadrive_icl_sft/checkpoints/epoch=000.ckpt/pytorch_model.pt"))
    ap.add_argument("--database", default=str(REPO_ROOT / "database/simlingo"))
    ap.add_argument("--icl-root", default=str(REPO_ROOT / "persona_drive_rag"))
    ap.add_argument("--icl-ckpt", default=None)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-per-style", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=11)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-styles", default="default,fast,normal,slow")
    ap.add_argument("--profiles", default="default,fast,normal,slow")
    ap.add_argument("--use-cot", action="store_true", default=True)
    ap.add_argument("--out-json", default=str(REPO_ROOT / "style_conditioning_results.json"))
    args = ap.parse_args()
    if args.hydra_config is None:
        # <run>/.hydra/config.yaml above the checkpoint: <run>/checkpoints/last.ckpt (DeepSpeed dir)
        # or <run>/checkpoints/epoch=000.ckpt/pytorch_model.pt (single file, as on Hugging Face)
        runs = [p for p in Path(args.ckpt).absolute().parents if (p / ".hydra" / "config.yaml").is_file()]
        args.hydra_config = str((runs[0] if runs else Path(args.ckpt).parents[1]) / ".hydra" / "config.yaml")
    if args.icl_ckpt is None:
        args.icl_ckpt = str(Path(args.icl_root) / "checkpoints/best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_styles = [s.strip() for s in args.query_styles.split(",") if s.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    data_root = Path(args.database) / "data" / "simlingo"

    print(f"[style] device={device}  K={args.k}  N/style={args.n_per_style}  "
          f"horizon={args.horizon} ({args.horizon * DT:.2f}s @ dt={DT})")

    # query sets, one per style
    samples: Dict[str, List[Dict[str, Any]]] = {}
    for st in q_styles:
        sub = data_root / STYLE_DIRS[st]
        samples[st] = discover_in(sub, args.n_per_style, args.horizon, args.seed)
        print(f"[style]   query set {st:8s} -> {len(samples[st]):3d} frames  ({sub.name})")

    cfg, model, tokenizer, conv_module, num_image_token = load_vlm(
        args.hydra_config, args.ckpt, device)

    # GT reference: cache GT waypoints + logged speed per sample
    gt_cache: Dict[str, List[Dict[str, Any]]] = {}
    for st in q_styles:
        rows = []
        for s in samples[st]:
            with open(s["meas_path"]) as f:
                meas = json.load(f)
            gt = ego_frame_waypoints(meas, s["frame_idx"], args.horizon)
            rows.append({"gt": gt,
                         "logged_speed": float(meas[s["frame_idx"]].get("speed", 0.0)),
                         "kin": kinematics(gt)})
        gt_cache[st] = rows
        sp = np.array([r["kin"]["speed_mean"] for r in rows])
        lg = np.array([r["logged_speed"] for r in rows])
        print(f"[style]   GT {st:8s}  implied speed {sp.mean():5.2f} m/s   "
              f"logged speed {lg.mean():5.2f} m/s   (dt sanity check)")

    results: Dict[str, Dict[str, Any]] = {}

    # Outer loop = profile, so each FAISS index is loaded exactly once.
    for prof in profiles:
        print(f"\n[style] ══ retrieval profile = {prof} ══")
        retriever = IclRetriever(checkpoint=args.icl_ckpt, profile=prof,
                                 root=args.icl_root, device=device, verbose=False)
        print(f"[style]   index ntotal = {retriever.index.ntotal}")

        for st in q_styles:
            ade_l, fde_l, kin_l, n_skip = [], [], [], 0
            for i, s in enumerate(samples[st]):
                try:
                    di, _, _ = build_inputs(
                        s, args.k, cfg, tokenizer, conv_module,
                        num_image_token, device, retriever=retriever,
                        use_cot=args.use_cot)
                    with torch.inference_mode():
                        speed_wps, _route, _lang = model(di)
                except Exception as e:
                    n_skip += 1
                    if n_skip <= 3:
                        print(f"    [skip {st}/{i}] {type(e).__name__}: {e}")
                    continue
                pred = speed_wps.float().cpu().numpy()[0]
                gt = gt_cache[st][i]["gt"]
                H = min(pred.shape[0], gt.shape[0])
                err = np.linalg.norm(pred[:H] - gt[:H], axis=1)
                ade_l.append(float(err.mean()))
                fde_l.append(float(err[-1]))
                kin_l.append(kinematics(pred[:H]))

            if not ade_l:
                print(f"  {st:8s}  no usable samples")
                continue
            cell = {
                "n": len(ade_l), "n_skipped": n_skip,
                "ADE_mean": float(np.mean(ade_l)),
                "ADE_sem": float(np.std(ade_l, ddof=1) / np.sqrt(len(ade_l))),
                "FDE_mean": float(np.mean(fde_l)),
                "FDE_sem": float(np.std(fde_l, ddof=1) / np.sqrt(len(fde_l))),
                "pred_speed_mean": float(np.mean([k["speed_mean"] for k in kin_l])),
                "pred_speed_sem": float(np.std([k["speed_mean"] for k in kin_l], ddof=1)
                                        / np.sqrt(len(kin_l))),
                "pred_path_len": float(np.mean([k["path_len"] for k in kin_l])),
                "ADE_per_sample": ade_l,
                "pred_speed_per_sample": [k["speed_mean"] for k in kin_l],
            }
            results[f"{st}|{prof}"] = cell
            print(f"  query={st:8s}  n={cell['n']:3d}  "
                  f"ADE {cell['ADE_mean']:.3f}+-{cell['ADE_sem']:.3f}   "
                  f"FDE {cell['FDE_mean']:.3f}   "
                  f"pred speed {cell['pred_speed_mean']:.2f} m/s")

        del retriever
        torch.cuda.empty_cache()

    # matrices
    def matrix(metric: str, prec: int = 3):
        print(f"\n=== {metric}  (rows = GT style, cols = retrieval profile) ===")
        hdr = "GT\\prof"
        print(f"{hdr:>10}" + "".join(f"{p:>12}" for p in profiles) + f"{'argmin':>10}")
        for st in q_styles:
            vals = [results.get(f"{st}|{p}", {}).get(metric) for p in profiles]
            cells = "".join(f"{v:>12.{prec}f}" if v is not None else f"{'--':>12}"
                            for v in vals)
            good = [(v, p) for v, p in zip(vals, profiles) if v is not None]
            best = min(good)[1] if good else "--"
            mark = " <=DIAG" if best == st else ""
            print(f"{st:>10}{cells}{best:>10}{mark}")

    matrix("ADE_mean")
    matrix("FDE_mean")
    print(f"\n=== predicted implied speed m/s  (rows = query frames, "
          f"cols = retrieval profile) ===")
    qhdr = "query\\prof"
    print(f"{qhdr:>10}" + "".join(f"{p:>12}" for p in profiles))
    for st in q_styles:
        vals = [results.get(f"{st}|{p}", {}).get("pred_speed_mean") for p in profiles]
        print(f"{st:>10}" + "".join(f"{v:>12.2f}" if v is not None else f"{'--':>12}"
                                    for v in vals))

    with open(args.out_json, "w") as f:
        json.dump({"args": vars(args), "dt": DT, "results": results}, f, indent=2)
    print(f"\n[style] wrote {args.out_json}")


if __name__ == "__main__":
    main()
