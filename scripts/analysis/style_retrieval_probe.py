"""
Diagnostic: does the retrieval *signal* differ across style profiles at all?

Runs no VLM. For a fixed set of query frames it retrieves K examples from each
profile's FAISS index and reports the style-relevant content of what came back:
mean speed of the retrieved examples and the implied speed of their waypoints.

This disambiguates the two ways style-conditioning can fail:
  - retrieved examples DO differ by profile, predictions don't  -> model ignores
    the in-context style.
  - retrieved examples DON'T differ by profile                  -> the indices
    aren't separating styles; the model never gets a style signal to follow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from style_conditioning_eval import STYLE_DIRS, DT, discover_in  # noqa: E402
from simlingo_training.utils.icl_retrieval import IclRetriever   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default=str(REPO_ROOT / "database/simlingo"))
    ap.add_argument("--icl-root", default=str(REPO_ROOT / "persona_drive_rag"))
    ap.add_argument("--icl-ckpt", default=None)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-per-style", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=11)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-styles", default="default,fast,normal,slow")
    ap.add_argument("--profiles", default="default,fast,normal,slow")
    ap.add_argument("--out-json", default=str(REPO_ROOT / "style_retrieval_probe.json"))
    args = ap.parse_args()
    if args.icl_ckpt is None:
        args.icl_ckpt = str(Path(args.icl_root) / "checkpoints/best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_styles = [s.strip() for s in args.query_styles.split(",") if s.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    data_root = Path(args.database) / "data" / "simlingo"

    samples = {}
    for st in q_styles:
        samples[st] = discover_in(data_root / STYLE_DIRS[st], args.n_per_style,
                                  args.horizon, args.seed)
        print(f"[probe] query set {st:8s} -> {len(samples[st])} frames")

    table = {}
    for prof in profiles:
        r = IclRetriever(checkpoint=args.icl_ckpt, profile=prof,
                         root=args.icl_root, device=device, verbose=False)
        print(f"\n[probe] profile={prof}  ntotal={r.index.ntotal}")
        for st in q_styles:
            ex_speed, ex_wp_speed, scores = [], [], []
            for s in samples[st]:
                with open(s["meas_path"]) as f:
                    meas = json.load(f)
                m = meas[s["frame_idx"]]
                try:
                    exs = r.retrieve(
                        query_image=s["rgb_path"],
                        query_state={"speed": round(float(m.get("speed", 0.0)), 1),
                                     "throttle": float(m.get("throttle", 0.0)),
                                     "steer": float(m.get("steer", 0.0)),
                                     "brake": float(m.get("brake", 0.0)),
                                     "command": int(m.get("command", 4))},
                        k=args.k, exclude_self_path=s["rgb_path"])
                except Exception as e:
                    print(f"  [skip] {type(e).__name__}: {e}")
                    continue
                for ex in exs:
                    sh = ex.get("speed_history") or []
                    if sh:
                        ex_speed.append(float(sh[-1]))
                    wp = np.asarray(ex.get("waypoints") or [], dtype=np.float32)
                    if wp.ndim == 2 and len(wp) >= 2:
                        steps = np.linalg.norm(
                            np.diff(np.vstack([[0.0, 0.0], wp]), axis=0), axis=1)
                        ex_wp_speed.append(float((steps / DT).mean()))
                    scores.append(ex.get("_score", float("nan")))
            if not ex_speed:
                print(f"  query={st:8s}  no retrievals")
                continue
            cell = {"n_ex": len(ex_speed),
                    "ex_speed_mean": float(np.mean(ex_speed)),
                    "ex_speed_sem": float(np.std(ex_speed, ddof=1) / np.sqrt(len(ex_speed))),
                    "ex_wp_speed_mean": float(np.mean(ex_wp_speed)) if ex_wp_speed else None,
                    "score_mean": float(np.nanmean(scores))}
            table[f"{st}|{prof}"] = cell
            wps = cell["ex_wp_speed_mean"]
            print(f"  query={st:8s}  retrieved-example speed "
                  f"{cell['ex_speed_mean']:5.2f}+-{cell['ex_speed_sem']:.2f} m/s   "
                  f"wp-implied {wps if wps is None else round(wps,2)}   "
                  f"sim {cell['score_mean']:.4f}")
        del r
        torch.cuda.empty_cache()

    print("\n=== retrieved-example speed m/s (rows=query frames, cols=profile) ===")
    hdr = "query\\prof"
    print(f"{hdr:>10}" + "".join(f"{p:>12}" for p in profiles))
    for st in q_styles:
        vals = [table.get(f"{st}|{p}", {}).get("ex_speed_mean") for p in profiles]
        print(f"{st:>10}" + "".join(f"{v:>12.2f}" if v is not None else f"{'--':>12}"
                                   for v in vals))

    with open(args.out_json, "w") as f:
        json.dump({"args": vars(args), "results": table}, f, indent=2)
    print(f"\n[probe] wrote {args.out_json}")


if __name__ == "__main__":
    main()
