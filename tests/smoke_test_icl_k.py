"""Smoke test for the PERSONADRIVE_NUM_ICL context-count knob.

Exercises the real retrieval path (IclRetriever + FAISS index + build_icl_prefix)
WITHOUT CARLA or the VLM, asserting that for each K:

  * the env-var override maps K correctly (0 -> ICL off; N>0 -> ICL on, k=N)
  * retrieve(k=K) returns exactly K neighbours
  * build_icl_prefix yields exactly K images and K "<image>" tokens
  * the agent's prompt injection produces K+1 "<image>" tokens
  * T_actual (= len(icl_imgs)+1) == the prompt's <image> count  (the assertion
    the agent enforces at agent_simlingo.py:835)

Run:
    cd ${PERSONADRIVE_ROOT}
    python tests/smoke_test_icl_k.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from simlingo_training.utils.icl_retrieval import IclRetriever, _normalize_path
from simlingo_training.dataloader.dataset_driving import build_icl_prefix


def apply_num_icl_override(use_icl, num_icl_examples, env_val):
    """Replica of agent_simlingo.py:269-279 - the override under test."""
    if env_val is not None and env_val.strip() != "":
        k = int(env_val)
        if k <= 0:
            return False, 0
        return True, k
    return use_icl, num_icl_examples


def pick_query_frame(retriever):
    for meta in retriever.metadata[:500]:
        p = _normalize_path(meta.get("frame_path", ""))
        if p and os.path.exists(p):
            return Image.open(p).convert("RGB")
    print("[smoke] no on-disk frame found; using synthetic")
    arr = np.random.default_rng(0).integers(0, 255, (512, 1024, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def simulate_prompt_image_tokens(icl_prefix, icl_imgs):
    """Replica of agent_simlingo.py prompt assembly (token-count only).

    Mirrors tick(): frame_arrays = icl_imgs + [query]; prompt gets the ICL
    prefix + 'Current scene:\\n<image>\\n' + base. Returns (n_image_tokens, T).
    """
    base_prompt = "<SAFETY> Current speed: 10.0 m/s. Predict the waypoints."
    if icl_imgs:
        prompt = icl_prefix + "Current scene:\n<image>\n" + base_prompt
        T_actual = len(icl_imgs) + 1
    else:
        prompt = base_prompt
        T_actual = 1
    # Agent prepends one <image> only when none present (line 813).
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt
    return prompt.count("<image>"), T_actual


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icl_root = os.environ.get(
        "PERSONADRIVE_ICL_ROOT", str(REPO_ROOT / "persona_drive_rag"))
    icl_ckpt = os.environ.get(
        "PERSONADRIVE_ICL_CKPT", os.path.join(icl_root, "checkpoints/best_model.pt"))
    profile = os.environ.get("PERSONADRIVE_ICL_PROFILE", "normal")

    print(f"[smoke] device={device} profile={profile}")
    retriever = IclRetriever(
        checkpoint=icl_ckpt, profile=profile, root=icl_root,
        index_subdir="faiss_index", device=device, verbose=True,
    )
    ntotal = retriever.index.ntotal
    print(f"[smoke] index has {ntotal} vectors")
    query_pil = pick_query_frame(retriever)
    query_state = dict(speed=10.0, throttle=0.5, steer=0.0, brake=0.0, command=4)

    # Config baseline mirrors the checkpoint: use_icl=True, num_icl_examples=2.
    cfg_use_icl, cfg_num = True, 2

    failures = []
    print("\n[smoke] ── K override + retrieval + token accounting ──")
    for K in [0, 1, 2, 3, 4]:
        env_val = str(K)
        use_icl, num = apply_num_icl_override(cfg_use_icl, cfg_num, env_val)

        if K == 0:
            ok = (use_icl is False and num == 0)
            n_tok, T = simulate_prompt_image_tokens(None, None)
            ok = ok and (n_tok == 1 and T == 1)
            status = "PASS" if ok else "FAIL"
            print(f"  K=0  use_icl={use_icl} num={num}  "
                  f"img_tokens={n_tok} T={T}  [{status}]")
            if not ok:
                failures.append(f"K=0 override/token check (use_icl={use_icl} "
                                f"num={num} tok={n_tok} T={T})")
            continue

        # K > 0: real retrieval against the live index.
        examples = retriever.retrieve(
            query_image=query_pil, query_state=query_state, k=num)
        prefix, imgs = build_icl_prefix(
            examples, num_icl_examples=num, cut_bottom_quarter=True)

        if prefix is None or imgs is None or len(imgs) != num:
            # This is exactly the silent-fallback branch in the agent.
            n_tok, T = simulate_prompt_image_tokens(None, None)
            print(f"  K={K}  use_icl={use_icl} num={num}  "
                  f"retrieved={len(examples)} usable_imgs="
                  f"{0 if imgs is None else len(imgs)}  -> FELL BACK to K=0 "
                  f"(img_tokens={n_tok} T={T})  [FALLBACK]")
            failures.append(f"K={K} fell back (got "
                            f"{0 if imgs is None else len(imgs)}/{num} imgs)")
            continue

        prefix_tok = prefix.count("<image>")
        n_tok, T = simulate_prompt_image_tokens(prefix, imgs)
        ok = (use_icl is True and num == K
              and len(examples) == K and len(imgs) == K
              and prefix_tok == K and n_tok == K + 1 and T == K + 1)
        status = "PASS" if ok else "FAIL"
        print(f"  K={K}  use_icl={use_icl} num={num}  "
              f"retrieved={len(examples)} imgs={len(imgs)} "
              f"prefix<image>={prefix_tok}  prompt<image>={n_tok} T={T} "
              f"(expect {K} / {K+1} / {K+1})  [{status}]")
        if not ok:
            failures.append(f"K={K} count mismatch")

    # Unset env -> must keep the config default (K=2).
    print("\n[smoke] ── unset env → config default ──")
    use_icl, num = apply_num_icl_override(cfg_use_icl, cfg_num, None)
    ok = (use_icl is True and num == 2)
    print(f"  unset: use_icl={use_icl} num={num} (expect True/2)  "
          f"[{'PASS' if ok else 'FAIL'}]")
    if not ok:
        failures.append("unset env did not preserve config default")

    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST PASSED — K knob retrieves and uses the right "
          "number of context points.")


if __name__ == "__main__":
    main()
