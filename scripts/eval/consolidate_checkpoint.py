"""Consolidate a DeepSpeed checkpoint directory once, before launching an evaluation.

    python scripts/eval/consolidate_checkpoint.py pretrained/personadrive_icl_sft/checkpoints/last.ckpt

Writes <ckpt_dir>/consolidated_fp32.pt (or $PERSONADRIVE_CONSOLIDATED_CKPT). The agent then loads that
file during setup instead of consolidating the ZeRO shards in every job (team_code/checkpoint_utils.py).
Needs no GPU, about 2x the checkpoint size in RAM, and a few minutes.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from team_code.checkpoint_utils import consolidated_path, load_checkpoint_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir", help="DeepSpeed checkpoint directory, e.g. .../checkpoints/last.ckpt")
    args = ap.parse_args()
    if not os.path.isdir(args.ckpt_dir):
        sys.exit(f"not a directory: {args.ckpt_dir} (single-file checkpoints need no consolidation)")
    state_dict = load_checkpoint_dir(args.ckpt_dir)
    out = consolidated_path(args.ckpt_dir)
    n = sum(v.numel() for v in state_dict.values())
    print(f"{len(state_dict)} tensors, {n / 1e9:.2f} B parameters -> {out}"
          + ("" if os.path.isfile(out) else " (NOT cached: directory not writable)"))


if __name__ == "__main__":
    main()
