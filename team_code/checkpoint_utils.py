"""Load a DeepSpeed checkpoint directory once, then reuse the consolidated weights.

Consolidating the ZeRO shards of `last.ckpt/` takes minutes and a lot of memory. Inside the agent's
setup that can exceed the leaderboard's 600 s setup timeout when many evaluation jobs start together,
and every job repeats the same work. `load_checkpoint_dir` consolidates once, writes the result to
`<ckpt_dir>/consolidated_fp32.pt` (or `$PERSONADRIVE_CONSOLIDATED_CKPT`), and later jobs load that
file. A file lock makes concurrent jobs wait for one consolidation instead of each running it.
If the directory is read-only the weights are returned without caching.

Run `python scripts/eval/consolidate_checkpoint.py <ckpt_dir>` once before launching an evaluation
so that no job has to consolidate during setup.
"""

import fcntl
import os

import torch

CONSOLIDATED_NAME = "consolidated_fp32.pt"


def consolidated_path(ckpt_dir):
    return os.environ.get("PERSONADRIVE_CONSOLIDATED_CKPT") or os.path.join(ckpt_dir, CONSOLIDATED_NAME)


def load_checkpoint_dir(ckpt_dir):
    """State dict of a DeepSpeed checkpoint directory (fp32), consolidated once and cached."""
    cache = consolidated_path(ckpt_dir)
    if os.path.isfile(cache):
        print(f"\033[92m[CHECKPOINT] loading consolidated weights: {cache}\033[0m", flush=True)
        return torch.load(cache, map_location="cpu")

    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    try:
        lock = open(cache + ".lock", "w")
    except OSError:
        lock = None  # not writable: consolidate in memory only
    try:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if os.path.isfile(cache):  # another job finished it while this one waited
                print(f"\033[92m[CHECKPOINT] loading consolidated weights: {cache}\033[0m", flush=True)
                return torch.load(cache, map_location="cpu")
        print(f"\033[92m[CHECKPOINT] consolidating DeepSpeed dir: {ckpt_dir}\033[0m", flush=True)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
        if lock is not None:
            tmp = f"{cache}.tmp{os.getpid()}"
            try:
                torch.save(state_dict, tmp)
                os.replace(tmp, cache)
                print(f"\033[92m[CHECKPOINT] cached consolidated weights: {cache}\033[0m", flush=True)
            except OSError as e:
                print(f"[CHECKPOINT] could not cache consolidated weights ({e}); continuing without", flush=True)
                if os.path.exists(tmp):
                    os.remove(tmp)
        return state_dict
    finally:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
