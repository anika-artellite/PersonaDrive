"""
scripts/data_processing/convert_run.py
--------------
Single-route worker used by `scripts/data_processing/records_integrating.py`. Merges all per-frame
files inside `<run_dir>/<subfolder>/` into a null-padded JSON array written
to `<run_dir>/<subfolder>.json`.

Null-padding semantics:
    Output list length = max_frame_idx + 1.
    Output[i] = data for frame i, or None if that frame's file is missing
    (including any leading gap when frames don't start at 0000).

This is what the SimLingo dataloader expects:
    frame_idx = sample_start + hist_len - 1
    commentary_data[frame_idx]   # must be the actual frame_idx, not a list
                                 # position that depends on what existed.
"""
import gzip
import json
import os
import tempfile


def _load_frame(path):
    """Load a single per-frame file. Supports .json and .json.gz."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _atomic_write_json(out_path, frames, dir_for_tmp):
    """Write `frames` to `out_path` atomically (temp file + rename)."""
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_for_tmp, suffix='.json.tmp')
    try:
        with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            json.dump(frames, f)
        os.rename(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def convert_run(run_dir, subfolder):
    """
    Merge `<run_dir>/<subfolder>/<NNNN>.json[.gz]` files into a single
    null-padded JSON list at `<run_dir>/<subfolder>.json`.

    Args:
        run_dir:   absolute path to the route/run directory
        subfolder: name of the per-frame subfolder
                   (one of 'measurements', 'dreamer', 'commentary', 'vqa')

    Returns one of:
        'OK'    : merged successfully
        'SKIP'  : output already exists, nothing to do
        'EMPTY' : per-frame folder has no recognised files
        'ERROR: <msg>' : something went wrong
    """
    out_path  = os.path.join(run_dir, subfolder + '.json')
    src_dir   = os.path.join(run_dir, subfolder)

    if os.path.exists(out_path):
        return 'SKIP'

    if not os.path.isdir(src_dir):
        return 'ERROR: source folder not found: {0}'.format(src_dir)

    # Collect (frame_idx, path) pairs. Prefer .json.gz when both exist for the
    # same stem. Skip files whose stem isn't a non-negative integer.
    indexed = {}
    for fname in os.listdir(src_dir):
        if fname.endswith('.json.gz'):
            stem = fname[:-len('.json.gz')]
        elif fname.endswith('.json'):
            stem = fname[:-len('.json')]
        else:
            continue
        try:
            frame_idx = int(stem)
        except ValueError:
            # Non-numeric filename - leave it out so we can't accidentally
            # shift the array. The merge is for numerically-indexed frames.
            continue
        if frame_idx < 0:
            continue
        # .json.gz wins over .json for the same frame
        existing = indexed.get(frame_idx)
        if existing is None or (existing.endswith('.json') and fname.endswith('.json.gz')):
            indexed[frame_idx] = os.path.join(src_dir, fname)

    if not indexed:
        return 'EMPTY'

    max_idx = max(indexed.keys())
    frames = [None] * (max_idx + 1)
    for frame_idx, path in indexed.items():
        try:
            frames[frame_idx] = _load_frame(path)
        except Exception as e:
            return 'ERROR: failed reading {0}: {1}'.format(os.path.basename(path), e)

    try:
        _atomic_write_json(out_path, frames, run_dir)
    except Exception as e:
        return 'ERROR: write failed: {0}'.format(e)

    return 'OK'
