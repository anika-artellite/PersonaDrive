"""Path resolution for stale absolute paths baked into dataset metadata.

Frame paths inside files like icl_examples.json.gz are absolute and were
captured under whatever root the data lived under at index-build time
(historically a job-scoped tmp dir). At runtime we remap them onto the
persistent on-share database root.
"""

import os


_PERSISTENT_DATA_ROOT = (
    f"{os.environ.get('PERSONADRIVE_ROOT', '/path/to/PersonaDrive')}/database"
)
_PATH_REWRITES = (
    ("/fast_dataset/data/simlingo/",       "/data/simlingo/fast/"),
    ("/slow_dataset/data/simlingo/",       "/data/simlingo/slow/"),
    ("/fast_dataset/drivelm/simlingo/",    "/drivelm/simlingo/fast/"),
    ("/slow_dataset/drivelm/simlingo/",    "/drivelm/simlingo/slow/"),
    ("/fast_dataset/commentary/simlingo/", "/commentary/simlingo/fast/"),
    ("/slow_dataset/commentary/simlingo/", "/commentary/simlingo/slow/"),
)

_DATA_ROOTS_CACHE = None
_PATH_CACHE: dict = {}


def _discover_data_roots():
    # PERSONADRIVE_DATA_ROOT (colon-separated) wins; otherwise use the persistent root.
    roots = []
    env = os.environ.get("PERSONADRIVE_DATA_ROOT", "")
    for r in env.split(":"):
        r = r.strip().rstrip("/")
        if r and os.path.isdir(r):
            roots.append(r)
    if os.path.isdir(_PERSISTENT_DATA_ROOT):
        roots.append(_PERSISTENT_DATA_ROOT)
    seen, out = set(), []
    for r in roots:
        rp = os.path.realpath(r)
        if rp not in seen:
            seen.add(rp); out.append(rp)
    return out


def _data_roots():
    global _DATA_ROOTS_CACHE
    if _DATA_ROOTS_CACHE is None:
        _DATA_ROOTS_CACHE = _discover_data_roots()
    return _DATA_ROOTS_CACHE


def _normalize_path(path: str) -> str:
    """Map a (possibly-stale) absolute path to a real path on this node.

    The configured data roots are searched first; the original absolute path is used only when no root
    holds the file (same order as simlingo_training.utils.icl_retrieval._normalize_path).
    """
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
    _PATH_CACHE[path] = path
    return path
