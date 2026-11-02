"""Pinned revisions of the Hugging Face models that are loaded with remote code (trust_remote_code=True)."""

# OpenGVLab/InternVL2-1B at the snapshot PersonaDrive was trained and evaluated with.
_REVISIONS = {"OpenGVLab/InternVL2-1B": "0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187"}


def hf_revision(variant):
    """Pinned revision for a model id, or None (the repo's default branch) for anything else, e.g. a local path."""
    return _REVISIONS.get(variant)
