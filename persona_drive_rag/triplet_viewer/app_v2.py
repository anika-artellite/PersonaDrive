"""
Triplet Viewer v2 - reads from triplets_v2/<profile>/triplets.jsonl.

Features over v1:
  - Auto-discover available profiles under triplets_v2/
  - Profile selector via ?profile=<name>
  - Stats bar: total triplets, action distribution, driver distribution
  - Filter by action (Follow Lane / Go Left / ...) and driver (driverN / lb1_split)
  - Anchor frame_id search (kept from v1)
  - Pagination (kept from v1)
  - Profile / driver / action badges on each anchor

Run:
    python app_v2.py
"""

import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

from flask import Flask, abort, render_template, request, send_file, url_for

app = Flask(__name__)

# Paths
SCRIPT_DIR   = Path(__file__).resolve().parent
TRIPLETS_DIR = (SCRIPT_DIR / ".." / "triplets_v2").resolve()

PER_PAGE       = 10
MAX_PAGE_LINKS = 9


# Profile + path helpers
def discover_profiles() -> list[str]:
    """Return sorted profile names that have a triplets.jsonl on disk."""
    if not TRIPLETS_DIR.is_dir():
        return []
    return sorted(
        d.name for d in TRIPLETS_DIR.iterdir()
        if d.is_dir() and (d / "triplets.jsonl").exists()
    )


def _parse_profile_driver(frame_path: str) -> tuple[str, str]:
    """Pull (profile, driver) out of an absolute frame_path of the form
       .../database/simlingo/data/simlingo/<profile>/<driver>/.../rgb/<n>.jpg

    Returns ("", "") if the pattern isn't found.
    """
    parts = frame_path.split("/")
    for i in range(len(parts) - 2):
        if parts[i] == "data" and parts[i + 1] == "simlingo":
            profile = parts[i + 2] if i + 2 < len(parts) else ""
            driver  = parts[i + 3] if i + 3 < len(parts) else ""
            return profile, driver
    return "", ""


@lru_cache(maxsize=8)
def load_triplets(profile: str) -> list:
    """Load + cache all triplets for a profile and tag each anchor with
    profile / driver derived from the frame_path."""
    p = TRIPLETS_DIR / profile / "triplets.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            anchor = t.setdefault("anchor", {})
            prof, drv = _parse_profile_driver(anchor.get("frame_path", ""))
            anchor.setdefault("profile", prof)
            anchor.setdefault("driver", drv)
            out.append(t)
    return out


def summarize(triplets: list) -> dict:
    actions = Counter(t["anchor"].get("action", "?") for t in triplets)
    drivers = Counter(t["anchor"].get("driver", "?") for t in triplets)
    return {
        "total":   len(triplets),
        "actions": actions.most_common(),
        "drivers": drivers.most_common(),
    }


# Routes
@app.route("/")
def index():
    profiles = discover_profiles()

    # Empty-state: no triplets_v2/ dir or nothing inside it
    if not profiles:
        return render_template(
            "index_v2.html",
            profiles=[], profile="",
            stats={"total": 0, "actions": [], "drivers": []},
            triplets=[], page=1, total_pages=1, total=0,
            anchor_id="", action_filter="", driver_filter="",
            start_page=1, end_page=1,
        )

    profile = request.args.get("profile", profiles[0], type=str)
    if profile not in profiles:
        profile = profiles[0]

    all_triplets = load_triplets(profile)
    stats        = summarize(all_triplets)

    anchor_id     = request.args.get("anchor_id", "", type=str).strip()
    action_filter = request.args.get("action",    "", type=str).strip()
    driver_filter = request.args.get("driver",    "", type=str).strip()

    working = all_triplets
    if anchor_id:
        working = [t for t in working if t["anchor"].get("frame_id") == anchor_id]
    if action_filter:
        working = [t for t in working if t["anchor"].get("action") == action_filter]
    if driver_filter:
        working = [t for t in working if t["anchor"].get("driver") == driver_filter]

    total       = len(working)
    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    page = max(1, min(request.args.get("page", 1, type=int), total_pages))

    start = (page - 1) * PER_PAGE
    paginated = working[start : start + PER_PAGE]

    # Compact pagination window
    if total_pages <= MAX_PAGE_LINKS:
        start_page, end_page = 1, total_pages
    else:
        half = MAX_PAGE_LINKS // 2
        if page <= half:
            start_page, end_page = 1, MAX_PAGE_LINKS
        elif page >= total_pages - half:
            start_page = total_pages - MAX_PAGE_LINKS + 1
            end_page   = total_pages
        else:
            start_page = page - half
            end_page   = page + half

    return render_template(
        "index_v2.html",
        profiles=profiles,
        profile=profile,
        stats=stats,
        triplets=paginated,
        page=page,
        total_pages=total_pages,
        total=total,
        anchor_id=anchor_id,
        action_filter=action_filter,
        driver_filter=driver_filter,
        start_page=start_page,
        end_page=end_page,
    )


@app.route("/<path:image_path>")
def get_image(image_path: str):
    full_path = "/" + image_path
    if not os.path.exists(full_path):
        abort(404)
    return send_file(full_path)


if __name__ == "__main__":
    app.run(debug=True, port=5050)