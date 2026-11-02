#!/usr/bin/env python3
"""
build_driven_cache.py

Caches every (route, persona) pair that has already been driven, so that
collect_session.sh and select_routes.py can skip finished work without
re-walking the whole dataset tree on every lookup.

Dataset layout scanned:
    <root>/<persona>/<person>/<scenario_type>/<route_name>/result.json

A (route, persona) pair counts as driven only when result.json exists. A bare
directory means the evaluator was started but killed before writing its output,
so it is deliberately not counted -- collect_session.sh cleans those up.

Why this exists: a duplicate check that walks the dataset with `find` once per
candidate route is the slowest part of starting a session, and a check that looks
only inside the `slow` persona directory reports a route driven as default/fast but
not slow as fresh. This cache scans every persona under every dataset root once, so
both scripts agree on what "driven" means and neither pays for a rescan.

Cache format (JSON):
{
  "roots":    ["/data/personadrive", "/archive/personadrive"],
  "personas": ["slow", "default", "fast"],
  "routes": {
    "307": {
      "scenario_type": "InterurbanActorFlow",
      "personas": {"slow": [...paths...], "default": [...], "fast": [...]},
      "complete": true
    }
  }
}

Usage:
  # full rebuild
  build_driven_cache.py --out cache/driven_routes.json \
      --dataset-dirs /data/personadrive /archive/personadrive

  # incremental: record one finished drive without a full rescan
  build_driven_cache.py --out cache/driven_routes.json \
      --mark 307 --mark-persona slow --mark-scenario InterurbanActorFlow \
      --mark-path /data/personadrive/slow/person1/InterurbanActorFlow/307

  # emit "route<TAB>persona<TAB>complete" for bash to read
  build_driven_cache.py --out cache/driven_routes.json --dump-tsv
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PERSONAS = ("slow", "default", "fast")


# ---------------------------------------------------------------------------
# Markdown manifests
# ---------------------------------------------------------------------------
#
# Routes collected outside the dataset tree (an earlier dataset, another rig) are
# listed in markdown manifests under
# <dataset>/logs/*.md. They have no result.json to scan, so they are merged in
# here on every rebuild -- injecting them straight into driven_routes.json would
# not survive, because a rebuild rewrites that file from scratch.
#
# Two bullet shapes are recognised:
#   - `BlockedIntersection/4418`   -> route "4418",              scenario "BlockedIntersection"
#   - Town01_Scenario1_0           -> route "Town01_Scenario1_0", scenario from the ### heading
#
# The route key is the XML stem, which is what the dataset directories and
# select_routes.py both key on.

_MD_HEADING = re.compile(r'^###\s+(.+?)\s*$')
_MD_SCENARIO_HEADING = re.compile(r'^([A-Za-z][A-Za-z0-9_]*)\s+\(\d+\s+routes?\)$')
_MD_BULLET = re.compile(r'^\s*-\s+`?([^`\n]+?)`?\s*$')


def parse_collected_md(md_path):
    """Return {route_name: scenario_type} listed in a markdown manifest."""
    routes = {}
    try:
        text = Path(md_path).read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        print(f"[cache] WARNING: could not read {md_path}: {exc}", file=sys.stderr)
        return routes

    current_scenario = None
    for line in text.splitlines():
        heading = _MD_HEADING.match(line)
        if heading:
            # "ControlLoss (50 routes)" names a scenario; dataset headings such as
            # "<name>_dataset - driver `<name>` (15 routes; ...)" do not.
            scen = _MD_SCENARIO_HEADING.match(heading.group(1))
            current_scenario = scen.group(1) if scen else None
            continue

        bullet = _MD_BULLET.match(line)
        if not bullet:
            continue

        entry = bullet.group(1).strip()
        if not entry or entry.startswith('|') or ' ' in entry:
            continue   # table rows and prose bullets are not route ids

        if '/' in entry:
            scenario, route = entry.rsplit('/', 1)
        else:
            route, scenario = entry, current_scenario

        route = route.strip()
        if route:
            routes[route] = (scenario or '?').strip()

    return routes


def merge_collected_md(routes, md_paths):
    """
    Fold manifest routes into a scanned cache.

    These are recorded as complete for every persona and flagged external. The
    manifests record which speeds each driver actually used (some only did two of
    three), but the point of listing them is that they are finished work, so they
    are excluded outright rather than offered again to fill a missing persona.
    A real scanned result always wins over a manifest entry.
    """
    added = 0
    for md_path in md_paths:
        listed = parse_collected_md(md_path)
        for route, scenario in listed.items():
            rec = routes.get(route)
            if rec is None:
                rec = {
                    "scenario_type": scenario,
                    "personas": {p: [f"external:{md_path}"] for p in PERSONAS},
                    "complete": True,
                    "external": True,
                    "source": str(md_path),
                }
                routes[route] = rec
                added += 1
            else:
                # Already collected for real; just note the manifest also lists it.
                rec.setdefault("source", str(md_path))
        print(f"[cache] manifest {md_path}: {len(listed)} route(s) listed", file=sys.stderr)
    return added


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_roots(roots, personas=PERSONAS):
    """
    Walk every dataset root and return route_name -> record.

    This looks under all three persona dirs, so a route driven as default/fast is
    recognised as partially done rather than new.
    """
    routes = {}

    for root in roots:
        root = Path(root)
        if not root.is_dir():
            print(f"[cache] WARNING: dataset root not found: {root}", file=sys.stderr)
            continue

        for persona in personas:
            persona_dir = root / persona
            if not persona_dir.is_dir():
                continue

            for person_dir in sorted(persona_dir.iterdir()):
                if not person_dir.is_dir():
                    continue
                for scenario_dir in sorted(person_dir.iterdir()):
                    if not scenario_dir.is_dir():
                        continue
                    for route_dir in sorted(scenario_dir.iterdir()):
                        if not route_dir.is_dir():
                            continue
                        if not (route_dir / "result.json").is_file():
                            continue

                        rec = routes.setdefault(
                            route_dir.name,
                            {
                                "scenario_type": scenario_dir.name,
                                "personas": {p: [] for p in personas},
                                "complete": False,
                            },
                        )
                        rec["personas"].setdefault(persona, [])
                        rec["personas"][persona].append(str(route_dir))

    for rec in routes.values():
        rec["complete"] = all(rec["personas"].get(p) for p in personas)

    return routes


def build_cache(roots, personas=PERSONAS, collected_md=()):
    routes = scan_roots(roots, personas)
    scanned = len(routes)
    added = merge_collected_md(routes, collected_md) if collected_md else 0
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "roots": [str(r) for r in roots],
        "manifests": [str(m) for m in collected_md],
        "personas": list(personas),
        "scanned_routes": scanned,
        "manifest_routes": added,
        "routes": routes,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_cache(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[cache] WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None


def write_cache(cache, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
    tmp.replace(path)

    write_text_report(cache, path.with_suffix(".txt"))


def write_text_report(cache, path):
    """Human-readable companion to the JSON, for eyeballing coverage."""
    routes = cache.get("routes", {})
    personas = cache.get("personas", list(PERSONAS))

    complete = {n: r for n, r in routes.items() if r.get("complete")}
    partial = {n: r for n, r in routes.items() if not r.get("complete")}

    by_scenario = {}
    for name, rec in routes.items():
        by_scenario.setdefault(rec.get("scenario_type", "?"), []).append(name)

    external = {n: r for n, r in routes.items() if r.get("external")}

    lines = [
        "Driven route cache",
        f"Generated : {cache.get('generated', '?')}",
        f"Roots     : {', '.join(cache.get('roots', []))}",
        f"Manifests : {', '.join(cache.get('manifests', [])) or 'none'}",
        "",
        f"Routes known   : {len(routes)}",
        f"  scanned      : {len(routes) - len(external)}  (result.json found on disk)",
        f"  from manifest: {len(external)}  (collected elsewhere, excluded outright)",
        f"  complete     : {len(complete)}  (all {len(personas)} personas)",
        f"  partial      : {len(partial)}",
        "",
        "Per scenario type:",
    ]
    for stype in sorted(by_scenario):
        names = sorted(by_scenario[stype])
        n_done = sum(1 for n in names if routes[n].get("complete"))
        n_ext = sum(1 for n in names if routes[n].get("external"))
        suffix = f"  ({n_ext} from manifest)" if n_ext else ""
        lines.append(f"  {stype:40s} {n_done}/{len(names)} complete{suffix}")

    if partial:
        lines += ["", "Partially driven (missing personas will be filled in on the next session):"]
        for name in sorted(partial):
            rec = partial[name]
            have = [p for p in personas if rec["personas"].get(p)]
            missing = [p for p in personas if not rec["personas"].get(p)]
            lines.append(
                f"  {rec.get('scenario_type', '?')}/{name}"
                f"   have={','.join(have) or '-'}   missing={','.join(missing)}"
            )

    scanned_complete = sorted(n for n in complete if not routes[n].get("external"))
    lines += ["", f"Complete — scanned from disk ({len(scanned_complete)}):"]
    for name in scanned_complete:
        lines.append(f"  {complete[name].get('scenario_type', '?')}/{name}")

    ext_complete = sorted(n for n in complete if routes[n].get("external"))
    lines += ["", f"Complete — from manifest ({len(ext_complete)}):"]
    for name in ext_complete:
        lines.append(f"  {complete[name].get('scenario_type', '?')}/{name}")

    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build/update the driven-route cache")
    ap.add_argument("--out", required=True, help="Path to driven_routes.json")
    ap.add_argument("--dataset-dirs", nargs="+", default=None,
                    help="Dataset roots to scan (required for a full rebuild)")
    ap.add_argument("--collected-md", nargs="*", default=[],
                    help="Markdown manifests of routes collected outside these roots "
                         "(e.g. <dataset>/logs/collected_routes.md). Merged on every "
                         "rebuild, since a rebuild rewrites the cache from scratch.")
    ap.add_argument("--mark", metavar="ROUTE",
                    help="Record a single finished drive instead of rescanning")
    ap.add_argument("--mark-persona", choices=list(PERSONAS))
    ap.add_argument("--mark-scenario", default=None)
    ap.add_argument("--mark-path", default=None)
    ap.add_argument("--dump-tsv", action="store_true",
                    help="Print route<TAB>persona<TAB>complete for each recorded drive")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # --- incremental mark -------------------------------------------------
    if args.mark:
        if not args.mark_persona:
            ap.error("--mark requires --mark-persona")

        cache = load_cache(args.out)
        if cache is None:
            cache = {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "roots": args.dataset_dirs or [],
                "personas": list(PERSONAS),
                "routes": {},
            }

        rec = cache["routes"].setdefault(
            args.mark,
            {
                "scenario_type": args.mark_scenario or "?",
                "personas": {p: [] for p in PERSONAS},
                "complete": False,
            },
        )
        if args.mark_scenario:
            rec["scenario_type"] = args.mark_scenario
        rec["personas"].setdefault(args.mark_persona, [])
        if args.mark_path and args.mark_path not in rec["personas"][args.mark_persona]:
            rec["personas"][args.mark_persona].append(args.mark_path)
        elif not args.mark_path and not rec["personas"][args.mark_persona]:
            rec["personas"][args.mark_persona].append("(marked)")

        rec["complete"] = all(rec["personas"].get(p) for p in PERSONAS)
        cache["generated"] = datetime.now().isoformat(timespec="seconds")

        write_cache(cache, args.out)
        if not args.quiet:
            state = "COMPLETE" if rec["complete"] else "partial"
            print(f"[cache] marked {args.mark} / {args.mark_persona} -> {state}",
                  file=sys.stderr)
        return

    # --- dump for bash ----------------------------------------------------
    if args.dump_tsv:
        cache = load_cache(args.out)
        if cache is None:
            return
        for name, rec in cache.get("routes", {}).items():
            complete = "1" if rec.get("complete") else "0"
            for persona in cache.get("personas", PERSONAS):
                if rec["personas"].get(persona):
                    print(f"{name}\t{persona}\t{complete}")
        return

    # --- full rebuild -----------------------------------------------------
    if not args.dataset_dirs:
        ap.error("--dataset-dirs is required for a full rebuild")

    md_paths = [m for m in args.collected_md if Path(m).is_file()]
    cache = build_cache(args.dataset_dirs, collected_md=md_paths)
    write_cache(cache, args.out)

    routes = cache["routes"]
    n_complete = sum(1 for r in routes.values() if r["complete"])
    n_external = sum(1 for r in routes.values() if r.get("external"))
    if not args.quiet:
        print(
            f"[cache] {len(routes)} route(s) known "
            f"({cache['scanned_routes']} scanned + {n_external} from manifests), "
            f"{n_complete} complete, {len(routes) - n_complete} partial -> {args.out}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
