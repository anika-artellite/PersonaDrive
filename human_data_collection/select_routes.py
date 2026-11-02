#!/usr/bin/env python3
"""
select_routes.py

Weighted random route selection for collect_session.sh.

Scans available routes from one or more route directories, excludes already-driven
routes (read from the driven-route cache, or detected by scanning the dataset root),
counts how many times each scenario type has been driven globally across all
participants, then performs weighted random sampling that prioritises undersampled
scenario types.

Weight formula: w = 1 / (global_driven_count_for_scenario_type + 1)
  - An undriven scenario type gets weight 1/(0+1) = 1.0
  - A scenario type driven 9 times gets weight 1/(9+1) = 0.1
  - This naturally converges to even coverage over many participants.

During sampling, weights are updated after each pick so the N selections within a
single session also push toward balance (not just across sessions).

Output: one line per selected route, tab-separated:
  <absolute_xml_path>\t<scenario_type>
"""

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def get_scenario_type(xml_path: Path) -> str:
    """Return the first scenario type found in a route XML, or 'noScenarios'."""
    try:
        tree = ET.parse(xml_path)
        for scenario in tree.iter('scenario'):
            t = scenario.attrib.get('type', '').strip()
            if t:
                return t
    except Exception:
        pass
    return 'noScenarios'


# ---------------------------------------------------------------------------
# Route index
# ---------------------------------------------------------------------------

def build_route_index(routes_dirs: list, index_cache: Path = None) -> dict:
    """
    Scan route directories and return:
      route_name (stem) -> (xml_path, scenario_type)

    If the same stem appears in multiple dirs, the last one wins - in practice
    training_1_scenario and training_3_scenarios use non-overlapping filenames.

    The route pool is ~25k XML files, and parsing all of them on every session
    is the slowest part of a run. When index_cache is given, the scenario type
    of each file is memoised against its (mtime, size), so unchanged files are
    stat'd rather than parsed. Edited or new files fall through to a real parse.
    """
    cached = {}
    if index_cache is not None:
        try:
            with Path(index_cache).open() as fh:
                cached = json.load(fh).get('entries', {})
        except (OSError, json.JSONDecodeError):
            cached = {}

    index = {}
    entries = {}
    hits = misses = 0

    # Sidecar files Windows leaves on downloaded route archives are not routes.
    def is_route_xml(p: Path) -> bool:
        return ':' not in p.name

    for routes_dir in routes_dirs:
        p = Path(routes_dir)
        if not p.exists():
            print(f'[select_routes] WARNING: routes dir not found: {p}', file=sys.stderr)
            continue
        for xml_path in sorted(x for x in p.rglob('*.xml') if is_route_xml(x)):
            key = str(xml_path)
            try:
                st = xml_path.stat()
                sig = [int(st.st_mtime), st.st_size]
            except OSError:
                continue

            prev = cached.get(key)
            if prev and prev.get('sig') == sig:
                scenario_type = prev['scenario_type']
                hits += 1
            else:
                scenario_type = get_scenario_type(xml_path)
                misses += 1

            entries[key] = {'sig': sig, 'scenario_type': scenario_type}
            index[xml_path.stem] = (xml_path, scenario_type)

    if index_cache is not None:
        try:
            Path(index_cache).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(index_cache) + '.tmp')
            with tmp.open('w') as fh:
                json.dump({'entries': entries}, fh)
            tmp.replace(index_cache)
        except OSError as exc:
            print(f'[select_routes] WARNING: could not write route index cache: {exc}',
                  file=sys.stderr)
        print(f'[select_routes] Route index: {hits} cached, {misses} parsed',
              file=sys.stderr)

    return index


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

PERSONAS = ('slow', 'default', 'fast')


def load_driven_cache(cache_path: Path):
    """
    Read the cache written by build_driven_cache.py.

    Returns (complete, partial): route names finished for every persona, and
    route names finished for some but not all. Returns None if the cache is
    missing or unreadable so the caller can fall back to a live scan.
    """
    try:
        with Path(cache_path).open() as fh:
            cache = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    complete, partial = set(), set()
    for name, rec in cache.get('routes', {}).items():
        (complete if rec.get('complete') else partial).add(name)
    return complete, partial


def get_driven_route_names(dataset_dir: Path) -> set:
    """
    Live fallback scan when no cache is available.

    Scans every persona. Looking only at 'slow' would report a route driven as
    default/fast as undriven, so the weighted sampler would keep handing it back out
    and collect_session.sh would then skip it - burning a slot in the session plan
    each time.

    Expected structure:
      <dataset_dir>/<persona>/<person>/<scenario_type>/<route_name>/
    """
    driven = set()

    for persona in PERSONAS:
        persona_dir = dataset_dir / persona
        if not persona_dir.is_dir():
            continue
        for person_dir in persona_dir.iterdir():
            if not person_dir.is_dir():
                continue
            for scenario_dir in person_dir.iterdir():
                if not scenario_dir.is_dir():
                    continue
                for route_dir in scenario_dir.iterdir():
                    # Only count as driven if result.json exists - a bare directory
                    # means the run was started but never completed.
                    if route_dir.is_dir() and (route_dir / 'result.json').exists():
                        driven.add(route_dir.name)
    return driven


def count_scenario_types(driven_names: set, route_index: dict) -> dict:
    """
    For each driven route, look up its scenario type and count totals.
    Routes not found in the index are ignored (they came from a different
    route set and don't affect our weighting).
    """
    counts = defaultdict(int)
    for route_name in driven_names:
        if route_name in route_index:
            _, scenario_type = route_index[route_name]
            counts[scenario_type] += 1
    return counts


# ---------------------------------------------------------------------------
# Weighted sampling
# ---------------------------------------------------------------------------

def weighted_sample(pool: list, scenario_counts: dict, n: int) -> list:
    """
    Weighted random sample of n routes without replacement.

    pool: list of (route_name, xml_path, scenario_type)
    scenario_counts: dict of scenario_type -> global driven count (mutated during sampling
                     so within-session picks also push toward balance)
    """
    pool = list(pool)   # local copy

    if len(pool) < n:
        print(
            f'[select_routes] WARNING: only {len(pool)} routes available, requested {n}. '
            f'Selecting all available routes.',
            file=sys.stderr,
        )
        n = len(pool)

    selected = []
    for _ in range(n):
        if not pool:
            break

        weights = [1.0 / (scenario_counts.get(stype, 0) + 1) for _, _, stype in pool]
        total_w = sum(weights)

        r = random.uniform(0, total_w)
        cumsum = 0.0
        chosen_idx = len(pool) - 1   # fallback to last if floating-point drift
        for i, w in enumerate(weights):
            cumsum += w
            if cumsum >= r:
                chosen_idx = i
                break

        route_name, xml_path, stype = pool[chosen_idx]
        selected.append((xml_path, stype))

        # Update counts so the next pick within this session also accounts for
        # what we just chose (drives within-session diversity as well).
        scenario_counts[stype] = scenario_counts.get(stype, 0) + 1
        pool.pop(chosen_idx)

    return selected


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Weighted route selection for collect_session.sh'
    )
    parser.add_argument(
        '--routes-dirs', nargs='+', required=True,
        help='Directories containing route XML files (searched recursively)',
    )
    parser.add_argument(
        '--dataset-dirs', nargs='+', required=True,
        help='Dataset root directories to scan for already-driven routes '
             '(pass both DATASET_ROOT and ARCHIVE_ROOT if both are in use)',
    )
    parser.add_argument(
        '--n', type=int, required=True,
        help='Number of routes to select',
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility (optional)',
    )
    parser.add_argument(
        '--driven-cache', default=None,
        help='driven_routes.json from build_driven_cache.py. Used instead of a live '
             'dataset walk; falls back to scanning --dataset-dirs if absent.',
    )
    parser.add_argument(
        '--index-cache', default=None,
        help='Path for the route-index cache (memoises scenario type per XML)',
    )
    parser.add_argument(
        '--include-partial', action='store_true',
        help='Also offer routes that have some but not all personas driven, so the '
             'missing personas get filled in rather than left half-collected',
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Build full route index from all source directories
    route_index = build_route_index(args.routes_dirs, args.index_cache)
    if not route_index:
        print('[select_routes] ERROR: No route XMLs found in the given routes dirs.', file=sys.stderr)
        sys.exit(1)

    # Collect all globally driven route names. Prefer the cache - it covers every
    # persona across every dataset root in one read, and is what collect_session.sh
    # consults too, so the two never disagree about what is already done.
    partial_names: set = set()
    cached = load_driven_cache(args.driven_cache) if args.driven_cache else None

    if cached is not None:
        driven_names, partial_names = cached
        print(f'[select_routes] Driven cache: {args.driven_cache}', file=sys.stderr)
    else:
        if args.driven_cache:
            print('[select_routes] WARNING: driven cache unreadable — falling back to a live scan.',
                  file=sys.stderr)
        driven_names = set()
        for dataset_dir_str in args.dataset_dirs:
            driven_names |= get_driven_route_names(Path(dataset_dir_str))

    # Routes with some personas done are excluded by default, but can be offered
    # again with --include-partial so run_route() fills in only what is missing.
    excluded = set(driven_names)
    if not args.include_partial:
        excluded |= partial_names

    available = [
        (route_name, xml_path, scenario_type)
        for route_name, (xml_path, scenario_type) in route_index.items()
        if route_name not in excluded
    ]

    if not available:
        print('[select_routes] ERROR: No undriven routes available.', file=sys.stderr)
        sys.exit(1)

    # Count global scenario type coverage from driven routes
    scenario_counts = dict(count_scenario_types(driven_names, route_index))

    # Print diagnostic summary to stderr (visible in terminal, not captured by bash)
    total_routes = len(route_index)
    print(f'[select_routes] Total routes in pool   : {total_routes}', file=sys.stderr)
    print(f'[select_routes] Already driven globally: {len(driven_names)}  (all personas)', file=sys.stderr)
    if partial_names:
        state = 'offered again' if args.include_partial else 'excluded'
        print(f'[select_routes] Partially driven       : {len(partial_names)}  ({state})', file=sys.stderr)
    print(f'[select_routes] Available to select    : {len(available)}', file=sys.stderr)
    print(f'[select_routes] Requesting             : {args.n}', file=sys.stderr)

    if scenario_counts:
        print('[select_routes]', file=sys.stderr)
        print('[select_routes] Global driven count per scenario type (ascending = most needed):', file=sys.stderr)
        for stype, count in sorted(scenario_counts.items(), key=lambda x: x[1]):
            weight = 1.0 / (count + 1)
            print(f'[select_routes]   {count:5d} driven  (w={weight:.3f})  {stype}', file=sys.stderr)
    else:
        print('[select_routes] No routes driven yet — all scenario types weighted equally.', file=sys.stderr)

    # Weighted sample
    selected = weighted_sample(available, scenario_counts, args.n)

    print(f'[select_routes]', file=sys.stderr)
    print(f'[select_routes] Selected routes:', file=sys.stderr)
    for xml_path, stype in selected:
        print(f'[select_routes]   {stype:40s}  {xml_path.name}', file=sys.stderr)

    # Output to stdout: one line per route, tab-separated xml_path and scenario_type
    for xml_path, stype in selected:
        print(f'{xml_path}\t{stype}')


if __name__ == '__main__':
    main()
