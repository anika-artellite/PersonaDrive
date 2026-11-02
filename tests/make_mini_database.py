"""Build a small copy of the PersonaDrive dataset for the end-to-end smoke test.

Copies a few complete routes of every kind the pipeline consumes, keeping the
exact on-disk layout described in datasets/README.md:

    <out>/simlingo/{data,commentary,drivelm}/simlingo/
        default|fast|slow/driver<N>/routes/<Scenario>/<id>/<Town...>/   human style routes
        fast|slow/lb1_split/routes_training/<Scenario>/<inst>/<Town...>/ expert style routes
        training_1_scenario/routes_training/<weather>/<Town...>/        PDM-lite routes

`icl_examples.json.gz` files are NOT copied: the smoke test regenerates them.

Usage:
    python tests/make_mini_database.py --src /path/to/database --out /path/to/mini_database
"""
import argparse
import glob
import os
import shutil
import sys

KINDS = ("data", "commentary", "drivelm")
SKIP_FILES = {"icl_examples.json.gz"}


def route_leaves(src_simlingo, pattern):
    """Sorted route leaf dirs (under data/simlingo) matching a glob pattern."""
    base = os.path.join(src_simlingo, "data", "simlingo")
    return sorted(d for d in glob.glob(os.path.join(base, pattern)) if os.path.isdir(d))


def copy_route(src_simlingo, out_simlingo, leaf):
    rel = os.path.relpath(leaf, os.path.join(src_simlingo, "data", "simlingo"))
    copied = 0
    for kind in KINDS:
        src = os.path.join(src_simlingo, kind, "simlingo", rel)
        if not os.path.isdir(src):
            if kind == "data":
                raise FileNotFoundError(src)
            continue
        dst = os.path.join(out_simlingo, kind, "simlingo", rel)
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=lambda d, names: [n for n in names if n in SKIP_FILES])
        copied += 1
    return rel, copied


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.environ.get("PERSONADRIVE_DATA_ROOT", "").split(":")[0],
                    help="Dataset root containing simlingo/ (default: $PERSONADRIVE_DATA_ROOT)")
    ap.add_argument("--out", required=True, help="Output dataset root (simlingo/ is created inside)")
    ap.add_argument("--n-human", type=int, default=1, help="human routes per style (default 1)")
    ap.add_argument("--n-expert", type=int, default=2, help="lb1_split routes per fast/slow (default 2)")
    ap.add_argument("--n-pdmlite", type=int, default=3, help="PDM-lite routes (default 3)")
    args = ap.parse_args()

    src_simlingo = os.path.join(args.src, "simlingo")
    out_simlingo = os.path.join(args.out, "simlingo")
    if not os.path.isdir(os.path.join(src_simlingo, "data", "simlingo")):
        sys.exit(f"ERROR: {src_simlingo}/data/simlingo not found (set --src or PERSONADRIVE_DATA_ROOT)")

    selection = []
    for style in ("default", "fast", "slow"):
        selection += route_leaves(src_simlingo, f"{style}/driver*/routes/*/*/Town*")[: args.n_human]
    for style in ("fast", "slow"):
        selection += route_leaves(src_simlingo, f"{style}/lb1_split/routes_training/*/*/Town*")[: args.n_expert]
    selection += route_leaves(src_simlingo, "training_1_scenario/routes_training/*/Town*")[: args.n_pdmlite]

    expected = 3 * args.n_human + 2 * args.n_expert + args.n_pdmlite
    if len(selection) != expected:
        sys.exit(f"ERROR: found {len(selection)} routes, expected {expected}")
    for leaf in selection:
        rel, n = copy_route(src_simlingo, out_simlingo, leaf)
        print(f"  copied ({n} trees) {rel}")
    print(f"Mini database with {len(selection)} routes at {out_simlingo}")


if __name__ == "__main__":
    main()
