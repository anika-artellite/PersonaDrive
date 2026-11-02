"""
Print the paper's result tables from your own evaluation outputs.

This script only formats. The numbers come from the original scripts:

  Closed loop (Table 1, Table 3, closed-loop rows of the K table)
      scripts/eval/aggregate_b2d_metrics.py    on <single_route>/        (K = 2, one row per profile)
      scripts/eval/aggregate_b2d_metrics_k.py  on <single_route>_k<N>/   (K = 1, 3, 4; profile normal)
  Open loop and per-tick cost (K table)
      scripts/analysis/waypoint_accuracy_sweep.py --use-cot     -> --sweep-json
      scripts/analysis/benchmark_icl_timing.py --k-list 0,1,2,3,4  -> --timing-json (retrieval overhead)
  Human speed statistics (appendix)
      scripts/analysis/style_speed_stats.py

Profiles: normal = no style (style-agnostic PDM-lite index); slow / default / fast =
Conservative / Neutral / Aggressive. Comfort is printed x100, as in the paper.
The K = 0 closed-loop column is SimLingo's published result (the paper cites it),
so it is printed as a constant.

Usage:
    python scripts/eval/paper_tables.py                       # closed loop only
    python scripts/eval/paper_tables.py --sweep-json waypoint_sweep_results.json \
        --timing-json bench_icl_timing.json --out-dir paper_tables/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
REPO_ROOT = HERE.parents[1]

# SimLingo, Bench2Drive (renz2025simlingo): the K = 0 column of the K table.
CITED_K0_SIMLINGO = {"DS": 85.07, "SuccessRate%": 67.27, "Efficiency": 259.23, "Comfortness": 0.3367}
STYLES = (("slow", "Conservative"), ("default", "Neutral"), ("fast", "Aggressive"))


def run(cmd: list) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO_ROOT)


def read_csv(path: Path, key: str) -> dict:
    with open(path, newline="") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def num(row: dict, col: str, scale: float = 1.0) -> str:
    v = (row or {}).get(col, "")
    return "—" if v in ("", None) else f"{float(v) * scale:.2f}"


def routes(row: dict) -> str:
    return "—" if not row else f"{row['n_valid']}/{row['n_total']}"


def print_table(title: str, header: list, rows: list) -> None:
    print(f"\n{title}")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for r in [header, ["-" * w for w in widths]] + rows:
        print("  " + "  ".join(str(x).rjust(w) if i else str(x).ljust(w) for i, (x, w) in enumerate(zip(r, widths))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-route",
                    default=os.environ.get("PERSONADRIVE_EVAL_OUT_ROOT", str(REPO_ROOT / "eval_results" / "single_route")),
                    help="K = 2 output root of scripts/eval/start_eval_simlingo.py; the K sweep lives in "
                         "the sibling <single-route>_k<N>/ dirs (default: $PERSONADRIVE_EVAL_OUT_ROOT)")
    ap.add_argument("--k", type=int, nargs="*", default=[1, 3, 4], help="K-sweep values to aggregate")
    ap.add_argument("--sweep-json", default=None, help="output of waypoint_accuracy_sweep.py --use-cot")
    ap.add_argument("--timing-json", default=None, help="output of benchmark_icl_timing.py")
    ap.add_argument("--out-dir", default="paper_tables", help="where the aggregator CSVs are written")
    args = ap.parse_args()

    single = Path(args.single_route).absolute()
    out = Path(args.out_dir).absolute()
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # closed loop, K = 2, one row per profile
    prof = {}
    if single.is_dir():
        run([py, HERE / "aggregate_b2d_metrics.py", "--root", single,
             "--csv", out / "b2d_summary.csv", "--per-route-csv", out / "b2d_per_route.csv"])
        prof = read_csv(out / "b2d_summary.csv", "profile")
    else:
        print(f"!! {single} not found: no closed-loop K = 2 results")

    # closed loop, K sweep (single_route_k<N>)
    by_k = {}
    k_found = [k for k in args.k if (single.parent / f"{single.name}_k{k}").is_dir()]
    if k_found and single.name == "single_route":
        run([py, HERE / "aggregate_b2d_metrics_k.py", "--base", single.parent, "--k", *k_found,
             "--csv", out / "b2d_k_summary.csv", "--per-route-csv", out / "b2d_k_per_route.csv"])
        by_k = {int(k): r for k, r in read_csv(out / "b2d_k_summary.csv", "k").items()}
    elif k_found:
        print(f"!! the K-sweep aggregator expects dirs named single_route_k<N>; rename {single.name}_k<N>")
    if prof.get("normal"):
        by_k[2] = prof["normal"]

    sweep = json.load(open(args.sweep_json))["results"] if args.sweep_json else {}
    timing = json.load(open(args.timing_json))["results"] if args.timing_json else {}

    # Table 1
    n = prof.get("normal")
    print_table("Table 1 — Bench2Drive without style (PersonaDrive row; profile normal, K = 2)",
                ["Method", "DS", "SR (%)", "Eff.", "Comf.", "routes"],
                [["PersonaDrive (no style)", num(n, "DS"), num(n, "SuccessRate%"), num(n, "Efficiency"),
                  num(n, "Comfortness", 100), routes(n)]])

    # K table
    ks = [0, 1, 2, 3, 4]
    header = ["Metric"] + [f"K={k}" for k in ks]
    rows = []
    if sweep:
        def sw(k, col, fmt):
            r = sweep.get(str(k))
            return "—" if r is None else fmt.format(r[col])
        rows += [["ADE (m)"] + [sw(k, "ADE_mean_m", "{:.3f}") for k in ks],
                 ["FDE (m)"] + [sw(k, "FDE_mean_m", "{:.3f}") for k in ks]]
    cl = {0: {k: str(v) for k, v in CITED_K0_SIMLINGO.items()}, **by_k}
    rows += [["DS"] + [num(cl.get(k), "DS") for k in ks],
             ["SR (%)"] + [num(cl.get(k), "SuccessRate%") for k in ks],
             ["Eff."] + [num(cl.get(k), "Efficiency") for k in ks],
             ["Comf."] + [num(cl.get(k), "Comfortness", 100) for k in ks],
             ["routes"] + ["cited"] + [routes(cl.get(k)) for k in ks[1:]]]
    if sweep:
        rag = {k: (timing[str(k)]["embedding"]["mean"] + timing[str(k)]["lookup"]["mean"]) if str(k) in timing else None
               for k in ks}
        rows += [["Tokens"] + [sw(k, "prompt_tokens_mean", "{:.0f}") for k in ks],
                 ["TTFT (ms)"] + [sw(k, "ttft_mean_ms", "{:.1f}") for k in ks],
                 ["TTLT (ms)"] + [sw(k, "ttlt_mean_ms", "{:.1f}") for k in ks],
                 ["Inference total (ms)"] + [sw(k, "total_mean_ms", "{:.1f}") for k in ks]]
        if timing:
            rows += [["RAG overhead (ms)"] + ["—" if rag[k] is None else f"{rag[k]:.2f}" for k in ks],
                     ["Total (ms)"] + ["—" if rag[k] is None or str(k) not in sweep else
                                       f"{sweep[str(k)]['total_mean_ms'] + rag[k]:.1f}" for k in ks]]
    print_table("Table 2 / appendix — retrieval depth (closed loop: profile normal; K = 0 is SimLingo, cited)",
                header, rows)

    # Table 3
    print_table("Table 3 — Bench2Drive per style (PersonaDrive rows, K = 2)",
                ["Style", "DS", "SR (%)", "Eff.", "Comfort", "Speed (m/s)", "Accel. (m/s²)", "routes"],
                [[label, num(prof.get(p), "DS"), num(prof.get(p), "SuccessRate%"), num(prof.get(p), "Efficiency"),
                  num(prof.get(p), "Comfortness", 100), num(prof.get(p), "Speed_mps"),
                  num(prof.get(p), "LonAccel_mps2"), routes(prof.get(p))] for p, label in STYLES])

    print("\nroutes = routes with a completed result / routes found. The aggregators average over routes with a "
          "result and, when a route was run more than once, keep its best run.")
    print("Human speed statistics (appendix): python scripts/analysis/style_speed_stats.py")


if __name__ == "__main__":
    main()
