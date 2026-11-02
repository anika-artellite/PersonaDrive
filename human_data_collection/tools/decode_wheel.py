#!/usr/bin/env python3
"""
decode_wheel.py — confirm the G923 byte pairs are 16-bit axes and recover their calibration.

map_wheel_bytes.py showed four byte pairs moving as units. This decides whether each pair
is one 16-bit value or two independent 8-bit ones, picks the endianness, and reports the
resting/min/max values needed to normalize each axis.

The test is smoothness: a physical control moves continuously, so the correct decode
produces small deltas between consecutive reports. A wrong endianness turns a one-step
move into a ~256-step jump, so its mean delta is dramatically larger.

    python3 decode_wheel.py wheel_capture.bin
"""

import argparse
import struct
import sys
from collections import Counter

REC_FMT = "dH"

PAIRS = [((43, 44), "steering"), ((45, 46), "throttle"),
         ((47, 48), "brake"), ((49, 50), "clutch")]


def load(path):
    records, hdr = [], struct.calcsize(REC_FMT)
    with open(path, "rb") as f:
        blob = f.read()
    off, out = 0, []
    while off + hdr <= len(blob):
        ts, n = struct.unpack(REC_FMT, blob[off:off + hdr])
        off += hdr
        out.append((ts, blob[off:off + n]))
        off += n
    return out


def mean_abs_delta(series):
    if len(series) < 2:
        return float("inf")
    deltas = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
    return sum(deltas) / len(deltas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", default="wheel_capture.bin")
    args = ap.parse_args()

    records = load(args.capture)
    if not records:
        sys.exit(f"No records in {args.capture}")
    print(f"{len(records)} reports from {args.capture}\n")

    print("=" * 70)
    print("  DECODE TEST — lower mean delta wins")
    print("=" * 70)

    results = {}
    for (lo_off, hi_off), name in PAIRS:
        le = [d[lo_off] | (d[hi_off] << 8) for _t, d in records if len(d) > hi_off]
        be = [d[hi_off] | (d[lo_off] << 8) for _t, d in records if len(d) > hi_off]
        b_lo = [d[lo_off] for _t, d in records if len(d) > hi_off]
        b_hi = [d[hi_off] for _t, d in records if len(d) > hi_off]

        d_le, d_be = mean_abs_delta(le), mean_abs_delta(be)
        # Compare against treating the low byte alone as an 8-bit axis, scaled to
        # the same 16-bit span so the numbers are comparable.
        d_8 = mean_abs_delta(b_lo) * 256

        best, series = ("little-endian 16-bit", le) if d_le <= d_be else \
                       ("big-endian 16-bit", be)
        print(f"\n  {name}  bytes[{lo_off},{hi_off}]")
        print(f"      LE 16-bit mean delta : {d_le:10.2f}")
        print(f"      BE 16-bit mean delta : {d_be:10.2f}")
        print(f"      low byte as 8-bit    : {d_8:10.2f}  (x256 for comparison)")
        print(f"      -> {best}")
        results[name] = (lo_off, hi_off, best, series, b_lo, b_hi)

    print("\n" + "=" * 70)
    print("  CALIBRATION")
    print("=" * 70)
    for name, (lo_off, hi_off, best, series, _bl, _bh) in results.items():
        mn, mx = min(series), max(series)
        rest = Counter(series).most_common(1)[0][0]
        print(f"\n  {name:9s} bytes[{lo_off},{hi_off}]  {best}")
        print(f"      observed range : {mn} .. {mx}")
        print(f"      resting value  : {rest}")
        if mx > mn:
            pct = 100.0 * (rest - mn) / (mx - mn)
            print(f"      rest sits at   : {pct:.1f}% of observed range")
        bits = (mx - mn).bit_length()
        print(f"      effective res  : ~{bits}-bit ({mx - mn + 1} distinct steps)")


if __name__ == "__main__":
    main()
