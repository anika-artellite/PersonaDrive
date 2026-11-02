#!/usr/bin/env python3
"""
map_wheel_bytes.py — work out which raw HID bytes carry steering, throttle, brake, clutch.

hid-generic drops the G923's axis fields ("ignoring exceeding usage max"), so evdev never
sees them, but they are plainly visible in the raw 64-byte reports on /dev/hidraw0. This
records the stream while you operate one control at a time, then segments the recording on
idle gaps and reports which bytes were active in each burst.

Segmenting on gaps rather than a fixed schedule means you do not have to hit any timing —
just pause a couple of seconds between controls.

    python3 map_wheel_bytes.py --seconds 80

Writes the raw capture to wheel_capture.bin so the analysis can be re-run without
having to operate the wheel again:

    python3 map_wheel_bytes.py --analyze wheel_capture.bin
"""

import argparse
import os
import select
import struct
import sys
import time

GAP = 1.2          # seconds of quiet that separates one control from the next
MIN_BURST = 0.35   # ignore blips shorter than this
REC_FMT = "dH"     # timestamp, report length; report bytes follow


def capture(device, seconds, out_path):
    try:
        fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    except PermissionError:
        sys.exit(f"ERROR: cannot read {device} — run: sudo chmod 666 /dev/hidraw*")
    except FileNotFoundError:
        sys.exit(f"ERROR: {device} does not exist.")

    print(f"Recording {device} for {seconds:.0f}s.\n")
    print("  Operate ONE control at a time, pausing ~2s between them:")
    print("    1. steering — full left, full right, re-center")
    print("    2. throttle — fully down, release")
    print("    3. brake    — fully down, release")
    print("    4. clutch   — fully down, release")
    print("\n  Pauses are what separate them; exact timing does not matter.\n", flush=True)

    records = []
    t0 = time.time()
    deadline = t0 + seconds
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        r, _, _ = select.select([fd], [], [], min(remaining, 0.2))
        if not r:
            continue
        try:
            data = os.read(fd, 512)
        except (BlockingIOError, OSError):
            continue
        if data:
            records.append((time.time() - t0, data))
    os.close(fd)

    with open(out_path, "wb") as f:
        for ts, data in records:
            f.write(struct.pack(REC_FMT, ts, len(data)))
            f.write(data)
    print(f"Captured {len(records)} reports -> {out_path}\n")
    return records


def load(path):
    records, hdr = [], struct.calcsize(REC_FMT)
    with open(path, "rb") as f:
        blob = f.read()
    off = 0
    while off + hdr <= len(blob):
        ts, n = struct.unpack(REC_FMT, blob[off:off + hdr])
        off += hdr
        records.append((ts, blob[off:off + n]))
        off += n
    return records


def analyze(records):
    if len(records) < 2:
        print("Not enough data to analyze.")
        return

    base = records[0][1]
    width = min(len(r[1]) for r in records)

    # A report is "active" if any byte differs from the previous report.
    events = []          # (timestamp, {offset: value})
    prev = base
    for ts, data in records:
        changed = {i: data[i] for i in range(width) if data[i] != prev[i]}
        if changed:
            events.append((ts, changed))
        prev = data

    if not events:
        print("No byte ever changed — nothing to map.")
        return

    # Segment on idle gaps.
    bursts, cur = [], [events[0]]
    for ev in events[1:]:
        if ev[0] - cur[-1][0] > GAP:
            bursts.append(cur)
            cur = [ev]
        else:
            cur.append(ev)
    bursts.append(cur)
    bursts = [b for b in bursts if b[-1][0] - b[0][0] >= MIN_BURST]

    print("=" * 66)
    print(f"  {len(bursts)} activity burst(s) detected")
    print("=" * 66)

    labels = ["steering", "throttle", "brake", "clutch"]
    for n, burst in enumerate(bursts):
        span = {}
        for _ts, changed in burst:
            for off, val in changed.items():
                lo, hi = span.get(off, (val, val))
                span[off] = (min(lo, val), max(hi, val))
        guess = labels[n] if n < len(labels) else f"burst {n + 1}"
        dur = burst[-1][0] - burst[0][0]
        print(f"\n  Burst {n + 1}  t={burst[0][0]:.1f}..{burst[-1][0]:.1f}s "
              f"({dur:.1f}s)   expected: {guess}")
        ranked = sorted(span.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True)
        for off, (lo, hi) in ranked:
            if hi - lo < 8:      # noise / jitter
                continue
            print(f"      byte[{off:2d}]  {lo:3d}..{hi:3d}   span {hi - lo}")

    # Bytes that only ever moved in one burst are unambiguous.
    print("\n" + "=" * 66)
    print("  PER-BYTE ATTRIBUTION")
    print("=" * 66)
    owner = {}
    for n, burst in enumerate(bursts):
        for _ts, changed in burst:
            for off in changed:
                owner.setdefault(off, set()).add(n)
    for off in sorted(owner):
        bs = sorted(owner[off])
        tag = (labels[bs[0]] if len(bs) == 1 and bs[0] < len(labels)
               else "shared/ambiguous")
        print(f"    byte[{off:2d}] active in burst(s) {[b + 1 for b in bs]}  -> {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/hidraw0")
    ap.add_argument("--seconds", type=float, default=80.0)
    ap.add_argument("--out", default="wheel_capture.bin")
    ap.add_argument("--analyze", metavar="CAPTURE",
                    help="re-analyze an existing capture instead of recording")
    args = ap.parse_args()

    records = load(args.analyze) if args.analyze else capture(
        args.device, args.seconds, args.out)
    analyze(records)


if __name__ == "__main__":
    main()
