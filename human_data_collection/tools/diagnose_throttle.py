#!/usr/bin/env python3
"""
diagnose_throttle.py — find why full throttle sometimes reads as no throttle.

Two causes produce that symptom and they need different fixes:

  * 16-bit underflow wrap. The raw field bottoms out at 0 at full press; a sensor value
    just below that wraps to 65535, which decodes as "released". Anomalies cluster at the
    bottom of travel and the jump is always ~0 -> ~65535. This is a software bug and is
    fixable in wheel_hidraw.py.

  * Electrical dropout (loose wiring). The signal vanishes and the wheel reports its
    default. Anomalies appear at arbitrary pedal positions, not just the extreme. This is
    hardware and no amount of decoding fixes it.

Press the throttle fully, hard, several times — including however you provoke the fault.

    python3 diagnose_throttle.py --seconds 60
"""

import argparse
import os
import select
import sys
import time

THROTTLE_LO = 45          # low byte of the little-endian 16-bit throttle field
FULL_SCALE = 65535
REPORT_SIZE = 64
JUMP = 20000              # a single-report change this large is not physically possible


def find_device():
    # wheel_hidraw.py lives one level up, alongside human_agent.py.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from wheel_hidraw import G923Reader
    return G923Reader._find_device()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or find_device()
    fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)

    print(f"Reading {device}.")
    print("\n  Press the throttle FULLY and HARD, several times.")
    print("  Reproduce the fault if you can. Sampling starts on first input.\n",
          flush=True)

    samples = []
    while not samples:
        r, _, _ = select.select([fd], [], [], 1.0)
        if r:
            d = os.read(fd, 512)
            if len(d) >= REPORT_SIZE:
                samples.append((0.0, d[THROTTLE_LO] | (d[THROTTLE_LO + 1] << 8)))
    print(f"  [input detected — sampling for {args.seconds:.0f}s]\n", flush=True)

    t0 = time.time()
    deadline = t0 + args.seconds
    while time.time() < deadline:
        r, _, _ = select.select([fd], [], [], 0.2)
        if not r:
            continue
        d = os.read(fd, 512)
        if len(d) < REPORT_SIZE:
            continue
        samples.append((time.time() - t0, d[THROTTLE_LO] | (d[THROTTLE_LO + 1] << 8)))
    os.close(fd)

    print("=" * 68)
    print(f"  {len(samples)} samples")
    print("=" * 68)

    values = [v for _t, v in samples]
    print(f"  raw range reached : {min(values)} .. {max(values)}")
    print(f"  decoded throttle  : {1 - max(values)/FULL_SCALE:.3f} .. "
          f"{1 - min(values)/FULL_SCALE:.3f}")

    anomalies = []
    for i in range(1, len(samples)):
        (_, prev), (t, cur) = samples[i - 1], samples[i]
        if abs(cur - prev) >= JUMP:
            anomalies.append((t, prev, cur))

    print(f"\n  impossible single-report jumps (>={JUMP}): {len(anomalies)}")
    if not anomalies:
        print("\n  No anomaly captured. If the fault did not occur during this run,")
        print("  re-run and reproduce it. If it DID occur, the problem is not a jump")
        print("  in this field — tell me what you saw and I will widen the search.")
        return 0

    near_bottom = 0
    for t, prev, cur in anomalies:
        kind = ""
        if prev < 3000 and cur > 60000:
            kind = "  <-- underflow wrap at full press (SOFTWARE)"
            near_bottom += 1
        elif cur > 60000:
            kind = "  <-- dropout to released from mid-travel (WIRING)"
        print(f"    t={t:6.2f}  {prev:5d} -> {cur:5d}   "
              f"(pedal {1-prev/FULL_SCALE:.2f} -> {1-cur/FULL_SCALE:.2f}){kind}")

    print("\n" + "=" * 68)
    print("  VERDICT")
    print("=" * 68)
    if near_bottom == len(anomalies):
        print("  Every anomaly is a 0 -> 65535 wrap at the bottom of travel.")
        print("  This is an underflow in the decode, NOT your wiring. Fixable.")
    elif near_bottom == 0:
        print("  Anomalies occur away from full press, jumping straight to the")
        print("  released value. That is an electrical dropout — check the pedal")
        print("  unit's DIN cable and the connector at the wheel base.")
    else:
        print(f"  Mixed: {near_bottom} wrap-like, {len(anomalies)-near_bottom} not.")
        print("  Likely a decode underflow AND a flaky connection. Fix the decode")
        print("  first, then re-run to see whether the rest survive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
