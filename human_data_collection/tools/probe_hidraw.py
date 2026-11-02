#!/usr/bin/env python3
"""
probe_hidraw.py — find which HID interface the wheel actually reports input on.

evdev only surfaces what hid-generic managed to parse. This reads the raw HID report
streams instead, so it sees data even on interfaces the kernel could not turn into an
input device. Run it, move the wheel and pedals, and it reports which interface produced
changing bytes and which byte offsets moved.

    sudo chmod 666 /dev/hidraw*        # once, so this can run without root
    python3 probe_hidraw.py --seconds 60

Byte offsets that move are the raw axis positions; that is enough to tell whether the
wheel is reporting at all, and on which interface.
"""

import argparse
import glob
import os
import select
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--devices", nargs="*", default=None,
                    help="defaults to every /dev/hidraw*")
    args = ap.parse_args()

    paths = args.devices or sorted(glob.glob("/dev/hidraw*"))
    if not paths:
        sys.exit("No /dev/hidraw* nodes found.")

    fds, names, denied = {}, {}, []
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError:
            denied.append(p)
            continue
        except OSError as e:
            print(f"  {p}: {e}")
            continue
        fds[fd] = p
        names[p] = os.path.basename(p)

    if denied:
        print("PERMISSION DENIED on: " + ", ".join(denied))
        print("Run:  sudo chmod 666 /dev/hidraw*\n")
    if not fds:
        sys.exit("No readable hidraw nodes — nothing to probe.")

    print(f"Probing {', '.join(names[p] for p in fds.values())} for {args.seconds:.0f}s.")
    print("Move the wheel lock to lock, then press each pedal fully.\n")

    # per device: report count, first report, and which byte offsets ever changed
    stats = {p: {"n": 0, "first": None, "moved": {}, "size": 0} for p in fds.values()}

    deadline = time.time() + args.seconds
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        r, _, _ = select.select(list(fds), [], [], min(remaining, 0.2))
        for fd in r:
            path = fds[fd]
            try:
                data = os.read(fd, 512)
            except (BlockingIOError, OSError):
                continue
            if not data:
                continue
            st = stats[path]
            st["n"] += 1
            st["size"] = max(st["size"], len(data))
            if st["first"] is None:
                st["first"] = data
                continue
            base = st["first"]
            for i in range(min(len(base), len(data))):
                if data[i] != base[i]:
                    lo, hi = st["moved"].get(i, (base[i], base[i]))
                    st["moved"][i] = (min(lo, data[i]), max(hi, data[i]))

    for fd in fds:
        os.close(fd)

    print("=" * 62)
    print("  RESULTS")
    print("=" * 62)
    active = False
    for path, st in stats.items():
        name = names[path]
        if st["n"] == 0:
            print(f"  {name}: silent — no reports at all")
            continue
        print(f"  {name}: {st['n']} reports, {st['size']} bytes each")
        if not st["moved"]:
            print("      all reports identical — device is answering polls with static data")
            continue
        active = True
        print("      CHANGING BYTES (offset: range) — this interface carries live input:")
        for off in sorted(st["moved"]):
            lo, hi = st["moved"][off]
            print(f"        byte[{off:2d}] {lo:3d}..{hi:3d}  (span {hi - lo})")
    print()
    if active:
        print("  At least one interface reports live input. If that interface is NOT the")
        print("  one backing /dev/input/event0, the kernel cannot parse it into axes and")
        print("  the wheel needs hid-logitech-hidpp (absent here) or a userspace reader.")
    else:
        print("  NOTHING changed on any interface while you were moving the wheel.")
        print("  The wheel is answering polls but not sensing input — check that the")
        print("  base's external power supply is connected and that it performed its")
        print("  power-on calibration spin.")


if __name__ == "__main__":
    main()
