#!/usr/bin/env python3
"""
calibrate_wheel.py — probe the racing wheel and report the evdev -> SDL axis mapping.

Reads /dev/input/event* directly, so it does not depend on pygame or SDL. Run it and
follow the timed prompts; it reports which axis moved during each phase, which is what
wheel_config.ini needs to get right.

    python3 calibrate_wheel.py                 # 40s, four 10s phases
    python3 calibrate_wheel.py --device /dev/input/event0 --phase-seconds 8

The SDL axis index (what wheel_config.ini stores) is the position of an ABS code among
the ABS codes the device reports, in ascending code order — ABS_X=0, ABS_Y=1, ABS_Z=2,
ABS_RX=3, ABS_RY=4, ABS_RZ=5 for a device exposing all six.
"""

import argparse
import fcntl
import os
import select
import struct
import sys
import time

# struct input_event { struct timeval time; __u16 type; __u16 code; __s32 value; }
EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)

EV_KEY, EV_ABS = 0x01, 0x03

ABS_NAMES = {
    0: "ABS_X", 1: "ABS_Y", 2: "ABS_Z",
    3: "ABS_RX", 4: "ABS_RY", 5: "ABS_RZ",
    16: "ABS_HAT0X", 17: "ABS_HAT0Y",
}

ABSINFO_SIZE = 24  # 6 x __s32
KEY_MAX = 0x2FF
KEY_BYTES = KEY_MAX // 8 + 1


def eviocgabs(axis):
    return (2 << 30) | (ABSINFO_SIZE << 16) | (ord("E") << 8) | (0x40 + axis)


def eviocgbit(ev, length):
    return (2 << 30) | (length << 16) | (ord("E") << 8) | (0x20 + ev)


def read_key_codes(fd):
    """Every EV_KEY code the device declares, ascending.

    SDL numbers joystick buttons by position in this list, so the index here is the
    button index wheel_config.ini wants for `reverse` and `handbrake`.
    """
    buf = bytearray(KEY_BYTES)
    try:
        fcntl.ioctl(fd, eviocgbit(EV_KEY, KEY_BYTES), buf, True)
    except OSError:
        return []
    return [c for c in range(KEY_MAX + 1) if buf[c // 8] >> (c % 8) & 1]


def read_absinfo(fd, axis):
    buf = bytearray(ABSINFO_SIZE)
    try:
        fcntl.ioctl(fd, eviocgabs(axis), buf, True)
    except OSError:
        return None
    value, minimum, maximum, fuzz, flat, resolution = struct.unpack("6i", bytes(buf))
    if maximum == minimum:
        return None
    return {"value": value, "min": minimum, "max": maximum,
            "fuzz": fuzz, "flat": flat, "res": resolution}


PHASES = [
    ("STEERING", "Turn the wheel fully LEFT, then fully RIGHT, then re-center."),
    ("THROTTLE", "Press the THROTTLE (right pedal) fully down, then release."),
    ("BRAKE",    "Press the BRAKE (middle pedal) fully down, then release."),
    ("CLUTCH",   "Press the CLUTCH (left pedal) fully down, then release."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/input/event0")
    ap.add_argument("--phase-seconds", type=float, default=10.0)
    ap.add_argument("--monitor", type=float, metavar="SECONDS",
                    help="Skip the timed phases: watch everything for SECONDS and "
                         "report whatever moved. Use when timing the phases is awkward.")
    args = ap.parse_args()

    phases = PHASES
    if args.monitor:
        phases = [("ALL", "Move EVERYTHING: wheel lock to lock, then each pedal "
                          "one at a time, then press some buttons.")]

    try:
        fd = os.open(args.device, os.O_RDONLY | os.O_NONBLOCK)
    except PermissionError:
        sys.exit(f"ERROR: cannot read {args.device} — add yourself to the 'input' group.")
    except FileNotFoundError:
        sys.exit(f"ERROR: {args.device} does not exist.")

    present = [a for a in sorted(ABS_NAMES) if read_absinfo(fd, a)]
    linear = [a for a in present if a < 16]          # hats are not SDL axes
    sdl_index = {code: i for i, code in enumerate(linear)}
    key_codes = read_key_codes(fd)
    sdl_button = {code: i for i, code in enumerate(key_codes)}

    print(f"Device : {args.device}")
    print(f"Axes   : {', '.join(ABS_NAMES[a] for a in present)}")
    for a in linear:
        info = read_absinfo(fd, a)
        bits = (info["max"] - info["min"] + 1).bit_length() - 1
        print(f"  sdl[{sdl_index[a]}] {ABS_NAMES[a]:10s} range {info['min']}..{info['max']}  (~{bits}-bit)")
    print(f"Buttons: {len(key_codes)}")
    print()

    # code -> (min seen, max seen, event count) per phase
    seen = {}
    buttons = set()
    total_events = 0

    for name, instruction in phases:
        duration = args.monitor if args.monitor else args.phase_seconds
        print("=" * 60)
        print(f"  {name}: {instruction}")
        print(f"  ({duration:.0f} seconds — starting now)")
        print("=" * 60, flush=True)

        deadline = time.time() + duration
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            r, _, _ = select.select([fd], [], [], min(remaining, 0.2))
            if not r:
                continue
            data = os.read(fd, EVENT_SIZE * 64)
            for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                _s, _us, etype, code, value = struct.unpack(EVENT_FMT, data[off:off + EVENT_SIZE])
                total_events += 1
                if etype == EV_ABS and code in sdl_index:
                    key = (name, code)
                    lo, hi, n = seen.get(key, (value, value, 0))
                    seen[key] = (min(lo, value), max(hi, value), n + 1)
                elif etype == EV_KEY and value == 1:
                    buttons.add(code)

        moved = sorted(((c, seen[(name, c)]) for c in linear if (name, c) in seen),
                       key=lambda kv: kv[1][1] - kv[1][0], reverse=True)
        if not moved:
            print(f"  -> no axis movement detected during {name}\n")
            continue
        for code, (lo, hi, n) in moved:
            span = hi - lo
            info = read_absinfo(fd, code)
            pct = 100.0 * span / (info["max"] - info["min"])
            print(f"  -> sdl[{sdl_index[code]}] {ABS_NAMES[code]:10s} "
                  f"moved {lo}..{hi} (span {span}, {pct:.0f}% of range, {n} events)")
        print()

    os.close(fd)

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if total_events == 0:
        print("  NO EVENTS AT ALL. The kernel received nothing from the wheel.")
        print("  The device is enumerated but not reporting — check the USB passthrough")
        print("  (usbipd attach), and that the wheel has completed its power-on spin.")
        return

    print(f"  {total_events} events received.\n")

    if args.monitor:
        print("  Axis activity (largest span first) — map these to pedals by "
              "how far each moved:")
        rows = [(c, seen[("ALL", c)]) for c in linear if ("ALL", c) in seen]
        for code, (lo, hi, n) in sorted(rows, key=lambda kv: kv[1][1] - kv[1][0],
                                        reverse=True):
            print(f"    sdl[{sdl_index[code]}] {ABS_NAMES[code]:10s} "
                  f"{lo}..{hi}  (span {hi - lo}, {n} events)")
        idle = [c for c in linear if ("ALL", c) not in seen]
        if idle:
            print("    never moved: " +
                  ", ".join(f"sdl[{sdl_index[c]}] {ABS_NAMES[c]}" for c in idle))
        if buttons:
            pressed = sorted((sdl_button.get(c, -1), c) for c in buttons)
            print("\n  Buttons pressed (use these indices for reverse / handbrake):")
            for idx, code in pressed:
                print(f"    button {idx}   (evdev code {code})")
        return

    print("  Suggested wheel_config.ini mapping (dominant axis per phase):")
    label = {"STEERING": "steering_wheel", "THROTTLE": "throttle",
             "BRAKE": "brake", "CLUTCH": "clutch"}
    for name, _ in phases:
        cands = [(c, seen[(name, c)]) for c in linear if (name, c) in seen]
        if not cands:
            print(f"    {label[name]:15s} = ?   (nothing moved)")
            continue
        code, (lo, hi, _n) = max(cands, key=lambda kv: kv[1][1] - kv[1][0])
        print(f"    {label[name]:15s} = {sdl_index[code]}   ({ABS_NAMES[code]}, observed {lo}..{hi})")

    if buttons:
        pressed = sorted((sdl_button.get(c, -1), c) for c in buttons)
        print("\n  Buttons pressed during the run (use these indices for "
              "reverse / handbrake):")
        for idx, code in pressed:
            print(f"    button {idx}   (evdev code {code})")


if __name__ == "__main__":
    main()
