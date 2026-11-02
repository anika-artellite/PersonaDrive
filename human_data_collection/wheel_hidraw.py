#!/usr/bin/env python3
"""
wheel_hidraw.py - read the Logitech G923 directly from /dev/hidraw*.

Why this exists: under WSL2 the kernel has no hid-logitech-hidpp, so hid-generic parses
the G923's report descriptor, logs "ignoring exceeding usage max", and builds an evdev
node whose axes are wired to byte offsets that never change. pygame therefore sees a
joystick with six permanently-dead axes. The actual axis data is present in every 64-byte
HID report as four little-endian 16-bit fields, which this module decodes directly.

Resolution is 16-bit here versus the 8-bit axes hid-generic would have exposed, so this
path is more precise than the "working" one, not a downgrade.

This file targets the Logitech G923 (046d:c267). For a different wheel, change
VENDOR_ID / PRODUCT_ID and re-derive the byte offsets with the scripts in tools/:
map_wheel_bytes.py finds which bytes move, decode_wheel.py confirms they are 16-bit
little-endian pairs and reports the resting value to use as STEER_CENTER.

Layout (confirmed empirically - see tools/decode_wheel.py):
    bytes 43,44  steering   0..65535, centered ~32613
    bytes 45,46  throttle   65535 released -> 0 fully pressed
    bytes 47,48  brake      65535 released -> 0 fully pressed
    bytes 49,50  clutch     65535 released -> 0 fully pressed

Verify the mapping and discover button bits:
    python3 wheel_hidraw.py --verify 40     # confirm throttle/brake are not swapped
    python3 wheel_hidraw.py --buttons 45    # list every button's byte and bit
    python3 wheel_hidraw.py --assign        # record reverse + handbrake by pressing them
"""

import errno
import glob
import os
import select
import struct
import sys
import time

REPORT_SIZE = 64
VENDOR_ID = "046D"
PRODUCT_ID = "C267"

AXES = {
    "steering": 43,
    "throttle": 45,
    "brake": 47,
    "clutch": 49,
}

STEER_CENTER = 32613
AXIS_MAX = 65535

# Bytes 43..50 are the axes; everything else in the report is buttons//hat/state.
AXIS_BYTES = set()
for _o in AXES.values():
    AXIS_BYTES.update((_o, _o + 1))


class WheelNotFound(Exception):
    pass


class G923Reader:
    """Non-blocking reader for the G923's raw HID stream.

    Call poll() every frame; it drains pending reports and keeps the latest decoded
    state. Values are normalized to CARLA conventions:
        steer    -1.0 (full left) .. +1.0 (full right)
        throttle  0.0 (released)  .. 1.0 (fully pressed)
        brake     0.0 (released)  .. 1.0 (fully pressed)
        clutch    0.0 (released)  .. 1.0 (fully pressed)
    """

    def __init__(self, device=None, steer_center=STEER_CENTER, deadzone=0.01):
        self.device = device or self._find_device()
        self.steer_center = steer_center
        self.deadzone = deadzone
        self._fd = None
        self._raw = None
        self.lost = False
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.raw_steer = steer_center
        self.raw_throttle = AXIS_MAX
        self.raw_brake = AXIS_MAX
        self.raw_clutch = AXIS_MAX
        self.buttons = b""
        self.open()

    @staticmethod
    def _find_device():
        """Locate the G923's input interface via sysfs.

        Identify by VID/PID rather than probing for traffic: the wheel only emits a
        report when something moves, so an idle wheel looks identical to an absent one.
        Of its three HID interfaces only the first carries input, distinguished by
        having an input/ directory (equivalently HID_PHYS ending in /input0). Node
        numbering is not stable across re-attach, hence the lookup.
        """
        candidates = []
        for sysfs in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
            uevent = os.path.join(sysfs, "device", "uevent")
            try:
                with open(uevent) as f:
                    fields = dict(
                        line.strip().split("=", 1)
                        for line in f if "=" in line
                    )
            except OSError:
                continue
            hid_id = fields.get("HID_ID", "").upper()
            if not (VENDOR_ID in hid_id and PRODUCT_ID in hid_id):
                continue
            node = "/dev/" + os.path.basename(sysfs)
            is_input0 = (os.path.isdir(os.path.join(sysfs, "device", "input"))
                         or fields.get("HID_PHYS", "").endswith("/input0"))
            candidates.append((node, is_input0, fields.get("HID_NAME", "")))

        if not candidates:
            raise WheelNotFound(
                f"No HID device matching {VENDOR_ID}:{PRODUCT_ID} found. Check the "
                "wheel is plugged in and powered (it spins to full lock on power-on). "
                "Under WSL2 it must also be forwarded into the VM: "
                "usbipd attach --wsl --busid <id>."
            )
        for node, is_input0, _name in candidates:
            if is_input0:
                if not os.access(node, os.R_OK):
                    raise WheelNotFound(
                        f"{node} is not readable — run: sudo chmod 666 /dev/hidraw*"
                    )
                return node
        raise WheelNotFound(
            "Found the wheel but not its input interface; interfaces seen: "
            + ", ".join(n for n, _i, _s in candidates)
        )

    def open(self):
        try:
            self._fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError as e:
            raise WheelNotFound(
                f"{self.device} is not readable — run: sudo chmod 666 /dev/hidraw*"
            ) from e

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    @staticmethod
    def _u16(data, off):
        return data[off] | (data[off + 1] << 8)

    def poll(self):
        """Drain pending reports, decode the newest. Returns True if state changed."""
        if self._fd is None:
            return False
        newest = None
        while True:
            r, _, _ = select.select([self._fd], [], [], 0)
            if not r:
                break
            try:
                data = os.read(self._fd, 512)
            except BlockingIOError:
                break
            except OSError as e:
                # The device went away mid-session (unplugged, or usbipd detached).
                # Flag it: callers hold the last decoded values between reports, so a
                # dead device would otherwise freeze the controls at whatever was last
                # applied - conceivably full throttle.
                if e.errno in (errno.ENODEV, errno.EIO, errno.ENXIO):
                    self.lost = True
                break
            if not data or len(data) < REPORT_SIZE:
                break
            newest = data
        if newest is None:
            return False

        self._raw = newest
        raw_steer = self._u16(newest, AXES["steering"])
        # Keep the raw counts too - normalized values cannot distinguish a healthy
        # sweep from one whose electrical range is reached early in pedal travel.
        self.raw_steer = raw_steer
        self.raw_throttle = self._u16(newest, AXES["throttle"])
        self.raw_brake = self._u16(newest, AXES["brake"])
        self.raw_clutch = self._u16(newest, AXES["clutch"])
        # Center is not exactly mid-scale, so scale each side against its own span.
        if raw_steer >= self.steer_center:
            span = AXIS_MAX - self.steer_center
            self.steer = (raw_steer - self.steer_center) / span if span else 0.0
        else:
            span = self.steer_center
            self.steer = (raw_steer - self.steer_center) / span if span else 0.0
        if abs(self.steer) < self.deadzone:
            self.steer = 0.0

        # Pedals are inverted: 65535 released, 0 fully pressed.
        self.throttle = 1.0 - self._u16(newest, AXES["throttle"]) / AXIS_MAX
        self.brake = 1.0 - self._u16(newest, AXES["brake"]) / AXIS_MAX
        self.clutch = 1.0 - self._u16(newest, AXES["clutch"]) / AXIS_MAX

        self.buttons = bytes(b for i, b in enumerate(newest) if i not in AXIS_BYTES)
        return True

    def button_bytes(self):
        """Raw report with axis bytes masked out - used for button discovery."""
        return self._raw


def verify(seconds):
    """Confirm the axis mapping and surface which bytes/bits the buttons use."""
    try:
        reader = G923Reader()
    except WheelNotFound as e:
        sys.exit(f"ERROR: {e}")

    print(f"Reading {reader.device} for {seconds:.0f}s.\n")
    print("  1. Press ONLY the THROTTLE (right pedal), a few times.")
    print("  2. Then press the buttons you want for REVERSE and HANDBRAKE.")
    print("  3. Leave the wheel and other pedals alone.\n", flush=True)

    peak = {"steering": 0.0, "throttle": 0.0, "brake": 0.0, "clutch": 0.0}
    base_report = None
    button_changes = {}

    # Don't start the clock until the first report lands. An idle wheel sends nothing,
    # so a window opened before the user is ready would otherwise expire empty.
    idle_limit = time.time() + 300.0
    while not reader.poll():
        if time.time() > idle_limit:
            reader.close()
            sys.exit("ERROR: no input for 5 minutes — is the wheel still attached?")
        time.sleep(0.005)
    print(f"  [first input seen — recording for {seconds:.0f}s]\n", flush=True)

    deadline = time.time() + seconds
    while time.time() < deadline:
        if not reader.poll():
            time.sleep(0.002)
            continue
        peak["steering"] = max(peak["steering"], abs(reader.steer))
        peak["throttle"] = max(peak["throttle"], reader.throttle)
        peak["brake"] = max(peak["brake"], reader.brake)
        peak["clutch"] = max(peak["clutch"], reader.clutch)

        raw = reader.button_bytes()
        if base_report is None:
            base_report = raw
            continue
        for i in range(min(len(raw), len(base_report))):
            if i in AXIS_BYTES:
                continue
            if raw[i] != base_report[i]:
                # Track every distinct value, not just the extremes: two buttons in
                # one byte are two separate bits, and min/max would collapse them.
                button_changes.setdefault(i, {base_report[i]}).add(raw[i])

    reader.close()

    print("=" * 62)
    print("  PEAK VALUES  (throttle should be the only large one)")
    print("=" * 62)
    for name, v in peak.items():
        bar = "#" * int(30 * min(v, 1.0))
        flag = "  <-- MOVED" if v > 0.2 else ""
        print(f"  {name:9s} {v:5.3f}  [{bar:<30}]{flag}")

    print()
    if peak["throttle"] > 0.5 and peak["brake"] < 0.2:
        print("  CONFIRMED: bytes[45,46] are the throttle. Mapping is correct.")
    elif peak["brake"] > 0.5 and peak["throttle"] < 0.2:
        print("  MISMAPPED: pressing throttle moved bytes[47,48].")
        print("  -> throttle and brake are swapped; swap them in AXES.")
    else:
        print("  INCONCLUSIVE — no clear single pedal movement. Re-run and press")
        print("  only the throttle.")

    print("\n" + "=" * 62)
    print("  BUTTON BYTES")
    print("=" * 62)
    if not button_changes:
        print("  No non-axis byte changed — no buttons were pressed.")
    else:
        for off in sorted(button_changes):
            values = sorted(button_changes[off])
            rest = values[0]
            print(f"    byte[{off:2d}]  resting={rest} ({rest:08b})")
            # Every bit that was ever set beyond the resting value is one button.
            bits = 0
            for v in values:
                bits |= v ^ rest
            for bit in range(8):
                mask = 1 << bit
                if bits & mask:
                    print(f"        bit {bit} (mask 0x{mask:02X})  -> a button")
            print(f"        distinct values seen: {values}")


def scan_buttons(seconds):
    """Build a complete button map: press every button, get every bit."""
    try:
        reader = G923Reader()
    except WheelNotFound as e:
        sys.exit(f"ERROR: {e}")

    print(f"Reading {reader.device}.\n")
    print("  Press EVERY button you might want to use, one at a time.")
    print("  Include the paddles, the D-pad, and anything on the wheel face.")
    print("  Do not touch the pedals or turn the wheel.\n", flush=True)

    idle_limit = time.time() + 300.0
    while not reader.poll():
        if time.time() > idle_limit:
            reader.close()
            sys.exit("ERROR: no input for 5 minutes — is the wheel still attached?")
        time.sleep(0.005)
    print(f"  [first input seen — recording for {seconds:.0f}s]\n", flush=True)

    base = reader.button_bytes()
    values = {}          # offset -> set of distinct values
    order = []           # (offset, mask) in the order first seen

    deadline = time.time() + seconds
    while time.time() < deadline:
        if not reader.poll():
            time.sleep(0.002)
            continue
        raw = reader.button_bytes()
        for i in range(min(len(raw), len(base))):
            if i in AXIS_BYTES or raw[i] == base[i]:
                continue
            values.setdefault(i, {base[i]}).add(raw[i])
            for bit in range(8):
                mask = 1 << bit
                if (raw[i] ^ base[i]) & mask and (i, mask) not in order:
                    order.append((i, mask))
    reader.close()

    print("=" * 62)
    print("  BUTTON MAP")
    print("=" * 62)
    if not order:
        print("  No button bits detected.")
        return
    for n, (off, mask) in enumerate(order):
        print(f"    [{n}]  byte[{off:2d}] mask 0x{mask:02X}  (bit {mask.bit_length() - 1})")
    print("\n  Raw values per byte:")
    for off in sorted(values):
        print(f"    byte[{off:2d}]: {sorted(values[off])}")
    print("\n  Pick two of the indices above for reverse and handbrake.")


def assign_buttons():
    """Record which physical buttons you want for reverse and handbrake.

    Avoids having to translate byte/mask indices by hand: press the button you want,
    and the byte and bit are captured directly.
    """
    try:
        reader = G923Reader()
    except WheelNotFound as e:
        sys.exit(f"ERROR: {e}")

    picks = {}
    for role in ("reverse", "handbrake"):
        print("=" * 62)
        print(f"  Press and RELEASE the button you want for {role.upper()}.")
        print("=" * 62, flush=True)

        # Settle: wait for everything to be released before sampling.
        base = None
        settle = time.time() + 1.0
        while time.time() < settle:
            if reader.poll():
                base = reader.button_bytes()
            time.sleep(0.005)
        while base is None:
            if reader.poll():
                base = reader.button_bytes()
            time.sleep(0.005)

        found = None
        idle_limit = time.time() + 120.0
        while found is None:
            if time.time() > idle_limit:
                reader.close()
                sys.exit(f"ERROR: no button pressed for {role} within 2 minutes.")
            if not reader.poll():
                time.sleep(0.003)
                continue
            raw = reader.button_bytes()
            for i in range(min(len(raw), len(base))):
                if i in AXIS_BYTES or raw[i] == base[i]:
                    continue
                diff = raw[i] ^ base[i]
                for bit in range(8):
                    mask = 1 << bit
                    if diff & mask and raw[i] & mask:
                        found = (i, mask)
                        break
                if found:
                    break

        picks[role] = found
        off, mask = found
        print(f"  -> {role}: byte[{off}] mask 0x{mask:02X}\n")
        # Wait for release so the next prompt does not capture the same press.
        while True:
            reader.poll()
            raw = reader.button_bytes()
            if raw is None or not (raw[off] & mask):
                break
            time.sleep(0.005)
        time.sleep(0.4)

    reader.close()

    print("=" * 62)
    print("  RESULT — add this to wheel_config.ini under [hidraw]")
    print("=" * 62)
    print("\n[hidraw]")
    for role, (off, mask) in picks.items():
        print(f"{role}_byte = {off}")
        print(f"{role}_mask = 0x{mask:02X}")
    return picks


if __name__ == "__main__":
    if "--assign" in sys.argv:
        assign_buttons()
    elif "--verify" in sys.argv:
        i = sys.argv.index("--verify")
        secs = float(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 40.0
        verify(secs)
    elif "--buttons" in sys.argv:
        i = sys.argv.index("--buttons")
        secs = float(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 45.0
        scan_buttons(secs)
    else:
        print(__doc__)
