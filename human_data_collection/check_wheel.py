#!/usr/bin/env python3
"""
check_wheel.py - pre-session sanity check for the wheel, through the real agent code.

Builds the agent's actual KeyboardControl and reports the carla.VehicleControl it
produces while you operate the wheel. This exercises the same path a collection run
uses, so a pass here means the wheel will behave during collect_session.sh.

    python3 check_wheel.py            # waits for input, then samples 25s

Run it before a session if the wheel has been re-attached - the hidraw permissions and
node numbering are recreated on every attach.

Needs the same environment as a collection run: source your env.sh first.
"""

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

HERE = os.path.dirname(os.path.realpath(__file__))

# Same search paths collect_session.sh exports, so this runs standalone. The agent pulls
# in SimLingo's DataAgent, which needs srunner and the CARLA PythonAPI agents package.
# Each tree is bundled in this package, so nothing has to be set; these env vars only
# override the bundled copy, exactly as the shell scripts treat them.
os.environ.setdefault("LEADERBOARD_ROOT", os.path.join(HERE, "leaderboard"))
os.environ.setdefault("SCENARIO_RUNNER_ROOT", os.path.join(HERE, "scenario_runner"))
os.environ.setdefault("SIMLINGO_ROOT", os.path.join(HERE, "simlingo"))
os.environ.setdefault("CARLA_PYTHONAPI", os.path.join(HERE, "PythonAPI", "carla"))

_missing = [v for v in ("LEADERBOARD_ROOT", "SCENARIO_RUNNER_ROOT", "CARLA_PYTHONAPI")
            if not os.path.isdir(os.path.expanduser(os.environ[v]))]
if _missing:
    sys.exit("Not a directory: " + ", ".join(f"{v}={os.environ[v]}" for v in _missing) +
             "\nUnset these to use the copies bundled in this package.")

for _p in (HERE,
           os.environ["LEADERBOARD_ROOT"],
           os.environ["SCENARIO_RUNNER_ROOT"],
           os.environ["CARLA_PYTHONAPI"]):
    _p = os.path.abspath(os.path.expanduser(_p))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pygame  # noqa: E402

pygame.init()
pygame.display.set_mode((320, 240))

from human_agent import KeyboardControl  # noqa: E402


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0

    control = KeyboardControl()
    if control._wheel is None:
        print("\nNo hidraw wheel in use.")
        if control._joystick is None:
            print("Keyboard control is active (W,A,S,D). The wheel is NOT usable.")
        else:
            print("Falling back to the pygame joystick.")
        return 1

    print(f"\nOperate the wheel: steer both ways, then each pedal.")
    print("Sampling starts on your first input.\n", flush=True)

    while not control._wheel.poll():
        time.sleep(0.005)
    print(f"  [input detected — sampling for {seconds:.0f}s]\n", flush=True)

    peak = {"steer_l": 0.0, "steer_r": 0.0, "throttle": 0.0, "brake": 0.0}
    both = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        control.parse_events(16)
        c = control._control
        peak["steer_l"] = min(peak["steer_l"], c.steer)
        peak["steer_r"] = max(peak["steer_r"], c.steer)
        peak["throttle"] = max(peak["throttle"], c.throttle)
        peak["brake"] = max(peak["brake"], c.brake)
        if c.throttle > 0.1 and c.brake > 0.1:
            both += 1
        time.sleep(0.008)

    print("=" * 58)
    print("  PEAK VehicleControl VALUES")
    print("=" * 58)
    print(f"  steer  left  : {peak['steer_l']:+.3f}")
    print(f"  steer  right : {peak['steer_r']:+.3f}")
    print(f"  throttle     :  {peak['throttle']:.3f}")
    print(f"  brake        :  {peak['brake']:.3f}")

    ok = True
    print()
    if peak["steer_l"] > -0.5 or peak["steer_r"] < 0.5:
        print("  WARNING: steering did not reach both extremes.")
        ok = False
    if peak["throttle"] < 0.5:
        print("  WARNING: throttle never exceeded 0.5.")
        ok = False
    if peak["brake"] < 0.5:
        print("  WARNING: brake never exceeded 0.5.")
        ok = False
    if both:
        print(f"  ERROR: throttle and brake were both applied on {both} frames.")
        ok = False

    print("  READY — the wheel drives the agent correctly." if ok
          else "  Re-run and exercise every control fully if the warnings look wrong.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
