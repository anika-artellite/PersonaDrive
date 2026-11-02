#!/usr/bin/env python3
"""
preview_route.py  --route <route.xml>  --town <TownXX>

Connects to the already-running CARLA server, loads the correct map if needed, draws
the route waypoints and scenario trigger points using the CARLA debug API, positions
the spectator in a bird's-eye view above the route, and waits for Enter before
returning control to collect_session.sh.

The server address comes from $CARLA_HOST / $CARLA_PORT (default 127.0.0.1:2000).
If the connection fails for any reason the script exits cleanly, so a failed preview
never blocks driving.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import carla
except ImportError:
    print("[preview] CARLA Python API not on PYTHONPATH — skipping route preview.")
    sys.exit(0)

# Default to 127.0.0.1 rather than "localhost": under WSL2, "localhost" resolves to ::1
# first, and a CARLA server running on the Windows host listens only on IPv4, so the
# client times out waiting for a simulator that is in fact running.
HOST    = os.environ.get("CARLA_HOST", "127.0.0.1")
PORT    = int(os.environ.get("CARLA_PORT", "2000"))
# Loading a large map is the slow path here, not the RPC. Town12 and Town13 routinely
# take well over a minute to load, and a short timeout makes the preview fail on exactly
# the towns this dataset uses - the map then finishes loading anyway, so the failure is
# silent and misleading. $CARLA_TIMEOUT overrides.
TIMEOUT = float(os.environ.get("CARLA_TIMEOUT", "180.0"))
LIFE    = 120.0   # seconds debug drawings persist in the CARLA world


# ---------------------------------------------------------------------------
# XML parsing (no srunner dependency)
# ---------------------------------------------------------------------------

def parse_route_xml(xml_path: Path):
    """Return (waypoints, scenarios) from a route XML file.

    waypoints : list of carla.Location
    scenarios : list of {'type': str, 'loc': carla.Location}
    """
    tree = ET.parse(xml_path)
    waypoints = []
    scenarios = []

    for route in tree.iter('route'):
        for pos in route.find('waypoints').iter('position'):
            waypoints.append(carla.Location(
                x=float(pos.attrib['x']),
                y=float(pos.attrib['y']),
                z=float(pos.attrib['z']),
            ))

        scenarios_elem = route.find('scenarios')
        if scenarios_elem is not None:
            for sc in scenarios_elem.iter('scenario'):
                tp = sc.find('trigger_point')
                if tp is not None:
                    scenarios.append({
                        'type': sc.attrib.get('type', ''),
                        'loc': carla.Location(
                            x=float(tp.attrib['x']),
                            y=float(tp.attrib['y']),
                            z=float(tp.attrib.get('z', '0')),
                        ),
                    })

    return waypoints, scenarios


# ---------------------------------------------------------------------------
# Debug drawing
# ---------------------------------------------------------------------------

def _lerp_color(t: float) -> carla.Color:
    """Green at t=0, red at t=1."""
    return carla.Color(r=int(255 * t), g=int(255 * (1.0 - t)), b=30)


def draw_route(debug, waypoints, scenarios):
    n = len(waypoints)

    # Arrows along the route, colour-coded green -> red
    for i in range(n - 1):
        t = i / max(n - 2, 1)
        color = _lerp_color(t)
        start = waypoints[i]     + carla.Location(z=0.5)
        end   = waypoints[i + 1] + carla.Location(z=0.5)
        debug.draw_arrow(start, end,
                         thickness=0.15, arrow_size=0.3,
                         color=color, life_time=LIFE)

    # Intermediate keypoints (small dots every ~5 waypoints)
    for i in range(1, n - 1, max(1, n // 20)):
        debug.draw_point(waypoints[i] + carla.Location(z=0.5),
                         size=0.1, color=carla.Color(200, 200, 200),
                         life_time=LIFE)

    # Start marker
    debug.draw_point(waypoints[0] + carla.Location(z=0.5),
                     size=0.3, color=carla.Color(0, 255, 100), life_time=LIFE)
    debug.draw_string(waypoints[0] + carla.Location(z=2.5),
                      'START', color=carla.Color(0, 255, 100), life_time=LIFE)

    # End marker
    debug.draw_point(waypoints[-1] + carla.Location(z=0.5),
                     size=0.3, color=carla.Color(255, 50, 50), life_time=LIFE)
    debug.draw_string(waypoints[-1] + carla.Location(z=2.5),
                      'END', color=carla.Color(255, 50, 50), life_time=LIFE)

    # Scenario trigger points (yellow)
    for sc in scenarios:
        loc = sc['loc'] + carla.Location(z=0.5)
        debug.draw_point(loc, size=0.35,
                         color=carla.Color(255, 220, 0), life_time=LIFE)
        debug.draw_string(loc + carla.Location(z=2.0),
                          sc['type'], color=carla.Color(255, 220, 0),
                          life_time=LIFE)


# ---------------------------------------------------------------------------
# Spectator positioning
# ---------------------------------------------------------------------------

def position_spectator(world, waypoints):
    """Move spectator to a bird's-eye view centred above the route."""
    xs = [wp.x for wp in waypoints]
    ys = [wp.y for wp in waypoints]
    zs = [wp.z for wp in waypoints]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0

    # Scale height above terrain to route's horizontal extent
    extent = max(max(xs) - min(xs), max(ys) - min(ys), 10.0)
    height_above_terrain = min(max(extent * 0.9, 40.0), 250.0)

    # Add terrain elevation so the spectator camera does not end up underground.
    # Town12 (and others) sit at z ~ 370m above sea level.
    spectator_z = max(zs) + height_above_terrain

    world.get_spectator().set_transform(carla.Transform(
        carla.Location(x=cx, y=cy, z=spectator_z),
        carla.Rotation(pitch=-89.0, yaw=0.0, roll=0.0),
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CARLA debug-API route preview before driving."
    )
    parser.add_argument('--route', required=True, help='Path to route XML file')
    parser.add_argument('--town',  default='',   help='Town name (e.g. Town01)')
    args = parser.parse_args()

    xml_path   = Path(args.route)
    route_name = xml_path.stem
    town       = args.town

    # Parse XML
    try:
        waypoints, scenarios = parse_route_xml(xml_path)
    except Exception as e:
        print(f"[preview] Failed to parse route XML: {e} — skipping preview.")
        sys.exit(0)

    if len(waypoints) < 2:
        print("[preview] Too few waypoints — skipping preview.")
        sys.exit(0)

    # Connect to CARLA
    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(TIMEOUT)
        world = client.get_world()
    except Exception as e:
        print(f"[preview] Could not connect to CARLA at {HOST}:{PORT} ({e}) — skipping preview.")
        sys.exit(0)

    try:
        current_map = world.get_map().name.split('/')[-1]   # e.g. "Town01"

        # Load the correct map if needed
        if town and town not in current_map:
            print(f"[preview] Loading map {town} (current: {current_map})...")
            world = client.load_world(town)
            time.sleep(2.0)   # let world settle before drawing
            world = client.get_world()

        draw_route(world.debug, waypoints, scenarios)
        position_spectator(world, waypoints)

        print(f"\n[ROUTE PREVIEW]  {route_name}  |  {town or current_map}")
        print(f"  Waypoints : {len(waypoints)}")
        if scenarios:
            print(f"  Scenarios : {', '.join(s['type'] for s in scenarios)}")
        print(f"\n  Route is drawn in the CARLA spectator window.")
        print(f"  (Green = start, Red = end, Yellow = scenario trigger)")
        print(f"  Debug markers persist for {int(LIFE)}s.\n")
        input("  Press Enter to start driving... ")
        print()

    except KeyboardInterrupt:
        print("\n[preview] Skipped.")
    except Exception as e:
        print(f"[preview] WARNING: preview failed ({e}) — continuing to drive.")

    sys.exit(0)


if __name__ == '__main__':
    main()
