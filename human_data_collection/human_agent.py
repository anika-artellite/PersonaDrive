#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
PersonaDrive human driving agent for the CARLA Leaderboard 2.0.

A child of SimLingo's DataAgent that replaces the autopilot's control output with a
human's - read from a Logitech force-feedback wheel and pedal set, or from the keyboard
when no wheel is present - while keeping the full sensor-recording pipeline (RGB, LiDAR,
bounding boxes, BEV) untouched. The autopilot still runs each frame in "plant" mode, so
every frame also carries the expert's privileged state alongside the human's actions.

Run it through the leaderboard evaluator, not directly:

    python3 leaderboard/leaderboard_evaluator.py --agent /path/to/human_agent.py ...

The SimLingo code this depends on is bundled in simlingo/; see README.md.
"""

import os
import sys
import datetime
import carla
import json
import math
import time
import numpy as np

# Pygame for visual rendering and keyboard control
try:
    import pygame
    from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_UP, K_a, K_d, K_s, K_w, K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

# This file is passed to leaderboard_evaluator.py as --agent, which puts the directory
# holding it on sys.path. wheel_hidraw.py therefore imports as a sibling, wherever the
# package is unpacked.
current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# The sensor stack (cameras, LiDAR, bounding boxes, BEV) is SimLingo's DataAgent; this
# agent only replaces the control source. The SimLingo code it needs is bundled in
# simlingo/ next to this file, so no external checkout is required. Set SIMLINGO_ROOT
# to use your own checkout instead of the bundled copy.
SIMLINGO_ROOT = os.path.abspath(os.path.expanduser(
    os.environ.get('SIMLINGO_ROOT') or os.path.join(current_dir, 'simlingo')))
simlingo_team_code_dir = os.path.join(SIMLINGO_ROOT, 'team_code')
simlingo_leaderboard_dir = os.path.join(SIMLINGO_ROOT, 'leaderboard')
if not os.path.isdir(simlingo_team_code_dir):
    raise RuntimeError(
        f'No team_code/ under {SIMLINGO_ROOT}. Leave SIMLINGO_ROOT unset to use the '
        'bundled copy, or point it at a SimLingo checkout.')

if simlingo_team_code_dir not in sys.path:
    sys.path.append(simlingo_team_code_dir)
if simlingo_leaderboard_dir not in sys.path:
    sys.path.append(simlingo_leaderboard_dir)

# The CARLA_Leaderboard_20 'leaderboard.autoagents' module is already loaded in memory.
# To allow importing 'autonomous_agent_local' from SimLingo's split leaderboard package,
# we need to dynamically extend the __path__ cache that importlib searches.
import leaderboard.autoagents
simlingo_autoagents_dir = os.path.join(simlingo_leaderboard_dir, 'leaderboard', 'autoagents')
if hasattr(leaderboard.autoagents, '__path__') and simlingo_autoagents_dir not in leaderboard.autoagents.__path__:
    leaderboard.autoagents.__path__.append(simlingo_autoagents_dir)

from data_agent import DataAgent

# Under WSL2 the kernel has no hid-logitech-hidpp, so hid-generic misparses the G923's
# report descriptor and builds an evdev node whose axes never move - pygame sees a
# joystick with six dead axes. wheel_hidraw reads the raw HID reports instead, where the
# axes are present as 16-bit fields. Optional: absent or unreadable, we fall back to the
# pygame joystick path and then to the keyboard, exactly as before.
try:
    from wheel_hidraw import G923Reader, WheelNotFound
except ImportError:
    G923Reader = None
    WheelNotFound = Exception


def get_entry_point():
    return 'HumanAgent'


class HumanInterface(object):
    """
    Class to visualize the camera feed to the human player.
    Modeled closely after human_agent_2.py for a simple 1x display experience.
    """
    def __init__(self, width, height):
        self._width = width
        self._height = height
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("PersonaDrive — human data collection")

        # Window capture. Saves the rendered surface INCLUDING the speed/command overlay,
        # which the sensor jpgs under rgb/ do not have. Driving is never interrupted:
        # capture is a surface blit to disk between frames.
        #   CARLA_SCREENSHOT_DIR    where to write (enables capture)
        #   CARLA_SCREENSHOT_EVERY  save every Nth frame; 0/unset = manual only
        # Press P while driving for a one-off shot.
        self._shot_dir = os.environ.get('CARLA_SCREENSHOT_DIR')
        self._shot_every = int(os.environ.get('CARLA_SCREENSHOT_EVERY', '0') or 0)
        self._shot_frame = 0
        self._shot_count = 0
        self._shot_key_was_down = False
        if self._shot_dir:
            os.makedirs(self._shot_dir, exist_ok=True)
            how = f"every {self._shot_every} frames" if self._shot_every else "on P keypress"
            print(f"Screenshots: {self._shot_dir}  ({how}; P captures anytime)")

    def run_interface(self, input_data, speed_mph=None, command_val=None, next_command_val=None, active_scenario=None):
        """
        Run the GUI displaying the 'rgb' forward-facing sensor.
        """
        self._display.fill((30, 30, 30))
        
        # Center (Main RGB)
        if 'rgb' in input_data:
            # DataAgent sensors output BGR images by default in shape [H, W, 4]
            # Convert to RGB to render using Pygame
            image_center = input_data['rgb'][1][:, :, -2::-1]
            surface_center = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
            surface_center = pygame.transform.smoothscale(surface_center, (self._width, self._height))
            self._display.blit(surface_center, (0, 0))

        if speed_mph is not None:
            font = pygame.font.SysFont('arial', 48, bold=True)
            speed_text = font.render(f"{speed_mph:.0f} MPH", True, (0, 255, 0))
            text_rect = speed_text.get_rect(bottomright=(self._width - 20, self._height - 20))
            self._display.blit(speed_text, text_rect)

        cmd_map = {1: "TURN LEFT", 2: "TURN RIGHT", 3: "GO STRAIGHT", 4: "FOLLOW LANE", 5: "CHANGE LEFT", 6: "CHANGE RIGHT"}

        y_offset = 20
        if command_val is not None:
            try:
                cmd_idx = int(getattr(command_val, 'value', command_val))
            except Exception:
                cmd_idx = 4
            cmd_str = cmd_map.get(cmd_idx, "FOLLOW LANE")
            
            font = pygame.font.SysFont('arial', 44, bold=True)
            cmd_text = font.render(f"NOW: {cmd_str}", True, (255, 255, 0))
            cmd_rect = cmd_text.get_rect(topright=(self._width - 20, y_offset))
            self._display.blit(cmd_text, cmd_rect)
            y_offset += 52

        if next_command_val is not None:
            try:
                next_cmd_idx = int(getattr(next_command_val, 'value', next_command_val))
            except Exception:
                next_cmd_idx = 4
            next_cmd_str = cmd_map.get(next_cmd_idx, "FOLLOW LANE")

            font = pygame.font.SysFont('arial', 36)
            next_text = font.render(f"NEXT: {next_cmd_str}", True, (200, 200, 100))
            next_rect = next_text.get_rect(topright=(self._width - 20, y_offset))
            self._display.blit(next_text, next_rect)
            y_offset += 46
            
        if active_scenario is not None:
            font = pygame.font.SysFont('arial', 36, bold=True)
            scen_text = font.render(f"Scenario: {active_scenario}", True, (255, 100, 100))
            scen_rect = scen_text.get_rect(topright=(self._width - 20, y_offset))
            self._display.blit(scen_text, scen_rect)

        self._maybe_capture()
        pygame.display.flip()

    def _maybe_capture(self):
        """Save the rendered window if periodic capture is due, or P was just pressed.

        Uses key.get_pressed() rather than the event queue: KeyboardControl drains events
        with pygame.event.get(), so a KEYDOWN for P would be consumed before we saw it.
        Edge-detected so holding P does not write a burst of identical frames.
        """
        if not self._shot_dir:
            return
        self._shot_frame += 1

        want = bool(self._shot_every) and (self._shot_frame % self._shot_every == 0)
        try:
            down = pygame.key.get_pressed()[pygame.K_p]
        except Exception:
            down = False
        if down and not self._shot_key_was_down:
            want = True
        self._shot_key_was_down = down

        if not want:
            return
        path = os.path.join(self._shot_dir, f"shot_{self._shot_count:05d}.png")
        try:
            pygame.image.save(self._display, path)
            self._shot_count += 1
        except Exception as e:
            print(f"screenshot failed: {e}")

    def set_black_screen(self):
        """Set the surface to black"""
        black_array = np.zeros([self._width, self._height, 3])
        self._surface = pygame.surfarray.make_surface(black_array.swapaxes(0,1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()


class KeyboardControl(object):
    """
    Keyboard and Logitech Wheel control for the human agent.
    Returns standard CARLA VehicleControl objects.
    """
    def __init__(self):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

        # Set CARLA_WHEEL_TRACE to a path to log every frame's raw and decoded pedal
        # values. Off unless asked for - this writes a line per frame.
        self._trace = None
        trace_path = os.environ.get('CARLA_WHEEL_TRACE')
        if trace_path:
            self._trace = open(trace_path, 'a', buffering=1)
            self._trace.write("# time,raw_steer,raw_throttle,raw_brake,raw_clutch,"
                              "steer_cmd,throttle_cmd,brake_cmd\n")
            print(f"Wheel: tracing pedal input to {trace_path}")

        # Preferred input: raw HID. Gives 16-bit axes where the evdev path gives none.
        self._wheel = None
        self._reverse_was_down = False
        self._hidraw_reverse = (5, 0x20)      # byte, mask - see wheel_hidraw --assign
        self._hidraw_handbrake = (5, 0x40)
        if G923Reader is not None:
            try:
                self._wheel = G923Reader()
                print(f"Wheel: raw HID via {self._wheel.device} (16-bit axes).")
            except WheelNotFound as e:
                print(f"Wheel: hidraw unavailable ({e});")
                print("       falling back to pygame joystick.")

        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            print("WARNING: More than one joystick connected. Taking the first one.")

        self._joystick = None
        if joystick_count >= 1 and self._wheel is None:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()

        elif joystick_count >= 1:
            print("Wheel: pygame joystick present but hidraw is in use; ignoring it.")

        if self._joystick is not None:

            self._parser = ConfigParser()
            config_path = os.path.join(current_dir, 'wheel_config.ini')
            if not os.path.exists(config_path):
                config_path = 'wheel_config.ini'
                
            if os.path.exists(config_path):
                self._parser.read(config_path)
                try:
                    # One section per wheel model, named after the device. The first
                    # section is used, so swapping wheels means editing the ini only.
                    section = self._parser.sections()[0]
                    self._steer_idx = int(self._parser.get(section, 'steering_wheel'))
                    self._throttle_idx = int(self._parser.get(section, 'throttle'))
                    self._brake_idx = int(self._parser.get(section, 'brake'))
                    self._reverse_idx = int(self._parser.get(section, 'reverse'))
                    self._handbrake_idx = int(self._parser.get(section, 'handbrake'))
                except Exception as e:
                    print(f"Failed to read specific wheel mapping: {e}")

            # Sanity-check the mapping before trusting it. If the kernel misparsed the
            # HID descriptor the axes sit at values that are never a real rest state,
            # and this is worse than having no wheel: a dead axis reads 0.0, which
            # through the pedal curve is ~0.62, so the car pulls away with 62% throttle
            # and 62% brake held permanently. No foot rests on both pedals at once.
            if hasattr(self, '_steer_idx') and not self._joystick_rest_is_sane():
                print("WARNING: joystick reports throttle AND brake applied at rest —")
                print("         the kernel has misparsed the wheel's HID descriptor.")
                print("         Ignoring it; using keyboard control (W,A,S,D).")
                self._joystick = None
        elif self._wheel is None:
            print("No Logitech Wheel recognized. Falling back to Keyboard keys (W,A,S,D / Arrows).")

    def parse_events(self, timestamp_ms):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp_ms)
        if self._wheel is not None:
            self._parse_vehicle_hidraw()
        elif self._joystick is not None and hasattr(self, '_steer_idx'):
            self._parse_vehicle_wheel()
        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 
            elif event.type == pygame.VIDEORESIZE:
                from pygame.display import get_surface, set_mode
                curr_surf = get_surface()
                if curr_surf is not None:
                    set_mode((event.w, event.h), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0
            elif event.type == pygame.JOYBUTTONDOWN:
                if hasattr(self, '_reverse_idx') and event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.8
        else:
            self._control.throttle = 0.0

        steer_increment = 3e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _pedal_curve(axis):
        """The agent's pedal response: +1 released -> 0.0, -1 pressed -> 1.0."""
        cmd = 1.6 + (2.05 * math.log10(-0.7 * axis + 1.4) - 1.2) / 0.92
        return min(max(cmd, 0.0), 1.0)

    def _joystick_rest_is_sane(self):
        """False when the joystick claims throttle and brake are both applied.

        Checks the decoded pedal commands rather than the raw axis values: what matters
        is whether the mapping produces a possible resting state, not what the numbers
        look like. Catches any misparse, not just the all-axes-identical case.
        """
        try:
            for _ in range(10):
                pygame.event.pump()
            axes = [self._joystick.get_axis(i)
                    for i in range(self._joystick.get_numaxes())]
        except pygame.error:
            return True     # display not up yet - don't reject on a failed probe
        if max(self._throttle_idx, self._brake_idx) >= len(axes):
            return False    # config points at axes this device does not have
        throttle = self._pedal_curve(axes[self._throttle_idx])
        brake = self._pedal_curve(axes[self._brake_idx])
        return not (throttle > 0.1 and brake > 0.1)

    def _parse_vehicle_hidraw(self):
        """
        Calculate new vehicle controls from the wheel's raw HID reports.

        The normalized values are mapped back onto the -1..+1 axis convention the
        pygame path uses, so the steering and pedal response curves below stay
        identical to _parse_vehicle_wheel(). Data collected either way is directly
        comparable; rewriting the curves would quietly change the pedal feel.
        """
        # Poll for fresh data, but apply the reader's state either way. The wheel only
        # emits a report when a value CHANGES, so holding a pedal against its stop -
        # or the wheel at a fixed angle - produces no reports at all. Returning early
        # here would leave the control at the 0.0 _parse_vehicle_keys just wrote, which
        # cut the throttle exactly when the pedal was floored and steady.
        self._wheel.poll()

        if self._wheel.lost:
            # Device disappeared mid-route. Do NOT keep applying the last values -
            # they could be full throttle. Drop to the keyboard, which _parse_vehicle_keys
            # has already applied this frame.
            print("ERROR: wheel disconnected mid-session — falling back to keyboard.")
            self._wheel.close()
            self._wheel = None
            return

        steer_axis = self._wheel.steer                      # already -1..+1
        throttle_axis = 1.0 - 2.0 * self._wheel.throttle    # +1 released, -1 pressed
        brake_axis = 1.0 - 2.0 * self._wheel.brake

        K1 = 1.0
        steerCmd = K1 * math.tan(1.1 * steer_axis)

        K2 = 1.6
        throttleCmd = K2 + (2.05 * math.log10(-0.7 * throttle_axis + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0: throttleCmd = 0
        elif throttleCmd > 1: throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(-0.7 * brake_axis + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0: brakeCmd = 0
        elif brakeCmd > 1: brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        # Opt-in trace for diagnosing pedal faults that only appear while driving:
        #   CARLA_WHEEL_TRACE=/tmp/wheel.csv bash collect_session.sh ...
        # Records raw counts alongside the decoded command, so a dropout or a range
        # that saturates early can be told apart after the fact.
        if self._trace is not None:
            self._trace.write(
                f"{time.time():.4f},{self._wheel.raw_steer},{self._wheel.raw_throttle},"
                f"{self._wheel.raw_brake},{self._wheel.raw_clutch},"
                f"{steerCmd:.5f},{throttleCmd:.5f},{brakeCmd:.5f}\n")

        raw = self._wheel.button_bytes()
        if raw is not None:
            # OR with the keyboard state _parse_vehicle_keys just set, so SPACE keeps
            # working as a handbrake. Assigning here instead would silently disable it,
            # and the wheel's handbrake button is not yet confirmed.
            hb_byte, hb_mask = self._hidraw_handbrake
            if hb_byte < len(raw):
                self._control.hand_brake = (self._control.hand_brake
                                            or bool(raw[hb_byte] & hb_mask))

            # Reverse is a toggle. hidraw exposes bit levels rather than pygame's
            # JOYBUTTONDOWN events, so fire on the low-to-high edge only - keying off
            # the level would flip reverse on every frame the button is held.
            rv_byte, rv_mask = self._hidraw_reverse
            if rv_byte < len(raw):
                down = bool(raw[rv_byte] & rv_mask)
                if down and not self._reverse_was_down:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0
                self._reverse_was_down = down

    def _parse_vehicle_wheel(self):
        """
        Calculate new vehicle controls based on input wheel
        """
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        jsButtons = [float(self._joystick.get_button(i)) for i in range(self._joystick.get_numbuttons())]

        K1 = 1.0 
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6 
        throttleCmd = K2 + (2.05 * math.log10(-0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0: throttleCmd = 0
        elif throttleCmd > 1: throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(-0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0: brakeCmd = 0
        elif brakeCmd > 1: brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd
        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])


class HumanAgent(DataAgent):
    """
    Overrides run_step and setup to bypass the AutoPilot algorithms
    and use the Human wheel/keyboard logic, but heavily relies on DataAgent's
    tick and save_sensors to accurately collect all simlingo sensor modalities.
    """
    def __init__(self, *args, **kwargs):
        from leaderboard.autoagents.autonomous_agent_local import Track
        from leaderboard.envs.sensor_interface import SensorInterface
        self.track = Track.MAP
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None

    def _get_route_scenario(self):
        if hasattr(self, '_route_scenario_instance'):
            return self._route_scenario_instance
        
        import gc
        for obj in gc.get_objects():
            if type(obj).__name__ == 'RouteScenario':
                self._route_scenario_instance = obj
                return obj
        
        return None

    def _draw_waypoints(self, waypoints, vertical_shift=0.1, size=0.1, downsample=1, timeout=10000.0):
        """
        Draw a list of waypoints at a certain height to guide human drivers in data collection.
        """
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        client = CarlaDataProvider.get_client()
        world = client.get_world()

        if not waypoints:
            return

        from agents.navigation.local_planner import RoadOption

        for i, w in enumerate(waypoints):
            if i % downsample != 0:
                continue

            wp = w[0].location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT: 
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT: 
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT: 
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT: 
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT: 
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0) 

            world.debug.draw_point(wp, size=size, color=color, life_time=timeout)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=2*size,
                                    color=carla.Color(0, 0, 128), life_time=timeout)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=2*size,
                                    color=carla.Color(128, 128, 128), life_time=timeout)

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        # Initialize all DataAgent setup (directories, models, etc).
        super().setup(path_to_conf_file, route_index, traffic_manager)
        
        # Disable all of Simlingo/AutoPilot's dense visual clutter (3D bounding boxes, intersection lights, hazard markers)
        if hasattr(self, 'config'):
            self.config.debug = False
        self.visualize = 0

        # Init Pygame UI explicitly to 1280x720, unlike human_data_agent which used 3x monitors.
        self._hic = HumanInterface(1280, 720)
        self._controller = KeyboardControl()
        self._prev_timestamp = 0
        self._pygame_clock = pygame.time.Clock()

        # 3D route markers are disabled - they appear in the RGB camera image.

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        """
        Bypasses super().run_step (which applies AutoPilot controllers)
        and instead routes control to Human input.
        """
        self.step += 1
        self.step_tmp += 1
        
        self._pygame_clock.tick_busy_loop(20)

        # Ensure DataAgent map and managers are initialized
        if not getattr(self, 'initialized', False):
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            client = CarlaDataProvider.get_client()
            world_map = client.get_world().get_map()
            self._init(world_map)

        # Convert LiDAR into ego coordinate (Logic inherited from DataAgent structure)
        import transfuser_utils as t_u
        if 'lidar' in input_data:
            input_data['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

        if getattr(self, 'datagen', False):
            self.augment_camera(sensors)

        # Process exactly the same DataAgent tick logic (Bounding boxes, LiDAR merging, BEV generation)
        tick_data = self.tick(input_data)
        tick_data.update(self.tick_autopilot(input_data))

        # Let AutoPilot compute its full privileged route planning, hazard checking, and aim_wp
        # so that our dataset contains perfectly dense waypoint/expert prediction information
        auto_control, driving_data = self._get_control(input_data, plant=True)

        # Update Pygame visuals with the RGB camera just processed
        speed_mph = tick_data.get('speed', 0.0) * 2.23694
        current_command = driving_data.get('command', 4)
        next_command = driving_data.get('next_command', 4)
        
        # Intercept the actual Leaderboard RouteScenario instance to find currently active `routes.xml` scenarios
        active_scenario_name = None
        route_scenario = self._get_route_scenario()
        if route_scenario is not None and hasattr(route_scenario, 'list_scenarios'):
            active_scenarios = []
            import py_trees
            for scenario in route_scenario.list_scenarios:
                if scenario.behavior_tree is not None:
                    # In `BasicScenario`, behavior_tree has 3 children: [Trigger, Scenario Behavior, End condition]
                    # We only care if the actual Scenario Behavior is RUNNING (meaning the trigger has already fired).
                    for child in scenario.behavior_tree.children:
                        if child.name == scenario.behavior_tree.name:
                            # This is the actual core scenario behavior
                            if child.status == py_trees.common.Status.RUNNING:
                                active_scenarios.append(scenario.name)
                            break
            
            if len(active_scenarios) > 0:
                active_scenario_name = " | ".join(active_scenarios)
            
        self._hic.run_interface(input_data, speed_mph=speed_mph, command_val=current_command,
                                next_command_val=next_command, active_scenario=active_scenario_name)

        # Use Human Interface (Wheel/Keyboard) for Control
        time_diff_ms = (timestamp - self._prev_timestamp) * 1000
        control = self._controller.parse_events(time_diff_ms)
        self._prev_timestamp = timestamp

        # Overwrite the AutoPilot's automated controls with our actual Human inputs
        driving_data['steer'] = control.steer
        driving_data['throttle'] = control.throttle
        driving_data['brake'] = bool(control.brake)
        
        if self.step % getattr(self.config, 'data_save_freq', 1) == 0:
            if getattr(self, 'save_path', None) is not None and getattr(self, 'datagen', False):
                # Save actual compressed image/lidar/box data exactly like DataAgent
                self.save_sensors(tick_data)
                
                # Append the exact measurements alongside human control inputs
                frame = self.step // self.config.data_save_freq
                measurements_dir = os.path.join(self.save_path, 'measurements')
                os.makedirs(measurements_dir, exist_ok=True)
                with open(os.path.join(measurements_dir, f'{frame:04d}.json'), 'w') as f:
                    json.dump(driving_data, f, indent=4)

        if 'lidar' in input_data:
            self.last_lidar = input_data['lidar']
        self.last_ego_transform = self._vehicle.get_transform()

        if plant:
            return {**tick_data, **driving_data}
        else:
            return control

    def destroy(self, results=None):
        if hasattr(self, '_hic') and self._hic is not None:
            self._hic.set_black_screen()
            self._hic._quit()
        super().destroy(results)
