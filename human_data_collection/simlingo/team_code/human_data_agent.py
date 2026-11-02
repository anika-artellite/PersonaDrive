"""
Child of the DataAgent that allows human control via PyGame + Logitech Wheel
while preserving all automated sensor data collection (boxes, LiDAR, RGB, semantics).
"""

import os
import sys
import datetime
import carla
import json
import math

# Try importing pygame for human control interface
try:
    import pygame
    from pygame.locals import K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_UP, K_a, K_d, K_s, K_w, K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ConfigParser for Logitech Wheel
if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

# Inherit from DataAgent so we get bounding_boxes, LiDAR sweeps, map parsing, and save_sensors for free.
from data_agent import DataAgent


def get_entry_point():
    return 'HumanDataAgent'


class HumanInterface(object):
    """
    Class to visualize the camera feed to the human player.
    Taken from human_agent.py
    """
    def __init__(self, width, height):
        self._surface = None

        # Force SDL (Pygame's backend) to launch the window at the top left corner of the entire 3-monitor space.
        # Under WSLg, 0,0 is perfectly at the top-left of the leftmost monitor's bounding box.
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        
        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        
        info = pygame.display.Info()
        # Since you are using 3 UHD TVs (4096x2160 each) in NVIDIA Surround,
        # WSLg/Pygame sometimes caps info.current_w (to 7680) or scales it unexpectedly
        # compared to when the monitors are separated.
        # We explicitly enforce the 3x UHD resolution here to ensure it spans perfectly.
        self._width = 4096 * 3  # 12288
        self._height = 2160
        # Use NOFRAME instead of FULLSCREEN. Borderless windows at '0,0' position span seamlessly.
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.NOFRAME)
            
        pygame.display.set_caption("SimLingo Human Data Agent")

    def run_interface(self, input_data, speed_mph=None):
        current_w, current_h = self._display.get_size()
        display_w = current_w // 3
        display_h = current_h
        
        # Draw a dark gray background. If left/right are missing from input_data, 
        # they will be gray, proving they are missing. If they are black, it means CARLA is sending black images.
        self._display.fill((30, 30, 30))
        
        # Calculate side mirror sizes (e.g., 40% width, 60% height of a single display)
        side_w = int(display_w * 0.4)
        side_h = int(display_h * 0.6)

        # 1) Center (Main RGB)
        if 'rgb' in input_data:
            image_center = input_data['rgb'][1][:, :, :3][:, :, ::-1]
            surface_center = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
            surface_center = pygame.transform.smoothscale(surface_center, (display_w, display_h))
            self._display.blit(surface_center, (display_w, 0))

        # 2) Left mirror (Stick to the right edge of the left display, vertically centered)
        if 'rgb_left' in input_data:
            image_left = input_data['rgb_left'][1][:, :, :3][:, :, ::-1]
            surface_left = pygame.surfarray.make_surface(image_left.swapaxes(0, 1))
            surface_left = pygame.transform.smoothscale(surface_left, (side_w, side_h))
            x_pos = display_w - side_w - 50 # 50px padding from the bezel
            y_pos = (display_h - side_h) // 2
            self._display.blit(surface_left, (x_pos, y_pos))

        # 3) Right mirror (Stick to the left edge of the right display, vertically centered)
        if 'rgb_right' in input_data:
            image_right = input_data['rgb_right'][1][:, :, :3][:, :, ::-1]
            surface_right = pygame.surfarray.make_surface(image_right.swapaxes(0, 1))
            surface_right = pygame.transform.smoothscale(surface_right, (side_w, side_h))
            x_pos = display_w * 2 + 50 # 50px padding from the bezel
            y_pos = (display_h - side_h) // 2
            self._display.blit(surface_right, (x_pos, y_pos))

        pygame.display.flip()

        if speed_mph is not None:
            font = pygame.font.SysFont('arial', 72, bold=True)
            speed_text = font.render(f"{speed_mph:.0f} MPH", True, (0, 255, 0))
            text_rect = speed_text.get_rect(center=(display_w * 1.5, display_h - 100))
            self._display.blit(speed_text, text_rect)
            pygame.display.update(text_rect)

    def set_black_screen(self):
        import numpy as np
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
    Taken from human_agent.py and adapted to directly return CARLA VehicleControl.
    """
    def __init__(self):
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

        # Initialize pygame joystick
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            print("WARNING: More than one joystick connected. Taking the first one.")

        self._joystick = None
        if joystick_count >= 1:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()

            self._parser = ConfigParser()
            # Look for wheel_config.ini in the root directory (where we copied it)
            config_path = os.path.join(os.environ.get('LEADERBOARD_ROOT', '.'), 'wheel_config.ini')
            if not os.path.exists(config_path):
                # Fallback to local
                config_path = 'wheel_config.ini'
                
            self._parser.read(config_path)
            self._steer_idx = int(self._parser.get('Logitech G923 Racing Wheel for PlayStation and PC', 'steering_wheel'))
            self._throttle_idx = int(self._parser.get('Logitech G923 Racing Wheel for PlayStation and PC', 'throttle'))
            self._brake_idx = int(self._parser.get('Logitech G923 Racing Wheel for PlayStation and PC', 'brake'))
            self._reverse_idx = int(self._parser.get('Logitech G923 Racing Wheel for PlayStation and PC', 'reverse'))
            self._handbrake_idx = int(self._parser.get('Logitech G923 Racing Wheel for PlayStation and PC', 'handbrake'))
        else:
            print("No Logitech Wheel recognized. Falling back to Keyboard keys (W,A,S,D / Arrows).")

    def parse_events(self, timestamp_ms):
        self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp_ms)
        if self._joystick is not None:
            self._parse_vehicle_wheel()
        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
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

        # Simple keyboard overrides
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

    def _parse_vehicle_wheel(self):
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


class HumanDataAgent(DataAgent):
    """
    Overrides `run_step` and `setup` to bypass the AutoPilot algorithms
    and use the Human wheel logic, but still uses DataAgent's `tick` and `save_sensors`.
    """
    def __init__(self, *args, **kwargs):
        # The standard leaderboard_evaluator passes (host, port, debug)
        # However, DataAgent inherits from AutoPilot which inherits from autonomous_agent_local
        # which expects (path_to_conf_file). We intercept and initialize the bare minimum
        # because AutoPilot passes the config_path through setup() anyway.
        from leaderboard.autoagents.autonomous_agent_local import Track
        from leaderboard.envs.sensor_interface import SensorInterface
        self.track = Track.MAP
        self._global_plan = None
        self._global_plan_world_coord = None
        self.sensor_interface = SensorInterface()
        self.wallclock_t0 = None

    def sensors(self):
        result = super().sensors()
        
        # Restored mirror resolution back to 1/2 of main camera for high fidelity
        mirror_w = self.config.camera_width // 2
        mirror_h = self.config.camera_height // 2

        # Left mirror (look back and left, pushed out safely beyond any vehicle mesh)
        result.append({
            'type': 'sensor.camera.rgb',
            'x': self.config.camera_pos[0],
            'y': self.config.camera_pos[1] - 1.0, # shifted left, guaranteed < 3.0m
            'z': self.config.camera_pos[2] + 0.2, # slightly higher
            'roll': self.config.camera_rot_0[0],
            'pitch': self.config.camera_rot_0[1],
            'yaw': self.config.camera_rot_0[2] - 160.0,
            'width': mirror_w,
            'height': mirror_h,
            'fov': self.config.camera_fov,
            'id': 'rgb_left'
        })
        # Right mirror (look back and right, pushed out safely beyond any vehicle mesh)
        result.append({
            'type': 'sensor.camera.rgb',
            'x': self.config.camera_pos[0],
            'y': self.config.camera_pos[1] + 1.0, # shifted right, guaranteed < 3.0m
            'z': self.config.camera_pos[2] + 0.2,
            'roll': self.config.camera_rot_0[0],
            'pitch': self.config.camera_rot_0[1],
            'yaw': self.config.camera_rot_0[2] + 160.0,
            'width': mirror_w,
            'height': mirror_h,
            'fov': self.config.camera_fov,
            'id': 'rgb_right'
        })

        return result

    def _draw_waypoints(self, waypoints, vertical_shift=0.1, size=0.1, downsample=1, timeout=10000.0):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        Used to guide human drivers in data collection.
        """
        # Ensure world is available
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

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(128, 128, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 128, 128)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(128, 32, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 32, 128)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(64, 64, 64)
            else:  # LANEFOLLOW
                color = carla.Color(0, 128, 0)  # Green

            world.debug.draw_point(wp, size=size, color=color, life_time=timeout)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=2*size,
                                    color=carla.Color(0, 0, 128), life_time=timeout)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=2*size,
                                    color=carla.Color(128, 128, 128), life_time=timeout)

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        # 1. Initialize all DataAgent setup (directories, models, etc).
        super().setup(path_to_conf_file, route_index, traffic_manager)
        
        # 2. Init Pygame UI elements
        # Force the exact full width of 3 NVIDIA Surround displays and NOFRAME flag in HumanInterface
        self._hic = HumanInterface(self.config.camera_width * 3, self.config.camera_height)
        self._controller = KeyboardControl()
        self._prev_timestamp = 0
        self._pygame_clock = pygame.time.Clock()

        # 3. Draw route arrows permanently for human driving collection
        if self._global_plan_world_coord is not None:
            self._draw_waypoints(self._global_plan_world_coord, vertical_shift=0.5, size=0.1, downsample=1)

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        """
        Bypasses `super().run_step` (which applies PID controls / IDM etc.),
        and instead routes control to Logitech Wheel.
        """
        self.step += 1
        self.step_tmp += 1
        
        self._pygame_clock.tick_busy_loop(20)

        # Ensure DataAgent map and managers are initialized
        if not self.initialized:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            client = CarlaDataProvider.get_client()
            world_map = client.get_world().get_map()
            self._init(world_map)

        # Convert LiDAR into ego coordinate (Logic inherited from DataAgent structure)
        import transfuser_utils as t_u
        input_data['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

        if self.datagen:
            self.augment_camera(sensors)

        # Process exactly the same DataAgent tick logic (Bounding boxes, LiDAR merging, BEV generation)
        tick_data = self.tick(input_data)
        tick_data.update(self.tick_autopilot(input_data))

        # Update Pygame visuals with the RGB camera just processed
        speed_mph = tick_data.get('speed', 0.0) * 2.23694
        self._hic.run_interface(input_data, speed_mph=speed_mph)

        # Use Wheel for Control instead of Privileged PID
        time_diff_ms = (timestamp - self._prev_timestamp) * 1000
        control = self._controller.parse_events(time_diff_ms)
        self._prev_timestamp = timestamp

        # Emulate the recording format of autopilot.py
        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake
        
        # We let AutoPilot compute its full privileged route planning, hazard checking, and aim_wp
        # so that our dataset contains perfectly dense waypoint information
        auto_control, driving_data = self._get_control(input_data, plant=True)

        # Overwrite the AutoPilot's automated controls with our actual Human inputs
        driving_data['steer'] = control.steer
        driving_data['throttle'] = control.throttle
        driving_data['brake'] = bool(control.brake)
        
        if self.step % self.config.data_save_freq == 0:
            if self.save_path is not None and self.datagen:
                # Save actual compressed data exactly like DataAgent
                self.save_sensors(tick_data)
                
                # Append the exact human control inputs to measurements
                frame = self.step // self.config.data_save_freq
                with open(os.path.join(self.save_path, 'measurements', f'{frame:04d}.json'), 'w') as f:
                    json.dump(driving_data, f, indent=4)

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
