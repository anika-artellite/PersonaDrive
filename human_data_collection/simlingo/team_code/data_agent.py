"""
Child of the autopilot that additionally runs data collection and storage.
"""

import cv2
import carla
import random
import torch
import numpy as np
import json
import os
import gzip
import laspy
from shapely.geometry import Polygon
from pathlib import Path
import concurrent.futures

from autopilot import AutoPilot
import transfuser_utils as t_u

from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from PIL import Image

from agents.tools.misc import (is_within_distance, get_trafficlight_trigger_location, compute_distance)

from agents.navigation.local_planner import LocalPlanner


# from: https://medium.com/codex/rgb-to-color-names-in-python-the-robust-way-ec4a9d97a01f
from scipy.spatial import KDTree
import webcolors

def convert_rgb_to_names(rgb_tuple):
    css3_db = {}
    try:
        css3_db = webcolors.CSS3_HEX_TO_NAMES
    except AttributeError:
        try:
            css3_db = webcolors.CSS2_HEX_TO_NAMES
        except AttributeError:
            try:
                css3_db = webcolors.css3_hex_to_names
            except AttributeError:
                pass
                
    names = []
    rgb_values = []
    
    if len(css3_db) > 0:
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            rgb_values.append(webcolors.hex_to_rgb(color_hex))
    else:
        # Fallback minimal color dictionary for newer webcolors versions 
        # that removed the raw dictionaries from public api
        css3_db = {'#000000': 'black', '#ffffff': 'white', '#ff0000': 'red', '#00ff00': 'green', '#0000ff': 'blue', '#ffff00': 'yellow', '#ffa500': 'orange', '#808080': 'gray'}
        for color_hex, color_name in css3_db.items():
            names.append(color_name)
            # manual hex_to_rgb so it doesn't fail
            rgb_values.append(tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
            
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'{names[index]}'


def get_entry_point():
    return 'DataAgent'


class DataAgent(AutoPilot):
    """
                Child of the autopilot that additionally runs data collection and storage.
                """

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager=None)

        self.SAVE_TF_LABELS = int(os.environ.get('SAVE_TF_LABELS', 0))

        self.weather_tmp = None
        self.step_tmp = 0

        self.tm = traffic_manager

        self.scenario_name = Path(path_to_conf_file).parent.name
        self.cutin_vehicle_starting_position = None

        if self.save_path is not None and self.datagen:
            (self.save_path / 'lidar').mkdir(exist_ok=True, parents=True)
            (self.save_path / 'rgb').mkdir(exist_ok=True, parents=True)
            (self.save_path / 'rgb_augmented').mkdir(exist_ok=True, parents=True)
            (self.save_path / 'boxes').mkdir(exist_ok=True, parents=True)
            if self.SAVE_TF_LABELS:
                (self.save_path / 'semantics').mkdir(exist_ok=True, parents=True)
                (self.save_path / 'semantics_augmented').mkdir(exist_ok=True, parents=True)
                (self.save_path / 'depth').mkdir(exist_ok=True, parents=True)
                (self.save_path / 'depth_augmented').mkdir(exist_ok=True, parents=True)
                (self.save_path / 'bev_semantics').mkdir(exist_ok=True, parents=True)
                (self.save_path / 'bev_semantics_augmented').mkdir(exist_ok=True, parents=True)

        self.tmp_visu = int(os.environ.get('TMP_VISU', 0))

        self._active_traffic_light = None
        self.last_lidar = None
        self.last_ego_transform = None
        
        # ThreadPoolExecutor to handle async data saving
        self._save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _init(self, hd_map):
        super()._init(hd_map)
        # if self.datagen:
        #     self.shuffle_weather()

        obs_config = {
                'width_in_pixels': self.config.lidar_resolution_width,
                'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,
                'pixels_per_meter': self.config.pixels_per_meter_collection,
                'history_idx': [-1],
                'scale_bbox': True,
                'scale_mask_col': 1.0,
                'map_folder': 'maps_2ppm_cv'
        }

        self.stop_sign_criteria = RunStopSign(self._world)
        self.ss_bev_manager = ObsManager(obs_config, self.config)
        self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)

        self.ss_bev_manager_augmented = ObsManager(obs_config, self.config)

        bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
        transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
        # Can't clone the carla vehicle object, so I use a dummy class with similar attributes.
        self.augmented_vehicle_dummy = t_u.CarlaActorDummy(self._vehicle.get_world(), bb_copy, transform_copy,
                                                                                                             self._vehicle.id)
        self.ss_bev_manager_augmented.attach_ego_vehicle(self.augmented_vehicle_dummy,
                                                                                                         criteria_stop=self.stop_sign_criteria)

        self._local_planner = LocalPlanner(self._vehicle, opt_dict={}, map_inst=self.world_map)
        
        # Cache static actors to avoid expensive `self._actors.filter('*')` roundtrips per frame
        all_actors = self._world.get_actors()
        self._traffic_lights_cache = all_actors.filter('*light*')
        self._stop_signs_cache = all_actors.filter('*stop*')
        
        self._traffic_lights_dict_cache = []
        for tl in self._traffic_lights_cache:
            transform = tl.get_transform()
            extent = tl.bounding_box.extent
            trigger = get_trafficlight_trigger_location(tl)
            wp = self.world_map.get_waypoint(trigger, project_to_road=False, lane_type=carla.libcarla.LaneType.Any)
            
            try:
                # Try to precompute the trigger box global position used by get_nearby_object equivalent
                tbgp = transform.transform(tl.trigger_volume.location)
                trigger_global_pos = carla.Location(x=tbgp.x, y=tbgp.y, z=tbgp.z)
            except:
                trigger_global_pos = transform.location
                
            self._traffic_lights_dict_cache.append({
                'actor': tl,
                'id': tl.id,
                'location': transform.location,
                'transform': transform,
                'rotation': transform.rotation,
                'matrix': np.array(transform.get_matrix()),
                'extent': [extent.x, extent.y, extent.z],
                'trigger_global_pos': trigger_global_pos,
                'wp_road_id': wp.road_id if wp else None,
                'wp_lane_id': wp.lane_id if wp else None,
                'wp_junction_id': wp.junction_id if wp else None,
            })
            
        self._stop_signs_dict_cache = []
        for ss in self._stop_signs_cache:
            transform = ss.get_transform()
            extent = ss.bounding_box.extent
            trigger = get_trafficlight_trigger_location(ss)
            wp = self.world_map.get_waypoint(trigger, project_to_road=False, lane_type=carla.libcarla.LaneType.Any)
            
            try:
                tbgp = transform.transform(ss.trigger_volume.location)
                trigger_global_pos = carla.Location(x=tbgp.x, y=tbgp.y, z=tbgp.z)
            except:
                trigger_global_pos = transform.location
                
            self._stop_signs_dict_cache.append({
                'actor': ss,
                'id': ss.id,
                'location': transform.location,
                'transform': transform,
                'rotation': transform.rotation,
                'matrix': np.array(transform.get_matrix()),
                'extent': [extent.x, extent.y, extent.z],
                'trigger_global_pos': trigger_global_pos,
                'wp_road_id': wp.road_id if wp else None,
                'wp_lane_id': wp.lane_id if wp else None,
                'wp_junction_id': wp.junction_id if wp else None,
            })
        
        # Cache per-actor properties once - statics never move, so the waypoint lookup and
        # matrix only need computing the first time each actor is seen. The cache is NOT
        # complete after _init: scenario props (construction cones, warning signs, barriers)
        # spawn when the ego nears the trigger point, long after this runs. Reconciliation
        # against the live actor list happens each frame in _refresh_static_actors_cache().
        self._static_actors_cache = []
        for static in all_actors.filter('static.*'):
            self._static_actors_cache.append(self._build_static_cache_entry(static))

    def _build_static_cache_entry(self, static):
        """Snapshot the properties of one static actor. Safe to do once: statics do not move."""
        transform = static.get_transform()
        wp = self.world_map.get_waypoint(transform.location, project_to_road=True, lane_type=carla.libcarla.LaneType.Any)

        # Use getattr with default to safely handle missing attributes without throwing errors during cache build
        mesh_path = static.attributes.get('mesh_path', None) if hasattr(static, 'attributes') else None

        return {
            'id': static.id,
            'type_id': static.type_id,
            'location': transform.location,
            'transform': transform,
            'rotation': transform.rotation,
            'matrix': np.array(transform.get_matrix()),
            'extent': static.bounding_box.extent if static.bounding_box else carla.Vector3D(0,0,0),
            'wp_road_id': wp.road_id if wp else None,
            'wp_lane_id': wp.lane_id if wp else None,
            'wp_junction_id': wp.junction_id if wp else None,
            'wp_lane_type': str(wp.lane_type) if wp else None,
            'mesh_path': mesh_path,
            'velocity': carla.Vector3D(0,0,0), # Static actors do not move
            'speed': 0.0
        }

    def _refresh_static_actors_cache(self):
        """Pick up statics that spawned after _init, and drop any that were destroyed.

        Scenario props are spawned by scenario_runner when the ego approaches the trigger
        point, so a cache built only in _init is permanently empty of them - which silently
        cost every ConstructionObstacle route its obstacle. Reuses existing entries by actor
        id so the per-actor waypoint lookup still happens only once, and filters the actor
        list already fetched at the top of get_bounding_boxes(), so this costs no extra RPC.
        """
        existing = {entry['id']: entry for entry in self._static_actors_cache}
        refreshed = []
        for static in self._actors.filter('static.*'):
            entry = existing.get(static.id)
            if entry is None:
                entry = self._build_static_cache_entry(static)
            refreshed.append(entry)
        self._static_actors_cache = refreshed

    def sensors(self):
        # workaraound that only does data augmentation at the beginning of the route
        if self.config.augment:
            self.augmentation_translation = np.random.uniform(low=self.config.camera_translation_augmentation_min,
                                                                                                                high=self.config.camera_translation_augmentation_max)
            self.augmentation_rotation = np.random.uniform(low=self.config.camera_rotation_augmentation_min,
                                                                                                         high=self.config.camera_rotation_augmentation_max)

        result = super().sensors()

        if self.save_path is not None and (self.datagen or self.tmp_visu):
            result += [{
                    'type': 'sensor.camera.rgb',
                    'x': self.config.camera_pos[0],
                    'y': self.config.camera_pos[1],
                    'z': self.config.camera_pos[2],
                    'roll': self.config.camera_rot_0[0],
                    'pitch': self.config.camera_rot_0[1],
                    'yaw': self.config.camera_rot_0[2],
                    'width': self.config.camera_width,
                    'height': self.config.camera_height,
                    'fov': self.config.camera_fov,
                    'id': 'rgb'
            }, {
                    'type': 'sensor.camera.rgb',
                    'x': self.config.camera_pos[0],
                    'y': self.config.camera_pos[1] + self.augmentation_translation,
                    'z': self.config.camera_pos[2],
                    'roll': self.config.camera_rot_0[0],
                    'pitch': self.config.camera_rot_0[1],
                    'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
                    'width': self.config.camera_width,
                    'height': self.config.camera_height,
                    'fov': self.config.camera_fov,
                    'id': 'rgb_augmented'
            }]

        result.append({
                'type': 'sensor.lidar.ray_cast',
                'x': self.config.lidar_pos[0],
                'y': self.config.lidar_pos[1],
                'z': self.config.lidar_pos[2],
                'roll': self.config.lidar_rot[0],
                'pitch': self.config.lidar_rot[1],
                'yaw': self.config.lidar_rot[2],
                'rotation_frequency': self.config.lidar_rotation_frequency,
                'points_per_second': self.config.lidar_points_per_second,
                'id': 'lidar'
        })

        if self.SAVE_TF_LABELS:
            result.append(
                {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': self.config.camera_pos[0],
                        'y': self.config.camera_pos[1],
                        'z': self.config.camera_pos[2],
                        'roll': self.config.camera_rot_0[0],
                        'pitch': self.config.camera_rot_0[1],
                        'yaw': self.config.camera_rot_0[2],
                        'width': self.config.camera_width,
                        'height': self.config.camera_height,
                        'fov': self.config.camera_fov,
                        'id': 'semantics'
                }, {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': self.config.camera_pos[0],
                        'y': self.config.camera_pos[1] + self.augmentation_translation,
                        'z': self.config.camera_pos[2],
                        'roll': self.config.camera_rot_0[0],
                        'pitch': self.config.camera_rot_0[1],
                        'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
                        'width': self.config.camera_width,
                        'height': self.config.camera_height,
                        'fov': self.config.camera_fov,
                        'id': 'semantics_augmented'
                }, {
                        'type': 'sensor.camera.depth',
                        'x': self.config.camera_pos[0],
                        'y': self.config.camera_pos[1],
                        'z': self.config.camera_pos[2],
                        'roll': self.config.camera_rot_0[0],
                        'pitch': self.config.camera_rot_0[1],
                        'yaw': self.config.camera_rot_0[2],
                        'width': self.config.camera_width,
                        'height': self.config.camera_height,
                        'fov': self.config.camera_fov,
                        'id': 'depth'
                }, {
                        'type': 'sensor.camera.depth',
                        'x': self.config.camera_pos[0],
                        'y': self.config.camera_pos[1] + self.augmentation_translation,
                        'z': self.config.camera_pos[2],
                        'roll': self.config.camera_rot_0[0],
                        'pitch': self.config.camera_rot_0[1],
                        'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
                        'width': self.config.camera_width,
                        'height': self.config.camera_height,
                        'fov': self.config.camera_fov,
                        'id': 'depth_augmented'
                }
            )



        return result

    def tick(self, input_data):
        result = {}

        if self.save_path is not None and (self.datagen or self.tmp_visu):
            rgb = input_data['rgb'][1][:, :, :3]
            rgb_augmented = input_data['rgb_augmented'][1][:, :, :3]

            if self.SAVE_TF_LABELS:
                # We store depth at 8 bit to reduce the filesize. 16 bit would be ideal, but we can't afford the extra storage.
                depth = input_data['depth'][1][:, :, :3]
                depth = (t_u.convert_depth(depth) * 255.0 + 0.5).astype(np.uint8)

                depth_augmented = input_data['depth_augmented'][1][:, :, :3]
                depth_augmented = (t_u.convert_depth(depth_augmented) * 255.0 + 0.5).astype(np.uint8)

                semantics = input_data['semantics'][1][:, :, 2]
                semantics_augmented = input_data['semantics_augmented'][1][:, :, 2]

        else:
            rgb = None
            rgb_augmented = None
            if self.SAVE_TF_LABELS:
                semantics = None
                semantics_augmented = None
                depth = None
                depth_augmented = None

        # The 10 Hz LiDAR only delivers half a sweep each time step at 20 Hz.
        # Here we combine the 2 sweeps into the same coordinate system
        if self.last_lidar is not None:
            ego_transform = self._vehicle.get_transform()
            ego_location = ego_transform.location
            last_ego_location = self.last_ego_transform.location
            relative_translation = np.array([
                    ego_location.x - last_ego_location.x, ego_location.y - last_ego_location.y,
                    ego_location.z - last_ego_location.z
            ])

            ego_yaw = ego_transform.rotation.yaw
            last_ego_yaw = self.last_ego_transform.rotation.yaw
            relative_rotation = np.deg2rad(t_u.normalize_angle_degree(ego_yaw - last_ego_yaw))

            orientation_target = np.deg2rad(ego_yaw)
            # Rotate difference vector from global to local coordinate system.
            rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                                                    [np.sin(orientation_target),
                                                                     np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
            relative_translation = rotation_matrix.T @ relative_translation

            lidar_last = t_u.algin_lidar(self.last_lidar, relative_translation, relative_rotation)
            # Combine back and front half of LiDAR
            lidar_360 = np.concatenate((input_data['lidar'], lidar_last), axis=0)
        else:
            lidar_360 = input_data['lidar']  # The first frame only has 1 half

        if self.step % self.config.data_save_freq == 0:
            bounding_boxes = self.get_bounding_boxes(lidar=lidar_360)
        else:
            bounding_boxes = []

        self.stop_sign_criteria.tick(self._vehicle)

        if self.SAVE_TF_LABELS:
            bev_semantics = self.ss_bev_manager.get_observation(self.close_traffic_lights)
            bev_semantics_augmented = self.ss_bev_manager_augmented.get_observation(self.close_traffic_lights)

            if self.tmp_visu:
                self.visualuize(bev_semantics['rendered'], rgb)

        result.update({
                'lidar': lidar_360,
                'rgb': rgb,
                'rgb_augmented': rgb_augmented,
                'bounding_boxes': bounding_boxes,
        })
        if self.SAVE_TF_LABELS:
            result.update({
                'semantics': semantics,
                'semantics_augmented': semantics_augmented,
                'depth': depth,
                'depth_augmented': depth_augmented,
                'bev_semantics': bev_semantics['bev_semantic_classes'],
                'bev_semantics_augmented': bev_semantics_augmented['bev_semantic_classes'],
            })

        return result

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        self.step_tmp += 1

        # Convert LiDAR into the coordinate frame of the ego vehicle
        input_data['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

        # Must be called before run_step, so that the correct augmentation shift is saved
        if self.datagen:
            self.augment_camera(sensors)

        control = super().run_step(input_data, timestamp, plant=plant)

        tick_data = self.tick(input_data)

        if self.step % self.config.data_save_freq == 0:
            if self.save_path is not None and self.datagen:
                self.save_sensors(tick_data)

        self.last_lidar = input_data['lidar']
        self.last_ego_transform = self._vehicle.get_transform()

        if plant:
            # Control contains data when run with plant
            return {**tick_data, **control}
        else:
            return control

    def augment_camera(self, sensors):
        # Update dummy vehicle
        if self.initialized:
            # We are still rendering the map for the current frame, so we need to use the translation from the last frame.
            last_translation = self.augmentation_translation
            last_rotation = self.augmentation_rotation
            bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
            transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
            augmented_loc = transform_copy.transform(carla.Location(0.0, last_translation, 0.0))
            transform_copy.location = augmented_loc
            transform_copy.rotation.yaw = transform_copy.rotation.yaw + last_rotation
            self.augmented_vehicle_dummy.bounding_box = bb_copy
            self.augmented_vehicle_dummy.transform = transform_copy

    def _get_night_mode(self, weather):
        """Check wheather or not the street lights need to be turned on"""
        SUN_ALTITUDE_THRESHOLD_1 = 15
        SUN_ALTITUDE_THRESHOLD_2 = 165

        # For higher fog and cloudness values, the amount of light in scene starts to rapidly decrease
        CLOUDINESS_THRESHOLD = 80
        FOG_THRESHOLD = 40

        # In cases where more than one weather conditition is active, decrease the thresholds
        COMBINED_THRESHOLD = 10

        altitude_dist = weather.sun_altitude_angle - SUN_ALTITUDE_THRESHOLD_1
        altitude_dist = min(altitude_dist, SUN_ALTITUDE_THRESHOLD_2 - weather.sun_altitude_angle)
        cloudiness_dist = CLOUDINESS_THRESHOLD - weather.cloudiness
        fog_density_dist = FOG_THRESHOLD - weather.fog_density

        # Check each parameter independetly
        if altitude_dist < 0 or cloudiness_dist < 0 or fog_density_dist < 0:
            return True

        # Check if two or more values are close to their threshold
        joined_threshold = int(altitude_dist < COMBINED_THRESHOLD)
        joined_threshold += int(cloudiness_dist < COMBINED_THRESHOLD)
        joined_threshold += int(fog_density_dist < COMBINED_THRESHOLD)

        if joined_threshold >= 2:
            return True

        return False

    def shuffle_weather(self):
        # change weather for visual diversity
        if self.weather_tmp is None:
            t = carla.WeatherParameters
            options = dir(t)[:22]
            chosen_preset = random.choice(options)
            self.chosen_preset = chosen_preset
            weather = t.__getattribute__(t, chosen_preset)
            self.weather_tmp = weather

        self._world.set_weather(self.weather_tmp)

        # night mode
        vehicles = self._world.get_actors().filter('*vehicle*')
        if self._get_night_mode(weather):
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        else:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState.NONE)

    def _async_save_worker(self, tick_data, frame, save_path):
        """ The background job that actually writes data to the disk. 
            Runs on a separate thread via the ThreadPoolExecutor. """
        try:
            # CARLA images are already in opencv's BGR format.
            cv2.imwrite(str(save_path / 'rgb' / (f'{frame:04}.jpg')), tick_data['rgb'])
            cv2.imwrite(str(save_path / 'rgb_augmented' / (f'{frame:04}.jpg')), tick_data['rgb_augmented'])

            if self.SAVE_TF_LABELS:
                cv2.imwrite(str(save_path / 'semantics' / (f'{frame:04}.png')), tick_data['semantics'])
                cv2.imwrite(str(save_path / 'semantics_augmented' / (f'{frame:04}.png')), tick_data['semantics_augmented'])

                cv2.imwrite(str(save_path / 'depth' / (f'{frame:04}.png')), tick_data['depth'])
                cv2.imwrite(str(save_path / 'depth_augmented' / (f'{frame:04}.png')), tick_data['depth_augmented'])

                cv2.imwrite(str(save_path / 'bev_semantics' / (f'{frame:04}.png')), tick_data['bev_semantics'])
                cv2.imwrite(str(save_path / 'bev_semantics_augmented' / (f'{frame:04}.png')),
                                        tick_data['bev_semantics_augmented'])

            # Specialized LiDAR compression format
            header = laspy.LasHeader(point_format=self.config.point_format)
            header.offsets = np.min(tick_data['lidar'], axis=0)
            header.scales = np.array([self.config.point_precision, self.config.point_precision, self.config.point_precision])

            with laspy.open(save_path / 'lidar' / (f'{frame:04}.laz'), mode='w', header=header) as writer:
                point_record = laspy.ScaleAwarePointRecord.zeros(tick_data['lidar'].shape[0], header=header)
                point_record.x = tick_data['lidar'][:, 0]
                point_record.y = tick_data['lidar'][:, 1]
                point_record.z = tick_data['lidar'][:, 2]

                writer.write_points(point_record)

            with gzip.open(save_path / 'boxes' / (f'{frame:04}.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump(tick_data['bounding_boxes'], f, indent=4)

            print(f'[DataAgent] Saved frame {frame:04d} -> {save_path}', flush=True)

        except Exception:
            import traceback
            print(f'[DataAgent] ERROR in _async_save_worker for frame {frame:04d}:', flush=True)
            print(traceback.format_exc(), flush=True)

    def save_sensors(self, tick_data):
        frame = self.step // self.config.data_save_freq
        print(f'[DataAgent] save_sensors called: step={self.step}, frame={frame}, save_path={self.save_path}, '
              f'bb_count={len(tick_data.get("bounding_boxes", []))}', flush=True)
        
        # Fire and forget the disk writing to a background thread
        # Important: Wait to pass deepcopies if tick_data lists are mutated by tick() in the next frame
        # NumPy arrays are typically re-instantiated every frame from input_data, so this is safe.
        self._save_executor.submit(self._async_save_worker, tick_data, frame, self.save_path)

    def destroy(self, results=None):
        torch.cuda.empty_cache()

        if hasattr(self, '_save_executor'):
            self._save_executor.shutdown(wait=True)

        if results is not None and self.save_path is not None:
            with gzip.open(os.path.join(self.save_path, 'results.json.gz'), 'wt', encoding='utf-8') as f:
                json.dump(results.__dict__, f, indent=2)

        super().destroy(results)
        
        
    def _wps_next_until_lane_end(self, wp):
        try:
            road_id_cur = wp.road_id
            lane_id_cur = wp.lane_id
            road_id_next = road_id_cur
            lane_id_next = lane_id_cur
            curr_wp = [wp]
            next_wps = []
            # https://github.com/carla-simulator/carla/issues/2511#issuecomment-597230746
            while road_id_cur == road_id_next and lane_id_cur == lane_id_next:
                next_wp = curr_wp[0].next(1)
                if len(next_wp) == 0:
                    break
                curr_wp = next_wp
                next_wps.append(next_wp[0])
                road_id_next = next_wp[0].road_id
                lane_id_next = next_wp[0].lane_id
        except:
            next_wps = []
            
        return next_wps

    def get_bounding_boxes(self, lidar=None):
        import time
        start_bb = time.time()
        results = []

        ego_transform = self._vehicle.get_transform()
        ego_control = self._vehicle.get_control()
        ego_velocity = self._vehicle.get_velocity()
        ego_matrix = np.array(ego_transform.get_matrix())
        ego_rotation = ego_transform.rotation
        ego_extent = self._vehicle.bounding_box.extent
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
        ego_yaw = np.deg2rad(ego_rotation.yaw)
        ego_brake = ego_control.brake

        relative_yaw = 0.0
        relative_pos = t_u.get_relative_transform(ego_matrix, ego_matrix)

        ego_wp = self.world_map.get_waypoint(self._vehicle.get_location(), project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
        
        # to compute lane_relative_to_ego for walkers and other cars we first have to precompute some in which direction the opposite lane is & the width of the center lane
        left_wp, right_wp = ego_wp.get_left_lane(), ego_wp.get_right_lane()
        left_decreasing_lane_id = left_wp is not None and left_wp.lane_id < ego_wp.lane_id or right_wp is not None and right_wp.lane_id > ego_wp.lane_id
        
        remove_lanes_for_lane_relative_to_ego = 1
        wp = ego_wp
        is_opposite = False
        while True:
            flag = ego_wp.lane_id > 0 and left_decreasing_lane_id or ego_wp.lane_id < 0 and not left_decreasing_lane_id
            if is_opposite:
                flag = not flag
            wp = wp.get_left_lane() if flag else wp.get_right_lane()
                
            if wp is None or wp.lane_type == carla.LaneType.Driving and ego_wp.lane_id * wp.lane_id < 0:
                break
            
            is_opposite = ego_wp.lane_id * wp.lane_id < 0
            
            if wp.lane_type != carla.LaneType.Driving:
                remove_lanes_for_lane_relative_to_ego += 1

        # how far is next junction
        next_wps = self._wps_next_until_lane_end(ego_wp)
        try:
            next_lane_wps_ego = next_wps[-1].next(1)
            if len(next_lane_wps_ego) == 0:
                next_lane_wps_ego = [next_wps[-1]]
        except:
            next_lane_wps_ego = []
        if ego_wp.is_junction:
            distance_to_junction_ego = 0.0
            # get distance to ego vehicle
        elif len(next_lane_wps_ego)>0 and next_lane_wps_ego[0].is_junction:
            distance_to_junction_ego = next_lane_wps_ego[0].transform.location.distance(ego_wp.transform.location)
        else:
            distance_to_junction_ego = None
            
        next_road_ids_ego = []
        next_next_road_ids_ego = []
        for i, wp in enumerate(next_lane_wps_ego):
                next_road_ids_ego.append(wp.road_id)
                next_next_wps = self._wps_next_until_lane_end(wp)
                try:
                    next_next_lane_wps_ego = next_next_wps[-1].next(1)
                    if len(next_next_lane_wps_ego) == 0:
                        next_next_lane_wps_ego = [next_next_wps[-1]]
                except:
                    next_next_lane_wps_ego = []
                for j, wp2 in enumerate(next_next_lane_wps_ego):
                    if wp2.road_id not in next_next_road_ids_ego:
                        next_next_road_ids_ego.append(wp2.road_id)

        tl = self._world.get_traffic_lights_from_waypoint(ego_wp, 50.0)
        if len(tl) == 0:
            tl_state = 'None'
        else:
            tl_state = str(tl[0].state)

        ego_lane_direction = ego_wp.lane_id / abs(ego_wp.lane_id)
        lanes_to_the_left = []
        lanes_to_the_right = []
        num_lanes_same_direction = 1 # ego lane
        lane_ids_same_direction = [ego_wp.lane_id]
        lane_id_left_most_lane_same_direction = ego_wp.lane_id
        lane_id_right_most_lane_opposite_direction = ego_wp.lane_id
        num_lanes_opposite_direction = 0
        shoulder_left = False
        shoulder_right = False
        parking_left = False
        parking_right = False
        sidewalk_left = False
        sidewalk_right = False
        bikelane_left = False
        bikelane_right = False

        # for loop over left and ride side of the road
        for i, lanes in enumerate([lanes_to_the_left, lanes_to_the_right]):
            lane_wp = ego_wp
            is_road = True
            # is_opposite is needed because get_left_lane() returns the left lane from the view point of the lane
            # this means if we dont do this and we have a oncoming lane to the left it would just toggle between
            # the oncoming lane and the ego lane
            is_opposite = False

            while is_road:
                # first we check for all lanes to the left
                if i == 0:
                    if not is_opposite:
                        lane_wp = lane_wp.get_left_lane()
                    else:
                        lane_wp = lane_wp.get_right_lane()
                # then we check for all lanes to the right
                else:
                    if not is_opposite:
                        lane_wp = lane_wp.get_right_lane()
                    else:
                        lane_wp = lane_wp.get_left_lane()


                if lane_wp is None:
                    is_road = False
                else:
                    direction = lane_wp.lane_id / abs(lane_wp.lane_id)
                    lane_type = lane_wp.lane_type
                    if lane_type == carla.LaneType.Driving and direction == ego_lane_direction:
                        num_lanes_same_direction += 1
                        lane_ids_same_direction.append(lane_wp.lane_id)
                        if i == 0:
                            lane_id_left_most_lane_same_direction = lane_wp.lane_id
                    elif lane_type == carla.LaneType.Driving and direction != ego_lane_direction:
                        num_lanes_opposite_direction += 1
                    elif lane_type == carla.LaneType.Shoulder and i == 0 and lane_wp.lane_width > 1.0:
                        shoulder_left = True
                    elif lane_type == carla.LaneType.Shoulder and i == 1 and lane_wp.lane_width > 1.0:
                        shoulder_right = True
                    elif lane_type == carla.LaneType.Parking and i == 0:
                        parking_left = True
                    elif lane_type == carla.LaneType.Parking and i == 1:
                        parking_right = True
                    elif lane_type == carla.LaneType.Sidewalk and i == 0:
                        sidewalk_left = True
                    elif lane_type == carla.LaneType.Sidewalk and i == 1:
                        sidewalk_right = True
                    elif lane_type == carla.LaneType.Biking and i == 0:
                        bikelane_left = True
                    elif lane_type == carla.LaneType.Biking and i == 1:
                        bikelane_right = True
                    else:
                        pass


                    if direction != ego_lane_direction:
                        if is_opposite == False:
                            lane_id_right_most_lane_opposite_direction = lane_wp.lane_id
                        is_opposite = True
                    lanes.append(lane_wp)

                # get ego lane number counted from left to right
                #https://www.asam.net/standards/detail/opendrive/
                # most left should be always the smallest number
                min_lane_id = min(lane_ids_same_direction)
                ego_lane_number = abs(ego_wp.lane_id - lane_id_left_most_lane_same_direction)

        ego_lane = {
            'type:': str(ego_wp.lane_type),
            'width': ego_wp.lane_width,
        }
        left_lanes = [
            {
                'type:': str(lane_wp.lane_type),
                'width': lane_wp.lane_width,
            } for lane_wp in lanes_to_the_left
        ]
        right_lanes = [
            {
                'type:': str(lane_wp.lane_type),
                'width': lane_wp.lane_width,
            } for lane_wp in lanes_to_the_right
        ]
        
        t0 = time.time()

        # Check for possible vehicle obstacles
        # Retrieve all relevant actors. This single RPC call is much faster than multiple .filter() calls.
        self._actors = self._world.get_actors()
        snapshot = self._world.get_snapshot()
        ego_loc = self._vehicle.get_location()
        
        all_vehicles = [a for a in self._actors if 'vehicle' in a.type_id]
        
        vehicle_dict_list = []
        for v in all_vehicles:
            if v.id == self._vehicle.id:
                continue
            actor_snap = snapshot.find(v.id)
            if not actor_snap:
                continue
            loc = actor_snap.get_transform().location
            dist = loc.distance(ego_loc)
            
            v_dict = {
                'actor': v,
                'id': v.id,
                'transform': actor_snap.get_transform(),
                'location': loc,
                'velocity': actor_snap.get_velocity(),
                'acceleration': actor_snap.get_acceleration(),
                'bounding_box': v.bounding_box,
                'distance': dist,
                'wp': None
            }
            
            if dist < max(self.config.bb_save_radius, 40.0):
                # Precompute the waypoint so we don't do it repetitively in V-Obstacles and Vehicles
                v_dict['wp'] = self.world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
                
            vehicle_dict_list.append(v_dict)

        walker_list = [a for a in self._actors if 'walker' in a.type_id]
        
        walker_dict_list = []
        for w in walker_list:
            actor_snap = snapshot.find(w.id)
            if not actor_snap:
                continue
            loc = actor_snap.get_transform().location
            dist = loc.distance(ego_loc)
            
            if dist < self.config.bb_save_radius:
                w_dict = {
                    'actor': w,
                    'id': w.id,
                    'transform': actor_snap.get_transform(),
                    'location': loc,
                    'velocity': actor_snap.get_velocity(),
                    'bounding_box': w.bounding_box,
                    'distance': dist,
                }
                w_dict['wp'] = self.world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.libcarla.LaneType.Any)
                walker_dict_list.append(w_dict)

        hazard_detected_10 = False
        affected_by_vehicle_10, aff_vehicle_id_10, aff_vehicle_dis_10 = self._vehicle_obstacle_detected(vehicle_dict_list, 10)
        if affected_by_vehicle_10:
                hazard_detected_10 = True
                
        hazard_detected_15 = False
        affected_by_vehicle_15, aff_vehicle_id_15, aff_vehicle_dis_15 = self._vehicle_obstacle_detected(vehicle_dict_list, 15)
        if affected_by_vehicle_15:
                hazard_detected_15 = True
                
        hazard_detected_20 = False
        affected_by_vehicle_20, aff_vehicle_id_20, aff_vehicle_dis_20 = self._vehicle_obstacle_detected(vehicle_dict_list, 20)
        if affected_by_vehicle_20:
                hazard_detected_20 = True
                
        hazard_detected_40 = False
        affected_by_vehicle_40, aff_vehicle_id_40, aff_vehicle_dis_40 = self._vehicle_obstacle_detected(vehicle_dict_list, 40)
        if affected_by_vehicle_40:
                hazard_detected_40 = True
                        
        try:
            next_action = self.tm.get_next_action(self._vehicle)[0]
        except:
            next_action = None
            
        t1 = time.time()
        result = {
                'class': 'ego_car',
                'extent': [ego_dx[0], ego_dx[1], ego_dx[2]],
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw,
                'num_points': -1,
                'distance': -1,
                'speed': ego_speed,
                'brake': ego_brake,
                'id': int(self._vehicle.id),
                'matrix': ego_transform.get_matrix()
        }
        results.append(result)

        for v_dict in vehicle_dict_list:
            if v_dict['distance'] < self.config.bb_save_radius:
                vehicle = v_dict['actor']
                vehicle_transform = v_dict['transform']
                vehicle_rotation = vehicle_transform.rotation
                vehicle_matrix = np.array(vehicle_transform.get_matrix())
                vehicle_control = vehicle.get_control() # RPC
                vehicle_velocity = v_dict['velocity']
                vehicle_extent = v_dict['bounding_box'].extent
                vehicle_id = v_dict['id']
                vehicle_wp = v_dict['wp']
                if vehicle_wp is None:
                    continue # Should not happen based on filtering logic
                
                same_road_as_ego = False
                lane_relative_to_ego = None
                same_direction_as_ego = False
                    
                next_wps = self._wps_next_until_lane_end(vehicle_wp)
                next_lane_wps = next_wps[-1].next(1)
                if len(next_lane_wps) == 0:
                    next_lane_wps = [next_wps[-1]]
                
                next_next_wps = []
                for i, wp in enumerate(next_lane_wps):
                    next_next_wps = self._wps_next_until_lane_end(wp)
                
                try:
                    next_next_lane_wps = next_next_wps[-1].next(1)
                    if len(next_next_lane_wps) == 0:
                        next_next_lane_wps = [next_next_wps[-1]]
                except:
                    next_next_lane_wps = []
                
                if vehicle_wp.is_junction:
                    distance_to_junction = 0.0
                    # get distance to ego vehicle
                elif next_lane_wps[0].is_junction:
                    distance_to_junction = next_lane_wps[0].transform.location.distance(vehicle_wp.transform.location)
                else:
                    distance_to_junction = None
                    
                next_road_ids = []
                for i, wp in enumerate(next_lane_wps):
                    if wp.road_id not in next_road_ids:
                        next_road_ids.append(wp.road_id)
                
                next_next_road_ids = []
                for i, wp in enumerate(next_next_lane_wps):
                    if wp.road_id not in next_next_road_ids:
                        next_next_road_ids.append(wp.road_id)
                
                is_at_traffic_light = vehicle.is_at_traffic_light()

                tl = self._world.get_traffic_lights_from_waypoint(vehicle_wp, 30.0)
                if len(tl) == 0:
                    tl_state_vehicle = 'None'
                else:
                    tl_state_vehicle = str(tl[0].state)

                if vehicle_wp.road_id == ego_wp.road_id:
                    same_road_as_ego = True

                    direction = vehicle_wp.lane_id / abs(vehicle_wp.lane_id)
                    if direction == ego_lane_direction:
                        same_direction_as_ego = True

                    lane_relative_to_ego = vehicle_wp.lane_id - ego_wp.lane_id
                    lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                    
                    if not same_direction_as_ego:
                        lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                    
                    lane_relative_to_ego = -lane_relative_to_ego

                vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
                yaw = np.deg2rad(vehicle_rotation.yaw)

                relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)
                vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
                vehicle_brake = vehicle_control.brake
                vehicle_steer = vehicle_control.steer
                vehicle_throttle = vehicle_control.throttle

                # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                # This now happens offline to save computation during simulation.
                num_in_bbox_points = -1

                distance = np.linalg.norm(relative_pos)
                try:
                    rgb = tuple(map(int, vehicle.attributes['color'].split(',')))
                    color_name = convert_rgb_to_names(rgb)
                except:
                    rgb = None
                    color_name = None
                try:
                    light_state = vehicle.get_light_state()
                except:
                    light_state = 99
                light_state_bin = bin(int(light_state))
                # get positions of 1's
                light_state_bin_pos = [i for i, x in enumerate(reversed(light_state_bin)) if x == '1']
                # get decimal value of 1's
                light_state_dec_pos = [2**i for i in light_state_bin_pos]
                # get VehicleLightState of 1's
                light_state = [carla.VehicleLightState.values[i] for i in light_state_dec_pos]
                
                try:
                    next_action = self.tm.get_next_action(vehicle)[0]
                except:
                    next_action = None
                    
                                            
                vehicle_cuts_in = False
                if (self.scenario_name == 'ParkingCutIn') and vehicle.attributes['role_name']=='scenario':
                    if self.cutin_vehicle_starting_position is None:
                        self.cutin_vehicle_starting_position = vehicle.get_location()

                    if vehicle.get_location().distance(self.cutin_vehicle_starting_position) > 0.2 and vehicle.get_location().distance(self.cutin_vehicle_starting_position) < 8: # to make sure the vehicle drives
                        vehicle_cuts_in = True
                        
                elif (self.scenario_name == 'StaticCutIn') and vehicle.attributes['role_name']=='scenario':
                    if vehicle_speed > 1.0 and abs(vehicle_steer) > 0.2:
                        vehicle_cuts_in = True
                
                        
                result = {
                        'class': 'car',
                        'color_rgb': rgb,
                        'color_name': color_name,
                        'next_action': next_action,
                        'vehicle_cuts_in': vehicle_cuts_in,
                        'road_id': vehicle_wp.road_id,
                        'lane_id': vehicle_wp.lane_id,
                        'lane_type': vehicle_wp.lane_type,
                        'lane_type_str': str(vehicle_wp.lane_type),
                        'is_in_junction': vehicle_wp.is_junction,
                        'junction_id': vehicle_wp.junction_id,
                        'distance_to_junction': distance_to_junction,
                        'next_junction_id': next_lane_wps[0].junction_id,
                        'next_road_ids': next_road_ids,
                        'next_next_road_ids': next_next_road_ids,
                        'same_road_as_ego': same_road_as_ego,
                        'same_direction_as_ego': same_direction_as_ego,
                        'lane_relative_to_ego': lane_relative_to_ego,
                        'light_state': light_state_dec_pos,
                        'traffic_light_state': tl_state_vehicle,
                        'is_at_traffic_light': is_at_traffic_light,
                        'base_type': vehicle.attributes['base_type'],
                        'number_of_wheels': vehicle.attributes['number_of_wheels'],
                        'extent': vehicle_extent_list,
                        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                        'yaw': relative_yaw,
                        'num_points': int(num_in_bbox_points),
                        'distance': distance,
                        'speed': vehicle_speed,
                        'brake': vehicle_brake,
                        'steer': vehicle_steer,
                        'throttle': vehicle_throttle,
                        'id': int(vehicle_id),
                        'role_name': vehicle.attributes['role_name'],
                        'type_id': vehicle.type_id,
                        'matrix': vehicle_transform.get_matrix()
                }
                results.append(result)

        t2 = time.time()

        # Filter locally to avoid Carla RPC roundtrips
        for w_dict in walker_dict_list:
            walker = w_dict['actor']
            walker_transform = w_dict['transform']
            walker_velocity = w_dict['velocity']
            walker_rotation = walker_transform.rotation
            walker_matrix = np.array(walker_transform.get_matrix())
            walker_id = w_dict['id']
            walker_extent = w_dict['bounding_box'].extent
            walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
            yaw = np.deg2rad(walker_rotation.yaw)

            relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, walker_matrix)

            walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)

            # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
            # This now happens offline to save computation during simulation.
            num_in_bbox_points = -1

            distance = w_dict['distance']
            
            walker_wp = w_dict['wp']
            lane_type = walker_wp.lane_type
            same_road_as_ego = False
            lane_relative_to_ego = None
            same_direction_as_ego = False
            
            if walker_wp.road_id == ego_wp.road_id:
                same_road_as_ego = True

                direction = walker_wp.lane_id / abs(walker_wp.lane_id)
                if direction == ego_lane_direction:
                    same_direction_as_ego = True

                lane_relative_to_ego = walker_wp.lane_id - ego_wp.lane_id
                lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                
                if not same_direction_as_ego:
                    lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                
                lane_relative_to_ego = -lane_relative_to_ego
            

            result = {
                    'class': 'walker',
                    'role_name': walker.attributes['role_name'],
                    'gender': walker.attributes['gender'],
                    'age': walker.attributes['age'],
                    'extent': walker_extent,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'num_points': int(num_in_bbox_points),
                    'distance': distance,
                    'speed': walker_speed,
                    'id': int(walker_id),
                    'lane_type': lane_type,
                    'same_road_as_ego': same_road_as_ego,
                    'same_direction_as_ego': same_direction_as_ego,
                    'lane_relative_to_ego': lane_relative_to_ego,
                    'matrix': walker_transform.get_matrix()
            }
            results.append(result)

        for traffic_light in self.close_traffic_lights:
            traffic_light_extent = [traffic_light[0].extent.x, traffic_light[0].extent.y, traffic_light[0].extent.z]

            traffic_light_transform = carla.Transform(traffic_light[0].location, traffic_light[0].rotation)
            traffic_light_rotation = traffic_light_transform.rotation
            traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
            yaw = np.deg2rad(traffic_light_rotation.yaw)

            relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, traffic_light_matrix)

            distance = np.linalg.norm(relative_pos)

            result = {
                    'class': 'traffic_light',
                    'extent': traffic_light_extent,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'distance': distance,
                    'state': str(traffic_light[1]),
                    'id': int(traffic_light[2]),
                    'affects_ego': traffic_light[3],
                    'matrix': traffic_light_transform.get_matrix()
            }
            results.append(result)

        for stop_sign in self.close_stop_signs:
            stop_sign_extent = [stop_sign[0].extent.x, stop_sign[0].extent.y, stop_sign[0].extent.z]

            stop_sign_transform = carla.Transform(stop_sign[0].location, stop_sign[0].rotation)
            stop_sign_rotation = stop_sign_transform.rotation
            stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
            yaw = np.deg2rad(stop_sign_rotation.yaw)

            relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, stop_sign_matrix)

            distance = np.linalg.norm(relative_pos)

            result = {
                    'class': 'stop_sign',
                    'extent': stop_sign_extent,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'distance': distance,
                    'id': int(stop_sign[1]),
                    'affects_ego': stop_sign[2],
                    'matrix': stop_sign_transform.get_matrix()
            }
            results.append(result)

        t3 = time.time()
            
            
        ### stop sign and traffic lights for vqa - add new entry becaus i don't want to mess with tranfuser setting
        
        # We manually process the dictionaries here instead of using `get_nearby_object` to avoid slow RPCs
        ego_loc = ego_transform.location
        radius = self.config.bb_save_radius
        
        traffic_lights = [tl for tl in self._traffic_lights_dict_cache if tl['trigger_global_pos'].distance(ego_loc) < radius]

        for tl_dict in traffic_lights:
            traffic_light_transform = tl_dict['transform']
            traffic_light_rotation = tl_dict['rotation']
            traffic_light_matrix = tl_dict['matrix']
            traffic_light_id = tl_dict['id']
            traffic_light_extent = tl_dict['extent']
            yaw = np.deg2rad(traffic_light_rotation.yaw)

            relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, traffic_light_matrix)
            
            # continue if light is ehind us
            if relative_pos[0] < 0:
                continue

            # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
            # This now happens offline to save computation during simulation.
            num_in_bbox_points = -1

            distance = np.linalg.norm(relative_pos)
            
            same_road_as_ego = False
            lane_relative_to_ego = None
            same_direction_as_ego = False
            
            try:
                if tl_dict['wp_road_id'] == ego_wp.road_id:
                    same_road_as_ego = True

                    direction = tl_dict['wp_lane_id'] / abs(tl_dict['wp_lane_id']) if tl_dict['wp_lane_id'] != 0 else 1
                    if direction == ego_lane_direction:
                        same_direction_as_ego = True

                    lane_relative_to_ego = tl_dict['wp_lane_id'] - ego_wp.lane_id
                    lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                    
                    if not same_direction_as_ego:
                        lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                    
                    lane_relative_to_ego = -lane_relative_to_ego
            except:
                pass
            
            # This is the single RPC call per traffic light (which is very fast compared to checking all of them)
            # Only done for nearby traffic lights
            tl_state = tl_dict['actor'].state

            result = {
                    'class': 'traffic_light_vqa',
                    'extent': traffic_light_extent,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'road_id': tl_dict['wp_road_id'],
                    'lane_id': tl_dict['wp_lane_id'],
                    'junction_id': tl_dict['wp_junction_id'],
                    'yaw': relative_yaw,
                    'num_points': int(num_in_bbox_points),
                    'distance': distance,
                    'state_str': str(tl_state),
                    'state': int(tl_state),
                    'same_road_as_ego': same_road_as_ego,
                    'same_direction_as_ego': same_direction_as_ego,
                    'affects_ego': same_direction_as_ego,
                    'lane_relative_to_ego': lane_relative_to_ego,
            }
            results.append(result)

        t4 = time.time()


        stop_signs = [ss for ss in self._stop_signs_dict_cache if ss['trigger_global_pos'].distance(ego_loc) < radius]

        for ss_dict in stop_signs:
            stop_sign_transform = ss_dict['transform']
            stop_sign_rotation = ss_dict['rotation']
            stop_sign_matrix = ss_dict['matrix']
            stop_sign_id = ss_dict['id']
            stop_sign_extent = ss_dict['extent']
            yaw = np.deg2rad(stop_sign_rotation.yaw)

            relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, stop_sign_matrix)

            # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
            # This now happens offline to save computation during simulation.
            num_in_bbox_points = -1

            distance = np.linalg.norm(relative_pos)
            
            same_road_as_ego = False
            lane_relative_to_ego = None
            same_direction_as_ego = False
            
            if ss_dict['wp_road_id'] == ego_wp.road_id:
                same_road_as_ego = True

                direction = ss_dict['wp_lane_id'] / abs(ss_dict['wp_lane_id']) if ss_dict['wp_lane_id'] != 0 else 1
                if direction == ego_lane_direction:
                    same_direction_as_ego = True

                lane_relative_to_ego = ss_dict['wp_lane_id'] - ego_wp.lane_id
                lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                
                if not same_direction_as_ego:
                    lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                
                lane_relative_to_ego = -lane_relative_to_ego

            result = {
                    'class': 'stop_sign_vqa',
                    'extent': stop_sign_extent,
                    'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                    'yaw': relative_yaw,
                    'num_points': int(num_in_bbox_points),
                    'distance': distance,
                    'road_id': ss_dict['wp_road_id'],
                    'lane_id': ss_dict['wp_lane_id'],
                    'junction_id': ss_dict['wp_junction_id'],
                    'same_road_as_ego': same_road_as_ego,
                    'same_direction_as_ego': same_direction_as_ego,
                    'affects_ego': same_direction_as_ego,
                    'lane_relative_to_ego': lane_relative_to_ego,
            }
            results.append(result)



        # Note this only saves static actors, which does not include static background objects
        # static_list = self._actors.filter('*static*')
        # for static in static_list:
        #     if static.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        #         static_transform = static.get_transform()
        #         static_velocity = static.get_velocity()
        #         static_rotation = static.get_transform().rotation
        #         static_matrix = np.array(static_transform.get_matrix())
        #         static_id = static.id
        #         static_extent = static.bounding_box.extent
        #         static_extent = [static_extent.x, static_extent.y, static_extent.z]
        #         yaw = np.deg2rad(static_rotation.yaw)

        #         relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
        #         relative_pos = t_u.get_relative_transform(ego_matrix, static_matrix)

        #         static_speed = self._get_forward_speed(transform=static_transform, velocity=static_velocity)

        #         # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
        #         if not lidar is None:
        #             num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, static_extent, lidar)
        #         else:
        #             num_in_bbox_points = -1

        #         distance = np.linalg.norm(relative_pos)

        #         result = {
        #                 'class': 'static',
        #                 'extent': static_extent,
        #                 'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
        #                 'yaw': relative_yaw,
        #                 'num_points': int(num_in_bbox_points),
        #                 'distance': distance,
        #                 'speed': static_speed,
        #                 'id': int(static_id),
        #                 'matrix': static_transform.get_matrix(),
        #                 'type_id': static.type_id,
        #                 'mesh_path': static.attributes['mesh_path'] if 'mesh_path' in static.attributes else None
        #         }
        #         results.append(result)
       

        ego_loc = self._vehicle.get_location()
        self._refresh_static_actors_cache()
        statics = self._static_actors_cache
        for static in statics:
            if static['location'].distance(ego_loc) < self.config.bb_save_radius:
                static_transform = static['transform']
                static_rotation = static['rotation']
                static_location = static['location']
                static_matrix = static['matrix']
                static_id = static['id']
                static_extent = static['extent']
                #BUG: static_extent x and y are swapped
                static_extent = [static_extent.y, static_extent.x, static_extent.z]
                yaw = np.deg2rad(static_rotation.yaw)

                relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
                relative_pos = t_u.get_relative_transform(ego_matrix, static_matrix)
                distance = np.linalg.norm(relative_pos)
                
                static_speed = 0.0

                # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
                # This now happens offline to save computation during simulation.
                num_in_bbox_points = -1

                same_road_as_ego = False
                lane_relative_to_ego = None
                same_direction_as_ego = False
                if static['wp_road_id'] == ego_wp.road_id:
                    same_road_as_ego = True
                    
                    direction = static['wp_lane_id'] / abs(static['wp_lane_id']) if static['wp_lane_id'] != 0 else 1
                    if direction == ego_lane_direction:
                        same_direction_as_ego = True

                    lane_relative_to_ego = static['wp_lane_id'] - ego_wp.lane_id
                    lane_relative_to_ego *= -1 if left_decreasing_lane_id else 1
                    
                    if not same_direction_as_ego:
                        lane_relative_to_ego += remove_lanes_for_lane_relative_to_ego * (1 if lane_relative_to_ego < 0 else -1)
                    
                    lane_relative_to_ego = -lane_relative_to_ego


                if static['type_id'] == 'static.prop.mesh':
                    if static['mesh_path'] and "Car" in static['mesh_path']:
                        result = {
                            'class': 'static_car',
                            'extent': static_extent,
                            'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                            'yaw': relative_yaw,
                            'distance': distance,
                            'speed': static_speed,
                            'num_points': int(num_in_bbox_points),
                            'road_id': static['wp_road_id'],
                            'junction_id': static['wp_junction_id'],
                            'lane_id': static['wp_lane_id'],
                            'on_lane_type': static['wp_lane_type'],
                            'same_road_as_ego': same_road_as_ego,
                            'same_direction_as_ego': same_direction_as_ego,
                            'lane_relative_to_ego': lane_relative_to_ego,
                            'matrix': static_transform.get_matrix(),
                            'mesh_path': static['mesh_path']
                        }
                    else:
                        pass
                elif static['type_id'] == 'static.prop.trafficwarning': # the huge traffic warning sign in the scenarios ConstructionObstacle and ConstructionObstacleTwoWays
                    result = {
                        'class': 'static_trafficwarning',
                        'extent': static_extent,
                        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                        'yaw': relative_yaw,
                        'num_points': int(num_in_bbox_points),
                        'distance': distance,
                        'mesh_path': static['mesh_path']
                    }
                else:
                    result = {
                        'class': 'static',
                        'type_id': static['type_id'],
                        'extent': static_extent,
                        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                        'yaw': relative_yaw,
                        'num_points': int(num_in_bbox_points),
                        'distance': distance,
                        'mesh_path': static['mesh_path']
                    }
                
                if 'class' in locals() and 'result' in locals():
                    results.append(result)
                    
        t5 = time.time()

        landmarks = ego_wp.get_landmarks(40.0)
        for landmark in landmarks:
            landmark_transform = landmark.transform
            landmark_location = landmark_transform.location
            landmark_rotation = landmark_transform.rotation
            landmark_matrix = np.array(landmark_transform.get_matrix())

            yaw = np.deg2rad(landmark_rotation.yaw)

            relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
            relative_pos = t_u.get_relative_transform(ego_matrix, landmark_matrix)
            distance = np.linalg.norm(relative_pos)

            result = {
                'class': 'landmark',
                'name': landmark.name,
                'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
                'yaw': relative_yaw,
                'distance': distance,
                'id': int(landmark.id),
                'text': landmark.text,
                'value': landmark.value,
            }
            results.append(result)

        # weather information:
        weather = self._world.get_weather()
        weather_info = {
            'class': 'weather',
            'cloudiness': weather.cloudiness,
            'dust_storm': weather.dust_storm,
            'fog_density': weather.fog_density,
            'fog_distance': weather.fog_distance,
            'fog_falloff': weather.fog_falloff,
            'mie_scattering_scale': weather.mie_scattering_scale,
            'precipitation': weather.precipitation,
            'precipitation_deposits': weather.precipitation_deposits,
            'rayleigh_scattering_scale': weather.rayleigh_scattering_scale,
            'scattering_intensity': weather.scattering_intensity,
            'sun_altitude_angle': weather.sun_altitude_angle,
            'sun_azimuth_angle': weather.sun_azimuth_angle,
            'wetness': weather.wetness,
            'wind_intensity': weather.wind_intensity,
        }
        results.append(weather_info)
        
        try:
            next_is_junction = next_lane_wps_ego[0].is_junction
            next_junction_id = next_lane_wps_ego[0].junction_id
        except:
            next_is_junction = None
            next_junction_id = None

        result = {
            'class': 'ego_info',
            'scenario': self.scenario_name,
            'traffic_light_state': tl_state,
            'distance_to_junction': distance_to_junction_ego,
            'ego_lane_number': ego_lane_number,
            'road_id': ego_wp.road_id,
            'lane_id': ego_wp.lane_id,
            'is_in_junction': ego_wp.is_junction,
            'is_intersection': ego_wp.is_intersection,
            'junction_id': ego_wp.junction_id,
            'next_road_junction': next_is_junction,
            'next_junction_id': next_junction_id,
            'next_road_ids': next_road_ids_ego,
            'next_next_road_ids_ego': next_next_road_ids_ego,
            'ego_lane': ego_lane,
            'left_lanes': left_lanes,
            'right_lanes': right_lanes,
            'num_lanes_same_direction': num_lanes_same_direction,
            'num_lanes_opposite_direction': num_lanes_opposite_direction,
            'lane_change': ego_wp.lane_change,
            'lane_change_str': str(ego_wp.lane_change),
            'lane_type': ego_wp.lane_type,
            'lane_type_str': str(ego_wp.lane_type),
            'left_lane_marking_color': ego_wp.left_lane_marking.color,
            'left_lane_marking_color_str': str(ego_wp.left_lane_marking.color),
            'left_lane_marking_type': ego_wp.left_lane_marking.type,
            'left_lane_marking_type_str': str(ego_wp.left_lane_marking.type),
            'right_lane_marking_color': ego_wp.right_lane_marking.color,
            'right_lane_marking_color_str': str(ego_wp.right_lane_marking.color),
            'right_lane_marking_type': ego_wp.right_lane_marking.type,
            'right_lane_marking_type_str': str(ego_wp.right_lane_marking.type),
            'shoulder_left': shoulder_left,
            'shoulder_right': shoulder_right,
            'parking_left': parking_left,
            'parking_right': parking_right,
            'sidewalk_left': sidewalk_left,
            'sidewalk_right': sidewalk_right,
            'bike_lane_left': bikelane_left,
            'bike_lane_right': bikelane_right,
            'hazard_detected_10': hazard_detected_10,
            'affects_ego_10': aff_vehicle_id_10,
            'hazard_detected_15': hazard_detected_15,
            'affects_ego_15': aff_vehicle_id_15,
            'hazard_detected_20': hazard_detected_20,
            'affects_ego_20': aff_vehicle_id_20,
            'hazard_detected_40': hazard_detected_40,
            'affects_ego_40': aff_vehicle_id_40,
        }
        results.append(result)

        t_end = time.time()
        print(f"BB Timing -> Total: {(t_end-start_bb)*1000:.1f}ms | Init: {(t0-start_bb)*1000:.1f}ms | V-Obstacles: {(t1-t0)*1000:.1f}ms | Vehicles: {(t2-t1)*1000:.1f}ms | Walkers: {(t3-t2)*1000:.1f}ms | Traffic Lights: {(t4-t3)*1000:.1f}ms | Stop Signs: {(t5-t4)*1000:.1f}ms | Statics: {(t_end-t5)*1000:.1f}ms", flush=True)

        return results

    def get_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
        """
                Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
                :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
                :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
                :param extent: List, Extent of the bounding box
                :param lidar: LiDAR point cloud
                :return: Returns the number of LiDAR hits within the bounding box of the
                vehicle
                """

        rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                                                [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

        # LiDAR in the with the vehicle as origin
        vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

        # check points in bbox
        x, y, z = extent[0], extent[1], extent[2]
        num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                                    (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
        return num_points

    def visualuize(self, rendered, visu_img):
        rendered = cv2.resize(rendered, dsize=(visu_img.shape[1], visu_img.shape[1]), interpolation=cv2.INTER_LINEAR)
        visu_img = cv2.cvtColor(visu_img, cv2.COLOR_BGR2RGB)

        final = np.concatenate((visu_img, rendered), axis=0)

        Image.fromarray(final).save(self.save_path / (f'{self.step:04}.jpg'))

    def _vehicle_obstacle_detected(self,
                                                                 vehicle_list=None,
                                                                 max_distance=None,
                                                                 up_angle_th=90,
                                                                 low_angle_th=0,
                                                                 lane_offset=0):
        """
                Method to check if there is a vehicle in front of the agent blocking its path.

                        :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                                If None, all vehicle in the scene are used
                        :param max_distance: max freespace to check for obstacles.
                                If None, the base threshold value is used
                """
        self._use_bbs_detection = False
        self._offset = 0

        def get_route_polygon():
            route_bb = []
            extent_y = self._vehicle.bounding_box.extent.y
            r_ext = extent_y + self._offset
            l_ext = -extent_y + self._offset
            r_vec = ego_transform.get_right_vector()
            p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
            p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
            route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            for wp, _ in self._local_planner.get_plan():
                if ego_location.distance(wp.transform.location) > max_distance:
                    break

                r_vec = wp.transform.get_right_vector()
                p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
                p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
                route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

            # Two points don't create a polygon, nothing to check
            if len(route_bb) < 3:
                return None

            return Polygon(route_bb)

        if vehicle_list is None:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        ego_wpt = self.world_map.get_waypoint(ego_location, lane_type=carla.libcarla.LaneType.Any)

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(self._vehicle.bounding_box.extent.x *
                                                                                                     ego_transform.get_forward_vector())

        opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
        use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

        # Get the route bounding box
        route_polygon = get_route_polygon()

        for target_dict in vehicle_list: # vehicle_list is now vehicle_dict_list!
            if target_dict['distance'] > max_distance:
                continue

            target_transform = target_dict['transform']
            target_wpt = target_dict['wp']
            if target_wpt is None:
                continue

            # General approach for junctions and vehicles invading other lanes due to the offset
            if (use_bbs or target_wpt.is_junction) and route_polygon:

                target_bb = target_dict['bounding_box']
                target_vertices = target_bb.get_world_vertices(target_transform)
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if route_polygon.intersects(target_polygon):
                    return (True, target_dict['id'], target_dict['distance'])

            # Simplified approach, using only the plan waypoints (similar to TM)
            else:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_dict['bounding_box'].extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                        x=target_extent * target_forward_vector.x,
                        y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    return (True, target_dict['id'], target_dict['distance'])

        return (False, None, -1)

    def _get_forward_speed(self, transform=None, velocity=None):
        """
                Calculate the forward speed of the vehicle based on its transform and velocity.

                Args:
                        transform (carla.Transform, optional): The transform of the vehicle. If not provided, it will be obtained from the vehicle.
                        velocity (carla.Vector3D, optional): The velocity of the vehicle. If not provided, it will be obtained from the vehicle.

                Returns:
                        float: The forward speed of the vehicle in m/s.
                """
        if not velocity:
            velocity = self._vehicle.get_velocity()

        if not transform:
            transform = self._vehicle.get_transform()

        # Convert the velocity vector to a NumPy array
        velocity_np = np.array([velocity.x, velocity.y, velocity.z])

        # Convert rotation angles from degrees to radians
        pitch_rad = np.deg2rad(transform.rotation.pitch)
        yaw_rad = np.deg2rad(transform.rotation.yaw)

        # Calculate the orientation vector based on pitch and yaw angles
        orientation_vector = np.array(
                [np.cos(pitch_rad) * np.cos(yaw_rad),
                 np.cos(pitch_rad) * np.sin(yaw_rad),
                 np.sin(pitch_rad)])

        # Calculate the forward speed by taking the dot product of velocity and orientation vectors
        forward_speed = np.dot(velocity_np, orientation_vector)

        return forward_speed
