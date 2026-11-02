#!/usr/bin/env python3
"""
compute_bbox_points.py - count LiDAR hits inside each recorded bounding box, offline.

Run after a route is driven; collect_session.sh calls it automatically. The counts let
the data loader drop boxes that were never actually visible to the sensor.

    python3 compute_bbox_points.py --dataset_root <route dir>

get_points_in_bbox() below is taken from SimLingo's team_code/data_agent.py
(Apache License 2.0) so the offline counts match the ones computed during collection.
"""

import os
import glob
import gzip
import json
import laspy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def get_points_in_bbox(vehicle_pos, vehicle_yaw, extent, lidar):
    """
    Checks for a given vehicle in ego coordinate system, how many LiDAR hits there are in its bounding box.
    """
    rotation_matrix = np.array([
        [np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
        [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0],
        [0.0, 0.0, 1.0]
    ])

    # LiDAR with the vehicle as origin
    vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

    # check points in bbox
    x, y, z = extent[0], extent[1], extent[2]
    num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & 
                  (vehicle_lidar[:, 1] < y) & (vehicle_lidar[:, 1] > -y) & 
                  (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
    return int(num_points)

def process_route_folder(route_path_str):
    route_path = Path(route_path_str)
    
    boxes_path = route_path / 'boxes'
    lidar_path = route_path / 'lidar'
    
    if not boxes_path.exists() or not lidar_path.exists():
        print(f"Skipping {route_path} because 'boxes' or 'lidar' directory is missing.")
        return

    box_files = sorted(list(boxes_path.glob('*.json.gz')))
    
    # Process each frame
    for box_file in tqdm(box_files, desc=f"Processing {route_path.name}"):
        frame_idx = box_file.name.split('.')[0]
        lidar_file = lidar_path / f"{frame_idx}.laz"
        
        if not lidar_file.exists():
            continue
            
        # Read LiDAR
        try:
            with laspy.open(str(lidar_file)) as f:
                las = f.read()
                lidar_points = np.vstack((las.x, las.y, las.z)).transpose()
        except Exception as e:
            print(f"Error loading {lidar_file}: {e}")
            continue

        # Read Boxes
        try:
            with gzip.open(box_file, 'rt', encoding='utf-8') as f:
                boxes = json.load(f)
        except Exception as e:
            print(f"Error loading {box_file}: {e}")
            continue
            
        modified = False
        for box in boxes:
            if 'num_points' in box and box['num_points'] == -1:
                if 'position' in box and 'yaw' in box and 'extent' in box:
                    pos = np.array(box['position'])
                    yaw = box['yaw']
                    extent = box['extent']
                    num_pts = get_points_in_bbox(pos, yaw, extent, lidar_points)
                    box['num_points'] = num_pts
                    modified = True
                    
        # Save back to file
        if modified:
            with gzip.open(box_file, 'wt', encoding='utf-8') as f:
                json.dump(boxes, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute bounding box lidar hit points offline.")
    parser.add_argument('--dataset_root', type=str, required=True, help="Path to the root of the dataset containing route folders.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Dataset root {dataset_root} does not exist.")
        exit(1)
        
    route_folders = [f for f in dataset_root.iterdir() if f.is_dir()]
    
    for route_folder in route_folders:
        process_route_folder(route_folder)
        
    print("Offline calculation complete!")
