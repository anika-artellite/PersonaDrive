#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pygame

def main(data_folder):
    measurements_file = os.path.join(data_folder, 'measurements.jsonl')
    if not os.path.exists(measurements_file):
        print(f"Error: {measurements_file} not found.")
        return

    # Load all measurements
    measurements = []
    with open(measurements_file, 'r') as f:
        for line in f:
            if line.strip():
                measurements.append(json.loads(line))

    if not measurements:
        print("No measurements found.")
        return

    print(f"Loaded {len(measurements)} frames.")

    pygame.init()
    pygame.font.init()
    
    width, height = 800, 800
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA LiDAR Bird's-Eye View")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('mono', 15)

    idx = 0
    paused = False
    
    # LiDAR display range in meters
    lidar_range = 50.0

    while idx < len(measurements):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    idx = min(len(measurements) - 1, idx + 1)
                elif event.key == pygame.K_LEFT:
                    idx = max(0, idx - 1)
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        if not paused:
            idx += 1
            if idx >= len(measurements):
                break

        display.fill((0, 0, 0))
        if idx < len(measurements):
            frame_data = measurements[idx]
            sensors = frame_data.get('sensors', {})
            
            lidar_file = sensors.get('LIDAR')
            if lidar_file:
                full_path = os.path.join(data_folder, lidar_file)
                if os.path.exists(full_path):
                    # Load the point cloud. Shape is usually (N, 3) or (N, 4)
                    pcd = np.load(full_path)
                    
                    if pcd.shape[0] > 0:
                        # We only need x and y for a top-down view (BEV)
                        points = pcd[:, :2].copy()
                        
                        # Scale from meters to pixels
                        # The CARLA LiDAR point is (X, Y) where X is forward and Y is right.
                        # We map it to Pygame coordinate space
                        scale_factor = min(width, height) / (2.0 * lidar_range)
                        points *= scale_factor
                        
                        # Center the points
                        points[:, 0] += width / 2.0
                        points[:, 1] += height / 2.0
                        
                        # Filter points outside the screen
                        good_points = np.logical_and.reduce((
                            points[:, 0] >= 0, points[:, 0] < width,
                            points[:, 1] >= 0, points[:, 1] < height
                        ))
                        points = points[good_points]
                        
                        # Convert to int coords
                        points = points.astype(np.int32)
                        
                        # Pygame 2D array representation is (width, height, 3)
                        lidar_img = np.zeros((width, height, 3), dtype=np.uint8)
                        
                        # Draw points (swap x,y for Pygame if needed. Pygame surface is [x][y])
                        if points.shape[0] > 0:
                            lidar_img[points[:, 0], points[:, 1]] = (255, 255, 255)
                        
                        surf = pygame.surfarray.make_surface(lidar_img)
                        display.blit(surf, (0, 0))
                        
                        # Draw ego vehicle indicator in the center (red rectangle)
                        # The point cloud is drawn (x, y) where x is forward in CARLA, but Pygame blits it sideways,
                        # so the car should be longer horizontally than vertically on the screen.
                        car_w, car_h = 20, 10
                        pygame.draw.rect(display, (255, 0, 0), (width//2 - car_w//2, height//2 - car_h//2, car_w, car_h))

            # Text overlay
            texts = [
                f"Frame: {idx+1}/{len(measurements)}",
                f"Time: {frame_data.get('timestamp', 0):.2f}",
                "SPACE=Pause/Play | L/R Arrows=Scrub | Q=Quit"
            ]

            for i, t in enumerate(texts):
                tsurf = font.render(t, True, (0, 255, 0))
                display.blit(tsurf, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(20) # Playback at ~20 FPS

    pygame.quit()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_lidar.py <path_to_data_folder>")
        print("Example: python visualize_lidar.py data_collection/2026-02-23_00-29-11")
        sys.exit(1)
    
    main(sys.argv[1])
