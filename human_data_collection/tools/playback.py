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
    
    # Setup display (1280x720 main camera + side cameras + lidar)
    # We will arrange them:
    # Top-Left: Center (640x360)
    # Bottom-Left: Left (320x180) | Rear (320x180) | Right (320x180) -> total width 960
    # Let's just make a window of say 1280 x 720 and blit scaled versions.
    
    width, height = 1280, 720
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA Data Playback")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('mono', 15)

    idx = 0
    paused = False

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
        frame_data = measurements[idx]
        sensors = frame_data.get('sensors', {})
        control = frame_data.get('control', {})
        
        # Helper to load and blit image
        def draw_img(sensor_id, pos, size):
            img_path = sensors.get(sensor_id)
            if img_path:
                full_path = os.path.join(data_folder, img_path)
                if os.path.exists(full_path):
                    arr = np.load(full_path)
                    # Swap axes for pygame (Width x Height x 3)
                    surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
                    surf = pygame.transform.scale(surf, size)
                    display.blit(surf, pos)

        # Draw Center (main)
        draw_img('Center', (0, 0), (800, 450))
        # Draw Left/Right/Rear
        draw_img('Left', (0, 450), (400, 225))
        draw_img('Rear', (400, 450), (400, 225))
        draw_img('Right', (800, 450), (400, 225))

        # Text overlay
        info_x = 820
        info_y = 20
        texts = [
            f"Frame: {idx}/{len(measurements)}",
            f"Time: {frame_data.get('timestamp', 0):.2f}",
            "",
            "--- CONTROLS ---",
            f"Steer:   {control.get('steer', 0.0):.3f}",
            f"Throttle:{control.get('throttle', 0.0):.3f}",
            f"Brake:   {control.get('brake', 0.0):.3f}",
            f"Reverse: {control.get('reverse', False)}",
            f"H-Brake: {control.get('hand_brake', False)}",
            "",
            "--- SENSORS ---",
            f"Speed: {sensors.get('speed', 0.0):.2f} m/s",
        ]
        
        imu = sensors.get('IMU')
        if imu:
            texts.append(f"IMU: {imu[0]:.2f}, {imu[1]:.2f}, {imu[2]:.2f}...")

        for i, t in enumerate(texts):
            tsurf = font.render(t, True, (255, 255, 255))
            display.blit(tsurf, (info_x, info_y + i * 20))

        # Instructions
        instr = font.render("SPACE=Pause/Play | L/R Arrows=Scrub | Q=Quit", True, (255, 255, 0))
        display.blit(instr, (info_x, info_y + 350))

        pygame.display.flip()
        clock.tick(20) # Playback at ~20 FPS

    pygame.quit()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python playback.py <path_to_data_folder>")
        print("Example: python playback.py data_collection/2026-02-23_00-29-11")
        sys.exit(1)
    
    main(sys.argv[1])
