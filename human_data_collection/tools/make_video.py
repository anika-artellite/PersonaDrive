#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pygame

try:
    import cv2
except ImportError:
    print("Error: OpenCV is required to build the video.")
    print("Please install it using: pip install opencv-python")
    sys.exit(1)

def main(data_folder, output_file="output.mp4", fps=20):
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

    num_frames = len(measurements)
    print(f"Loaded {num_frames} frames. Creating video at {output_file}...")

    # Set up Pygame for offscreen rendering
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.font.init()
    
    width, height = 1280, 720
    display = pygame.Surface((width, height))
    font = pygame.font.SysFont('mono', 15)

    # Initialize OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for idx, frame_data in enumerate(measurements):
        display.fill((0, 0, 0))
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
            f"Frame: {idx+1}/{num_frames}",
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

        # Extract pixel data from Pygame surface (use array3d to avoid locking the surface)
        # Pygame uses RGB. OpenCV uses BGR.
        view = pygame.surfarray.array3d(display)
        
        # Numpy array shape is (width, height, 3). We need (height, width, 3) 
        frame_rgb = np.transpose(view, (1, 0, 2))
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Write to video
        out.write(frame_bgr)

        # Print progress
        if (idx + 1) % 50 == 0 or (idx + 1) == num_frames:
            print(f"Processed {idx + 1}/{num_frames} frames...")

    out.release()
    pygame.quit()
    print(f"Video saved to: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python make_video.py <path_to_data_folder> [output_file.mp4]")
        print("Example: python make_video.py data_collection/2026-02-23_00-29-11 session.mp4")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
    
    main(data_dir, output_file=out_file)
