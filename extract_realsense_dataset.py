import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse

def extract_from_bag(bag_fname):
    base_name = os.path.splitext(os.path.basename(bag_fname))[0]
    output_dir = os.path.join(os.getcwd(), base_name)
    color_dir = os.path.join(output_dir, 'rgb')
    os.makedirs(color_dir, exist_ok=True)

    depth_matrices = []
    timestamps = []

    config = rs.config()
    pipeline = rs.pipeline()
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)
    rs.config.enable_device_from_file(config, bag_fname, repeat_playback=False)
    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    i = 0
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=100)
            if frames.size() < 2:
                continue
        except RuntimeError:
            print(f'Finished reading frames. Total frames: {i}')
            break

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        scaled_depth_image = depth_image * depth_scale
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        filename = f'{i:04d}.png'
        filepath = os.path.join(color_dir, filename)
        cv2.imwrite(filepath, color_image)

        # Get timestamp in seconds
        timestamp = color_frame.get_timestamp() / 1000.0  # convert ms to seconds
        timestamps.append((timestamp, f'rgb/{filename}'))

        depth_matrices.append(scaled_depth_image.copy())
        i += 1

    pipeline.stop()

    # Save depth data
    np.save(os.path.join(output_dir, 'depth_matrices.npy'), np.array(depth_matrices))

    # Save rgb.txt
    with open(os.path.join(output_dir, 'rgb.txt'), 'w') as f:
        for t, fname in timestamps:
            f.write(f"{t:.5f} {fname}\n")

    print(f"Saved {i} frames and timestamps to '{output_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True, help='Path to the RealSense .bag file')
    args = parser.parse_args()
    extract_from_bag(args.bag)

