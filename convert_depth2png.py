import os
import numpy as np
import cv2
import argparse

def convert_depth_npy_to_png(npy_path):
    # Expand user and absolute path
    npy_path = os.path.expanduser(npy_path)
    output_base_dir = os.path.dirname(npy_path)
    output_dir = os.path.join(output_base_dir, "depth")
    os.makedirs(output_dir, exist_ok=True)

    # Load .npy file
    depth_stack = np.load(npy_path)  # shape: (N, H, W)
    print(f"Loaded depth shape: {depth_stack.shape}")

    for i, depth in enumerate(depth_stack):
        # Convert meters to millimeters and store as uint16
        depth_mm = (depth * 1000.0).astype(np.uint16)
        filename = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(filename, depth_mm)

    print(f"✅ Saved {len(depth_stack)} depth frames to '{output_dir}' as 16-bit PNGs.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', required=True, help='Path to depth_matrices.npy')
    args = parser.parse_args()

    convert_depth_npy_to_png(args.npy)

