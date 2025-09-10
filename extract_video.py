import cv2
import os
import argparse

def extract_from_lrv(video_path):
    # Create output folder named after the video file (without extension)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.getcwd(), base_name)
    rgb_dir = os.path.join(output_dir, 'rgb')

    os.makedirs(rgb_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print(f"Video FPS: {fps}, Total frames: {frame_count}, Duration: {duration:.2f}s")

    timestamp_log = []

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps
        filename = f"{frame_id:04d}.png"
        filepath = os.path.join(rgb_dir, filename)
        cv2.imwrite(filepath, frame)

        timestamp_log.append(f"{timestamp:.6f} rgb/{filename}")
        frame_id += 1

    cap.release()

    # Write timestamps to file
    txt_path = os.path.join(output_dir, 'rgb.txt')
    with open(txt_path, 'w') as f:
        for line in timestamp_log:
            f.write(f"{line}\n")

    print(f"Extracted {frame_id} frames to '{rgb_dir}'")
    print(f"Timestamps saved to '{txt_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrv', required=True, help='Path to the .LRV file')
    args = parser.parse_args()

    extract_from_lrv(args.lrv)
