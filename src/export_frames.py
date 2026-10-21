"""
Utility: Frame Scraper & Dataset Extractor for E-Eye Photorefractor Footage
==========================================================================
Scrapes and exports PNG frames from videos in data/ for labeling, annotation,
and model training.
"""

import argparse
import glob
import os
import cv2


def extract_frames(video_path: str, output_dir: str, stride: int = 5, prefix: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return 0

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            out_name = f"{prefix}{base_name}_f{frame_idx:04d}.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} -> {output_dir}")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from E-Eye videos for dataset labeling.")
    parser.add_argument("--input", "-i", default="../data/reference/ideal captured.mp4",
                        help="Video file or folder containing MP4 videos")
    parser.add_argument("--output", "-o", default="../data/extracted_frames",
                        help="Destination directory for PNG frames")
    parser.add_argument("--stride", "-s", type=int, default=5,
                        help="Frame stride (sample every Nth frame). Default: 5")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        videos = glob.glob(os.path.join(args.input, "**", "*.mp4"), recursive=True)
        print(f"Found {len(videos)} videos in {args.input}")
        total = 0
        for v in videos:
            total += extract_frames(v, args.output, stride=args.stride)
        print(f"Total exported frames: {total}")
    else:
        extract_frames(args.input, args.output, stride=args.stride)


if __name__ == "__main__":
    main()
