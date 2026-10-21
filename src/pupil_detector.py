"""
Classic OpenCV Hough-circle based pupil detector for E-Eye photorefraction videos.
Provides full-frame detection, glint/overlay visualization, and video export helpers.
"""

import argparse
import subprocess
import shutil
import cv2
import numpy as np

SAMPLE_FRAMES = 10


def remove_glint(gray: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Suppress bright specular reflection using morphological opening."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


def detect_pupil(
    gray: np.ndarray,
    min_radius: int = 8,
    max_radius: int = 35,
    debug: bool = False
):
    """
    Search for circular pupil candidate using Hough Circles and candidate scoring.
    Returns (cx, cy, radius, brightness) or None.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(20, min(h, w) // 5),
        param1=50.0,
        param2=18.0,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        if debug:
            print("  Hough full-frame: no circles found")
        return None

    best = None
    best_score = -1.0

    for cx, cy, r in circles[0]:
        if r <= 0 or not (0 <= cx < w and 0 <= cy < h):
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]

        # Prioritize dark circular regions
        score = (255.0 - mean_intensity) * float(r)
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), int(r), float(mean_intensity))

    if debug and best is not None:
        print(f"  Hough full-frame: candidate ({best[0]}, {best[1]}) r={best[2]} b={best[3]:.1f}")

    return best


def draw_overlay(
    frame: np.ndarray,
    pupil: tuple,
    frame_idx: int = 1,
    total_frames: int = 1
) -> np.ndarray:
    """Draw circular detection ring and status hud on the frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # HUD Box
    hud_w, hud_h = min(360, w - 20), 55
    cv2.rectangle(vis, (10, 10), (10 + hud_w, 10 + hud_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (10, 10), (10 + hud_w, 10 + hud_h), (0, 255, 0), 2)

    if pupil is not None:
        cx, cy, r, b = pupil
        # Green pupil boundary
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
        # Small center glint dot
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

        cv2.putText(
            vis,
            f"Pupil Detected | Frame {frame_idx}/{total_frames}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Diameter: {r * 2}px | Brightness: {b:.1f}",
            (18, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis,
            f"Pupil Searching... | Frame {frame_idx}/{total_frames}",
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


def reencode_for_compatibility(path: str):
    """Optional ffmpeg pass to ensure standard H.264 profile."""
    if shutil.which("ffmpeg") is None:
        return
    tmp_out = path.rsplit(".", 1)[0] + "_tmp.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        tmp_out
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode == 0:
        shutil.move(tmp_out, path)


def main():
    parser = argparse.ArgumentParser(description="Classic OpenCV Pupil Detector")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("-o", "--output", default="classic_output.mp4", help="Output path")
    parser.add_argument("--debug", action="store_true", help="Print debug diagnostics")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pupil = detect_pupil(gray, debug=(args.debug and idx == 0))
        overlay = draw_overlay(frame, pupil, idx + 1, total_frames)
        writer.write(overlay)
        idx += 1

    cap.release()
    writer.release()
    print(f"Processed {idx} frames -> {args.output}")


if __name__ == "__main__":
    main()
