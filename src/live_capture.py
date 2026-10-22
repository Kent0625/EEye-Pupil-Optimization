"""
E-Eye Live Pupil Capture & Real-Time Tracking Program
====================================================
Runs real-time pupil detection directly from a live camera (Webcam, USB Camera,
or Raspberry Pi Camera) using the Zoom-In Then Capture pipeline.

Hotkeys during live preview:
  - 'v'      : Toggle display view (Zoomed Eye -> Side-by-Side -> Full Camera Frame)
  - 's'/SPACE: Save a high-resolution snapshot picture
  - 'r'      : Start or stop recording a video clip
  - 'q'/ESC  : Quit live capture
"""

import argparse
import datetime
import os
import sys
import time
import cv2
import numpy as np
import torch
from torchvision import transforms

# Ensure local modules can be imported
sys.path.insert(0, os.path.dirname(__file__))
try:
    from unet import UNet
except ImportError:
    UNet = None

from optimized_inference import (
    detect_eye_crop_box,
    detect_pupil_in_zoomed_roi,
    render_shaded_overlay,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="E-Eye Live Camera Pupil Capture & Tracking."
    )
    parser.add_argument(
        "--camera",
        "-c",
        default="0",
        help="Camera device index (e.g. 0, 1) or video path. Default: 0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested camera capture width. Default: 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested camera capture height. Default: 480",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested camera capture frame rate. Default: 30",
    )
    parser.add_argument(
        "--view",
        "-v",
        choices=["zoomed", "side-by-side", "full"],
        default="zoomed",
        help="Starting view mode: 'zoomed' (eye only, default), 'side-by-side', or 'full'.",
    )
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=480,
        help="Pixel dimension of the square zoomed eye workspace. Default: 480",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model weights checkpoint. If omitted, auto-loads the best trained model.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output video filename to automatically record the entire session.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Headless mode (for servers or headless Raspberry Pi).",
    )
    return parser.parse_args()


def load_best_checkpoint(device: torch.device, user_ckpt: str = None):
    if UNet is None:
        return None

    candidates = []
    if user_ckpt:
        candidates.append(user_ckpt)
    candidates.extend([
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pupil-unet-zoomed-best.ckpt"),
        os.path.join(os.path.dirname(__file__), "checkpoints", "pupil-unet-zoomed-best.ckpt"),
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", "pupil-unet-epoch48-val_dice0.7170.ckpt"),
    ])

    for cp in candidates:
        if os.path.exists(cp):
            try:
                model = UNet(in_channels=1, num_classes=1)
                ckpt = torch.load(cp, map_location=device)
                sd = ckpt.get("state_dict", ckpt)
                unet_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
                model.load_state_dict(unet_sd or sd, strict=False)
                model = model.to(device)
                model.eval()
                print(f"Loaded trained neural model: {os.path.basename(cp)}")
                return model
            except Exception:
                pass
    return None


def main():
    args = parse_args()
    device = torch.device("cpu")

    print("=" * 65)
    print("E-EYE LIVE CAMERA PUPIL CAPTURE & TRACKING")
    print("=" * 65)
    print(f"Camera Source : {args.camera}")
    print(f"Initial View  : {args.view.upper()} (Eye Only, Exterior Removed)")
    print(f"Workspace     : {args.zoom_size}x{args.zoom_size} px")
    print("Hotkeys:")
    print("  'v'       : Toggle view (Zoomed -> Side-by-Side -> Full)")
    print("  's'/SPACE : Take a snapshot picture")
    print("  'r'       : Start/stop video recording")
    print("  'q'/ESC   : Quit program")
    print("=" * 65)

    # Open Camera Device
    try:
        cam_idx = int(args.camera)
        cap = cv2.VideoCapture(cam_idx)
    except ValueError:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Error: Could not open camera source '{args.camera}'.")
        print("Please check that your camera or USB device is plugged in.")
        sys.exit(1)

    # Set requested camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Connected to camera successfully: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    # Load neural model
    model = load_best_checkpoint(device, args.checkpoint)

    # Create capture directory for snapshots and recordings
    capture_dir = os.path.join(os.path.dirname(__file__), "..", "captures")
    os.makedirs(capture_dir, exist_ok=True)

    # Recording setup
    writer = None
    manual_recording = False
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Resolution matches view mode
        out_dim = (args.zoom_size, args.zoom_size) if args.view == "zoomed" else (actual_w, actual_h)
        writer = cv2.VideoWriter(args.output, fourcc, 20.0, out_dim)
        print(f"Recording live session to: {args.output}")

    view_mode = args.view
    view_cycle = ["zoomed", "side-by-side", "full"]

    prev_state = {"center": None, "radius": None, "eye_box": None, "consecutive_misses": 0}
    frame_idx = 0
    fps_history = []
    last_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Notice: Camera frame stream ended.")
            break

        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        instant_fps = 1.0 / dt if dt > 0 else 30.0
        fps_history.append(instant_fps)
        if len(fps_history) > 15:
            fps_history.pop(0)
        avg_fps = float(np.mean(fps_history))

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Localize and Smooth Eye Aperture Box
        prev_box = prev_state.get("eye_box")
        ex, ey, ew, eh = detect_eye_crop_box(gray, prev_box=prev_box)
        prev_state["eye_box"] = (ex, ey, ew, eh)

        # 2. Zoom in strictly on the eye opening (remove black housing)
        eye_crop = frame[ey:ey+eh, ex:ex+ew]
        zoomed_eye = cv2.resize(eye_crop, (args.zoom_size, args.zoom_size), interpolation=cv2.INTER_LINEAR)
        zoomed_gray = cv2.cvtColor(zoomed_eye, cv2.COLOR_BGR2GRAY)

        # 3. Detect Pupil inside Zoomed Eye Workspace
        lcx, lcy, lr, (lgx, lgy), lellipse = detect_pupil_in_zoomed_roi(
            zoomed_gray, model=model, device=device, threshold=0.35
        )

        detected = (lcx is not None and lr is not None and lr >= 8)
        scale_x = ew / float(args.zoom_size)
        scale_y = eh / float(args.zoom_size)
        orig_diameter = (lr * 2.0 * ((scale_x + scale_y) / 2.0)) if detected else 0.0

        # 4. Render Shaded Overlay in Zoomed View
        zoomed_display = render_shaded_overlay(
            zoomed_eye, lcx, lcy, lr, fitted_ellipse=lellipse, alpha=0.40
        )

        # Add HUD overlay labels
        cv2.putText(
            zoomed_display,
            f"Pupil D: {orig_diameter:.1f}px" if detected else "Pupil D: Searching...",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            zoomed_display,
            f"Live: {avg_fps:.1f} FPS",
            (20, args.zoom_size - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            zoomed_display,
            f"View: {view_mode.upper()}",
            (args.zoom_size - 180, args.zoom_size - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        if manual_recording:
            cv2.circle(zoomed_display, (args.zoom_size - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(zoomed_display, "REC", (args.zoom_size - 85, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # 5. Format Output Display Canvas
        if view_mode == "zoomed":
            display_frame = zoomed_display
        elif view_mode == "full":
            full_display = frame.copy()
            gcx = int(ex + lcx * scale_x) if detected else None
            gcy = int(ey + lcy * scale_y) if detected else None
            gr = int(lr * ((scale_x + scale_y) / 2.0)) if detected else None
            cv2.rectangle(full_display, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            cv2.putText(full_display, "Eye Region", (ex, max(20, ey - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            full_display = render_shaded_overlay(full_display, gcx, gcy, gr, alpha=0.40)
            cv2.putText(full_display, f"Pupil D: {orig_diameter:.1f}px" if detected else "Searching...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(full_display, f"Live: {avg_fps:.1f} FPS", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)
            display_frame = full_display
        else:
            # Side-by-side mode
            canvas = np.zeros((args.zoom_size, args.zoom_size * 2, 3), dtype=np.uint8)
            canvas[:, :args.zoom_size] = zoomed_display
            full_small = cv2.resize(frame, (args.zoom_size, args.zoom_size))
            canvas[:, args.zoom_size:] = full_small
            cv2.line(canvas, (args.zoom_size, 0), (args.zoom_size, args.zoom_size), (80, 80, 80), 2)
            display_frame = canvas

        # Save to video writer if active
        if writer is not None:
            writer.write(display_frame)

        # GUI Window Display & Key Handling
        if not args.no_show:
            cv2.imshow("E-Eye Live Pupil Capture & Tracking", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Exit requested by user.")
                break
            elif key == ord("v"):
                # Cycle view
                curr_idx = view_cycle.index(view_mode)
                view_mode = view_cycle[(curr_idx + 1) % len(view_cycle)]
                print(f"Switched view to: {view_mode.upper()}")
            elif key in (ord("s"), ord(" ")):
                # Snapshot
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(capture_dir, f"snapshot_{ts}.png")
                cv2.imwrite(snap_path, display_frame)
                print(f"Saved snapshot -> {snap_path}")
            elif key == ord("r"):
                # Toggle manual recording
                manual_recording = not manual_recording
                if manual_recording:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join(capture_dir, f"live_record_{ts}.mp4")
                    rec_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    rec_dim = (display_frame.shape[1], display_frame.shape[0])
                    writer = cv2.VideoWriter(rec_path, rec_fourcc, 20.0, rec_dim)
                    print(f"Started recording -> {rec_path}")
                else:
                    if writer is not None:
                        writer.release()
                        writer = None
                    print("Stopped recording.")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("=" * 65)
    print("LIVE CAPTURE CLOSED")
    print(f"Total Frames Streamed: {frame_idx}")
    print("=" * 65)


if __name__ == "__main__":
    main()
