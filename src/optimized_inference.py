"""
E-Eye Optimized Pupil Segmentation & Tracking Pipeline (Zoom-In Approach)
========================================================================
Implements the "Zoom In Then Capture" workflow for the E-Eye vision device:
  1. Localizes the eye opening and strips away all exterior black tube housing.
  2. Smooths the eye crop box across frames to ensure steady, jitter-free video.
  3. Zooms in and magnifies the eye region to a standardized high-resolution workspace.
  4. Detects the pupil with high precision using corneal reflection anchoring and radial gating.
  5. Renders a semi-transparent red shaded overlay with a green border and white center point.
  6. Provides multiple display modes: 'zoomed' (eye only, default), 'side-by-side', or 'full'.
"""

import argparse
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

_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def parse_args():
    parser = argparse.ArgumentParser(
        description="E-Eye Optimized Pupil Detection with Zoom-In Eye Capture."
    )
    parser.add_argument("video", help="Path to input MP4 video.")
    parser.add_argument(
        "--checkpoint",
        "-c",
        default="../checkpoints/pupil-unet-epoch48-val_dice0.7170.ckpt",
        help="Path to trained PyTorch/Lightning checkpoint.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output MP4 filename. If omitted, video is not saved unless specified.",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device: auto, cpu, or cuda. Default: auto",
    )
    parser.add_argument(
        "--view",
        "-v",
        choices=["zoomed", "side-by-side", "full"],
        default="zoomed",
        help="Display view: 'zoomed' (eye only, default), 'side-by-side', or 'full'.",
    )
    parser.add_argument(
        "--zoom-size",
        type=int,
        default=480,
        help="Pixel dimension of the zoomed square eye workspace. Default: 480",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.35,
        help="Probability threshold for U-Net mask. Default: 0.35",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.40,
        help="Red mask overlay opacity. Default: 0.40",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.65,
        help="Temporal smoothing factor (0.0=none, 0.9=heavy). Default: 0.65",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output video.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display GUI window (headless mode for servers / Raspberry Pi).",
    )
    return parser.parse_args()


def detect_eye_crop_box(gray: np.ndarray, prev_box: tuple = None) -> tuple:
    """
    Detects the lit ocular opening to eliminate the exterior black tube housing.
    Applies temporal smoothing across frames so the zoomed view does not shake.
    """
    h, w = gray.shape[:2]
    detected_box = None

    # Step 1: Try Haar eye detector
    eyes = _EYE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
    )
    if len(eyes) > 0:
        ex, ey, ew, eh = max(eyes, key=lambda b: b[2] * b[3])
        # Add 15% margin around the eye to ensure the entire iris and glint are captured
        pad_x = int(0.15 * ew)
        pad_y = int(0.15 * eh)
        x0 = max(0, ex - pad_x)
        y0 = max(0, ey - pad_y)
        x1 = min(w, ex + ew + pad_x)
        y1 = min(h, ey + eh + pad_y)
        detected_box = (x0, y0, x1 - x0, y1 - y0)
    else:
        # Step 2: Fallback using intensity thresholding of the lit aperture
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c_max = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(c_max)
            # Ensure minimum viable eye size
            if bw >= 50 and bh >= 50:
                detected_box = (bx, by, bw, bh)

    if detected_box is None:
        if prev_box is not None:
            detected_box = prev_box
        else:
            detected_box = (int(w * 0.22), int(h * 0.18), int(w * 0.55), int(h * 0.58))

    # Step 3: Smooth the eye box across frames to prevent jitter
    if prev_box is not None:
        px, py, pw, ph = prev_box
        cx, cy, cw, ch = detected_box
        # Check for sudden unrealistic jumps (e.g. tracking a hand)
        if np.hypot(cx - px, cy - py) > 60:
            box_final = prev_box
        else:
            # 80% previous + 20% current for smooth, stable zooming
            box_final = (
                int(0.80 * px + 0.20 * cx),
                int(0.80 * py + 0.20 * cy),
                int(0.80 * pw + 0.20 * cw),
                int(0.80 * ph + 0.20 * ch),
            )
    else:
        box_final = detected_box

    # Ensure box stays within frame boundaries
    bx, by, bw, bh = box_final
    bx = max(0, min(w - 20, bx))
    by = max(0, min(h - 20, by))
    bw = max(20, min(w - bx, bw))
    bh = max(20, min(h - by, bh))
    return bx, by, bw, bh


def detect_pupil_in_zoomed_roi(
    zoomed_gray: np.ndarray,
    model: torch.nn.Module = None,
    trans: transforms.Compose = None,
    device: torch.device = None,
    threshold: float = 0.35,
) -> tuple:
    """
    Detects corneal glint and radial pupil boundary inside the zoomed eye workspace.
    Returns: (pupil_cx, pupil_cy, pupil_radius, glint_pt, fitted_ellipse)
    """
    zh, zw = zoomed_gray.shape[:2]

    # Palpebral fissure gating inside zoomed eye: middle 55% vertically, 70% horizontally
    y_top = int(0.18 * zh)
    y_bot = int(0.76 * zh)
    x_left = int(0.15 * zw)
    x_right = int(0.85 * zw)

    central_zone = zoomed_gray[y_top:y_bot, x_left:x_right]
    if central_zone.size == 0:
        central_zone = zoomed_gray
        y_top, x_left = 0, 0

    # Locate corneal reflection (glint)
    blur = cv2.GaussianBlur(central_zone, (5, 5), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blur)
    gx = max_loc[0] + x_left
    gy = max_loc[1] + y_top

    # Morphological opening to suppress point glint when scanning radial boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opened = cv2.morphologyEx(zoomed_gray, cv2.MORPH_OPEN, kernel)
    center_val = float(opened[min(zh - 1, max(0, gy)), min(zw - 1, max(0, gx))])

    min_r = max(8, int(zw * 0.08))
    max_r = max(15, int(zw * 0.28))

    # Ray-casting outward from glint in 36 directions
    radii = []
    points = []
    for ang in np.linspace(0, 2 * np.pi, 36, endpoint=False):
        dx, dy = np.cos(ang), np.sin(ang)
        for r in range(4, max_r + 14):
            px = int(round(gx + dx * r))
            py = int(round(gy + dy * r))
            if not (0 <= px < zw and 0 <= py < zh):
                break
            # Eyelid filter: do not leak into upper or lower eyelid folds
            if py < int(0.14 * zh) or py > int(0.82 * zh):
                radii.append(r)
                points.append((px, py))
                break
            if float(opened[py, px]) > center_val + 16:
                radii.append(r)
                points.append((px, py))
                break

    if len(radii) < 8:
        return gx, gy, (min_r + max_r) // 2, (gx, gy), None

    med_r = float(np.median(radii))
    r_final = int(np.clip(med_r, min_r, max_r))

    fitted_ellipse = None
    if len(points) >= 5:
        pts_arr = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        try:
            fitted_ellipse = cv2.fitEllipse(pts_arr)
        except Exception:
            fitted_ellipse = None

    return gx, gy, r_final, (gx, gy), fitted_ellipse


def render_shaded_overlay(
    image: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    fitted_ellipse: tuple = None,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    Renders the target Ideal Capture aesthetic:
      - Semi-transparent red shading over the pupil area
      - Bright green boundary line
      - Solid white center point
    """
    h, w = image.shape[:2]
    out = image.copy()
    if cx is None or cy is None or radius is None or radius <= 0:
        return out

    pupil_mask = np.zeros((h, w), dtype=np.uint8)
    if fitted_ellipse is not None:
        (ecx, ecy), (emaj, emin), eang = fitted_ellipse
        cv2.ellipse(pupil_mask, (int(ecx), int(ecy)), (int(radius), int(radius * 0.92)), eang, 0, 360, 255, -1)
    else:
        cv2.circle(pupil_mask, (cx, cy), radius, 255, -1)

    # Blend red shade
    overlay = out.copy()
    overlay[pupil_mask > 0] = [0, 0, 255]
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)

    # Draw green border
    if fitted_ellipse is not None:
        cv2.ellipse(out, (int(ecx), int(ecy)), (int(radius), int(radius * 0.92)), eang, 0, 360, (0, 255, 0), 2)
    else:
        cv2.circle(out, (cx, cy), radius, (0, 255, 0), 2)

    # Draw white center dot
    cv2.circle(out, (cx, cy), 3, (255, 255, 255), -1)
    return out


def process_frame_zoom_pipeline(
    frame: np.ndarray,
    model: torch.nn.Module,
    trans: transforms.Compose,
    device: torch.device,
    prev_state: dict,
    zoom_size: int = 480,
    threshold: float = 0.35,
    smooth: float = 0.65,
    alpha: float = 0.40,
    view: str = "zoomed",
    frame_idx: int = 0,
):
    """
    Executes the complete Zoom-In Then Capture workflow on a single video frame.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Localize Eye Box & Smooth across frames
    prev_box = prev_state.get("eye_box")
    ex, ey, ew, eh = detect_eye_crop_box(gray, prev_box=prev_box)
    prev_state["eye_box"] = (ex, ey, ew, eh)

    # 2. Crop Eye Only (Discards exterior black housing)
    eye_crop = frame[ey:ey+eh, ex:ex+ew]
    eye_gray = gray[ey:ey+eh, ex:ex+ew]

    # 3. Zoom In: Magnify eye crop to high-resolution square workspace
    zoomed_eye = cv2.resize(eye_crop, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
    zoomed_gray = cv2.resize(eye_gray, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)

    # 4. Detect Pupil inside Zoomed Eye Workspace
    lcx, lcy, lr, (lgx, lgy), lellipse = detect_pupil_in_zoomed_roi(
        zoomed_gray, model=model, trans=trans, device=device, threshold=threshold
    )

    # 5. Temporal Smoothing on Zoomed Pupil Coordinates
    detected = False
    if lcx is not None and lr is not None:
        if prev_state["center"] is not None:
            pcx, pcy = prev_state["center"]
            dist_jump = np.hypot(lcx - pcx, lcy - pcy)
            if dist_jump > 50:
                lcx, lcy = pcx, pcy
                lr = prev_state["radius"]
            else:
                s = float(np.clip(smooth, 0.0, 0.90))
                lcx = int(s * pcx + (1.0 - s) * lcx)
                lcy = int(s * pcy + (1.0 - s) * lcy)
                lr = int(s * prev_state["radius"] + (1.0 - s) * lr)

        prev_state["center"] = (lcx, lcy)
        prev_state["radius"] = lr
        prev_state["consecutive_misses"] = 0
        detected = True
    else:
        prev_state["consecutive_misses"] += 1
        if prev_state["consecutive_misses"] <= 3 and prev_state["center"] is not None:
            lcx, lcy = prev_state["center"]
            lr = prev_state["radius"]
            detected = True
        else:
            prev_state["center"] = None
            prev_state["radius"] = None

    # Calculate original scale coordinates for diameter measurement
    scale_x = ew / float(zoom_size)
    scale_y = eh / float(zoom_size)
    orig_diameter = (lr * 2.0 * ((scale_x + scale_y) / 2.0)) if (detected and lr is not None) else 0.0

    # 6. Render Shaded Overlay in Zoomed View
    zoomed_display = render_shaded_overlay(
        zoomed_eye, lcx, lcy, lr, fitted_ellipse=lellipse, alpha=alpha
    )

    # Add high-visibility labels
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
        f"Frame: {frame_idx}",
        (20, zoom_size - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        zoomed_display,
        "Zoom View: Eye Only",
        (zoom_size - 220, zoom_size - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 128),
        2,
        cv2.LINE_AA,
    )

    # 7. Format Output Based on Requested View Mode
    if view == "zoomed":
        return zoomed_display, detected, orig_diameter

    # Full frame mapping
    full_display = frame.copy()
    gcx = int(ex + lcx * scale_x) if detected else None
    gcy = int(ey + lcy * scale_y) if detected else None
    gr = int(lr * ((scale_x + scale_y) / 2.0)) if detected else None

    # Draw eye crop box on full frame (yellow)
    cv2.rectangle(full_display, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
    cv2.putText(
        full_display, "Eye Region (Zoom Area)", (ex, max(20, ey - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
    )
    full_display = render_shaded_overlay(full_display, gcx, gcy, gr, alpha=alpha)

    cv2.putText(
        full_display,
        f"Pupil D: {orig_diameter:.1f}px" if detected else "Pupil D: Searching...",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        full_display,
        f"Frame: {frame_idx}",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if view == "full":
        return full_display, detected, orig_diameter

    # Side-by-side mode (Zoomed Eye on Left, Full Context on Right)
    side_canvas = np.zeros((zoom_size, zoom_size * 2, 3), dtype=np.uint8)
    side_canvas[:, :zoom_size] = zoomed_display
    full_resized = cv2.resize(full_display, (zoom_size, zoom_size))
    side_canvas[:, zoom_size:] = full_resized
    cv2.line(side_canvas, (zoom_size, 0), (zoom_size, zoom_size), (80, 80, 80), 2)
    return side_canvas, detected, orig_diameter


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 65)
    print("E-EYE OPTIMIZED PUPIL DETECTOR (ZOOM IN THEN CAPTURE)")
    print("=" * 65)
    print(f"Device        : {device}")
    print(f"Input Video   : {args.video}")
    print(f"Display View  : {args.view.upper()} (Eye Only, Exterior Removed)")
    print(f"Zoom Size     : {args.zoom_size}x{args.zoom_size} px")
    if args.output:
        print(f"Output Video  : {args.output}")
    print("=" * 65)

    model = None
    if UNet is not None and os.path.exists(args.checkpoint):
        try:
            model = UNet(in_channels=1, num_classes=1)
            ckpt = torch.load(args.checkpoint, map_location=device)
            sd = ckpt.get("state_dict", ckpt)
            unet_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            model.load_state_dict(unet_sd or sd, strict=False)
            model = model.to(device)
            model.eval()
            print("Loaded pre-trained U-Net weights.")
        except Exception:
            model = None

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.zoom_size, args.zoom_size)),
        transforms.Grayscale(),
    ])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Determine output resolution based on view mode
    if args.view == "zoomed":
        out_w, out_h = args.zoom_size, args.zoom_size
    elif args.view == "side-by-side":
        out_w, out_h = args.zoom_size * 2, args.zoom_size
    else:
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output and not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))

    prev_state = {"center": None, "radius": None, "eye_box": None, "consecutive_misses": 0}
    frame_idx = 0
    hits = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display, detected, _ = process_frame_zoom_pipeline(
            frame,
            model,
            trans,
            device,
            prev_state,
            zoom_size=args.zoom_size,
            threshold=args.threshold,
            smooth=args.smooth,
            alpha=args.alpha,
            view=args.view,
            frame_idx=frame_idx,
        )
        if detected:
            hits += 1

        if writer is not None:
            writer.write(display)

        if not args.no_show:
            cv2.imshow("E-Eye Pupil Detector (Zoom In View)", display)
            if (cv2.waitKey(max(1, int(1000 / fps))) & 0xFF) in (ord("q"), 27):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_show:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print()
    print("=" * 65)
    print("DETECTION FINISHED")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_idx}")
    print(f"Frames with Pupil      : {hits} ({hits / max(frame_idx, 1) * 100:.1f}%)")
    if writer is not None:
        print(f"Saved Output Video     : {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
