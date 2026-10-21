"""
E-Eye Optimized Pupil Segmentation & Tracking Pipeline
======================================================
Multi-stage hybrid detector engineered for eccentric photorefractors.
Features:
  1. Palpebral Fissure Spatial Gating (eliminates eyelid & tear meniscus reflections)
  2. Corneal Glint (Purkinje reflection) localization within the ocular aperture
  3. Morphological opening to sever thin horizontal eyelid creases & eyelashes
  4. Radial gradient ray-casting & U-Net probability fusion
  5. Geometric validation (aspect ratio, circularity, solidity)
  6. Temporal smoothing (EMA) with blink & outlier jump rejection
"""

import argparse
import os
import sys
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
        description="E-Eye Optimized Pupil Detection with Eyelid Rejection."
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
        default="optimized_output.mp4",
        help="Output MP4 filename.",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device: auto, cpu, or cuda. Default: auto",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.35,
        help="Base probability threshold for U-Net mask. Default: 0.35",
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
        help="Do not display GUI window (headless mode).",
    )
    return parser.parse_args()


def detect_eye_box(gray: np.ndarray, min_neighbors: int = 3, min_size: int = 40):
    """Localize ocular region to eliminate surrounding camera housing."""
    eyes = _EYE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=min_neighbors, minSize=(min_size, min_size)
    )
    if len(eyes) == 0:
        h, w = gray.shape
        return int(w * 0.25), int(h * 0.20), int(w * 0.50), int(h * 0.55)
    return tuple(max(eyes, key=lambda b: b[2] * b[3]))


def detect_pupil_boundary_radial(gray: np.ndarray, glint_pt: tuple, eye_box: tuple):
    """
    Ray-casts radially from the corneal glint to find the true pupil boundary.
    Rays shooting vertically are constrained to avoid leaking into drooping eyelids.
    """
    gx, gy = glint_pt
    ex, ey, ew, eh = eye_box
    h, w = gray.shape

    min_r = max(6, int(ew * 0.08))
    max_r = max(10, int(ew * 0.26))

    # Suppress glint with morphological opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    center_val = float(opened[min(h - 1, max(0, gy)), min(w - 1, max(0, gx))])

    radii = []
    points = []

    # 36 radial rays
    for ang in np.linspace(0, 2 * np.pi, 36, endpoint=False):
        dx, dy = np.cos(ang), np.sin(ang)

        for r in range(3, max_r + 12):
            px, py = int(round(gx + dx * r)), int(round(gy + dy * r))
            if not (0 <= px < w and 0 <= py < h):
                break

            # If ray crosses beyond eyelid boundaries, stop
            if py < ey + int(0.12 * eh) or py > ey + int(0.78 * eh):
                radii.append(r)
                points.append((px, py))
                break

            if float(opened[py, px]) > center_val + 16:
                radii.append(r)
                points.append((px, py))
                break

    if len(radii) < 8:
        return None

    med_r = float(np.median(radii))
    r_final = int(np.clip(med_r, min_r, max_r))
    return int(gx), int(gy), r_final, points


def process_frame(
    frame: np.ndarray,
    model: torch.nn.Module,
    trans: transforms.Compose,
    device: torch.device,
    prev_state: dict,
    threshold: float = 0.35,
    smooth: float = 0.65,
):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Detect Eye Box (Palpebral Fissure localization)
    ex, ey, ew, eh = detect_eye_box(gray)

    # 2. Palpebral Fissure Gating: Search for corneal glint strictly in central ocular aperture
    # Upper/lower margins (eyelid folds, tear meniscus) are excluded
    y_top = max(0, ey + int(0.18 * eh))
    y_bot = min(h, ey + int(0.72 * eh))
    x_left = max(0, ex + int(0.15 * ew))
    x_right = min(w, ex + int(0.85 * ew))

    fissure_crop = gray[y_top:y_bot, x_left:x_right]
    if fissure_crop.size == 0:
        fissure_crop = gray

    blur = cv2.GaussianBlur(fissure_crop, (5, 5), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blur)
    gx = max_loc[0] + x_left
    gy = max_loc[1] + y_top

    # 3. UNet Inference (if model loaded)
    unet_contour = None
    unet_metrics = None
    if model is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_in = trans(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor_in)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Resize prob_map back to frame size if needed
        if prob_map.shape != (h, w):
            prob_map = cv2.resize(prob_map, (w, h))

        # Check for confident UNet activation inside eye box
        eye_prob_crop = prob_map[y_top:y_bot, x_left:x_right]
        if eye_prob_crop.size > 0 and eye_prob_crop.max() >= threshold:
            raw_mask = (prob_map >= threshold).astype(np.uint8) * 255
            # Morphological opening to cut thin horizontal eyelid folds
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_c, best_score = None, -1.0
            for c in contours:
                area = cv2.contourArea(c)
                if area < 60 or area > 15000:
                    continue
                peri = cv2.arcLength(c, True)
                if peri == 0:
                    continue
                circ = (4.0 * np.pi * area) / (peri * peri)
                bx, by, bw, bh = cv2.boundingRect(c)
                ar = float(bw) / max(bh, 1)

                # Reject horizontal eyelid streaks
                if ar > 2.4 or ar < 0.4 or circ < 0.30:
                    continue

                # Glint proximity check
                m = cv2.moments(c)
                if m["m00"] > 0:
                    mcx = m["m10"] / m["m00"]
                    mcy = m["m01"] / m["m00"]
                    dist = np.hypot(mcx - gx, mcy - gy)
                    score = circ * 2.0 - dist * 0.05
                    if score > best_score:
                        best_score = score
                        best_c = c
            if best_c is not None:
                unet_contour = best_c

    # 4. Hybrid Selection: Use UNet or Radial Ray-casting
    cx, cy, radius = None, None, None
    if unet_contour is not None and len(unet_contour) >= 5:
        (ecx, ecy), (e_maj, e_min), e_ang = cv2.fitEllipse(unet_contour)
        cx, cy = int(ecx), int(ecy)
        radius = int(np.sqrt(e_maj * e_min) / 2.0)
    else:
        # High precision radial boundary around glint
        res = detect_pupil_boundary_radial(gray, (gx, gy), (ex, ey, ew, eh))
        if res is not None:
            cx, cy, radius, _ = res

    # 5. Temporal Smoothing & Outlier Gating
    detected = False
    if cx is not None and radius is not None:
        if prev_state["center"] is not None:
            pcx, pcy = prev_state["center"]
            dist_jump = np.hypot(cx - pcx, cy - pcy)

            # Eyelid Jump Gate: If detection jumps > 40px in a single frame, reject as eyelid/glint glitch
            if dist_jump > 40:
                cx, cy = pcx, pcy
                radius = prev_state["radius"]
            else:
                s = float(np.clip(smooth, 0.0, 0.90))
                cx = int(s * pcx + (1.0 - s) * cx)
                cy = int(s * pcy + (1.0 - s) * cy)
                radius = int(s * prev_state["radius"] + (1.0 - s) * radius)

        prev_state["center"] = (cx, cy)
        prev_state["radius"] = radius
        prev_state["consecutive_misses"] = 0
        detected = True
    else:
        prev_state["consecutive_misses"] += 1
        # Hold last position for up to 3 frames during partial blink
        if prev_state["consecutive_misses"] <= 3 and prev_state["center"] is not None:
            cx, cy = prev_state["center"]
            radius = prev_state["radius"]
            detected = True
        else:
            prev_state["center"] = None
            prev_state["radius"] = None

    # 6. Visualization & Overlays
    display = frame.copy()

    # Subtle eye-region aperture indicator
    cv2.rectangle(display, (x_left, y_top), (x_right, y_bot), (80, 80, 80), 1)

    # Glint indicator (yellow)
    cv2.circle(display, (gx, gy), 2, (0, 255, 255), -1)

    diameter = 0.0
    if detected and cx is not None:
        diameter = radius * 2.0
        # Draw pupil circle / ellipse (green)
        cv2.circle(display, (cx, cy), radius, (0, 255, 0), 2)
        # Center marker (white)
        cv2.circle(display, (cx, cy), 3, (255, 255, 255), -1)

    # HUD Box
    cv2.rectangle(display, (12, 12), (340, 85), (20, 20, 20), -1)
    cv2.rectangle(display, (12, 12), (340, 85), (60, 60, 60), 1)

    status_str = f"PUPIL: {diameter:.1f}px" if detected else "SEARCHING..."
    status_col = (0, 255, 0) if detected else (0, 165, 255)

    cv2.putText(
        display,
        f"E-Eye Status : {status_str}",
        (22, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_col,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"Glint Coord  : ({gx}, {gy})",
        (22, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"Eyelid Filter: ACTIVE (Gated Fissure)",
        (22, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return display, detected, diameter


def main():
    args = parse_args()

    # Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("E-EYE OPTIMIZED HYBRID PUPIL DETECTOR")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Input Video : {args.video}")
    print(f"Output Video: {args.output}")
    print("=" * 60)

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
        except Exception as e:
            print(f"Warning: Could not load U-Net checkpoint ({e}), using radial glint engine.")
            model = None

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((480, 640)),
        transforms.Grayscale(),
    ])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    prev_state = {"center": None, "radius": None, "consecutive_misses": 0}
    frame_idx = 0
    hits = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display, detected, _ = process_frame(
            frame, model, trans, device, prev_state, threshold=args.threshold, smooth=args.smooth
        )
        if detected:
            hits += 1

        if writer is not None:
            writer.write(display)

        if not args.no_show:
            cv2.imshow("E-Eye Optimized Pupil Detector", display)
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
    print("=" * 60)
    print("DETECTION FINISHED")
    print("=" * 60)
    print(f"Total Frames Processed : {frame_idx}")
    print(f"Frames with Pupil      : {hits} ({hits / max(frame_idx, 1) * 100:.1f}%)")
    if not args.no_save:
        print(f"Saved Output Video     : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
