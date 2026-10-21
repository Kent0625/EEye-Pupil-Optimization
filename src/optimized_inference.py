"""
E-Eye Optimized Pupil Segmentation & Tracking Pipeline
======================================================
Addresses false positive detections on eyelids, eyelashes, and camera housing
by combining:
  1. Deep U-Net probability segmentation
  2. Corneal glint (Purkinje reflection) detection & co-occurrence
  3. Morphological opening (eliminates thin horizontal eyelid slivers)
  4. Geometric scoring (aspect ratio, circularity, solidity)
  5. Inter-frame temporal smoothing (exponential moving average)
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
from unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="E-Eye Optimized Pupil Segmentation with Eyelid Rejection."
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
        default="optimized_pupil_output.mp4",
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
        default=0.40,
        help="Base probability threshold for U-Net mask. Default: 0.40",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Mask overlay alpha opacity. Default: 0.35",
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
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="Temporal smoothing factor (0.0=none, 0.9=heavy). Default: 0.6",
    )
    return parser.parse_args()


def detect_corneal_glints(gray: np.ndarray, top_k: int = 3):
    """
    Find brightest specular reflections (corneal glints) in the eye frame.
    In eccentric photorefraction, the IR LED array produces bright reflections
    on the cornea within or adjacent to the pupil aperture.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Glints in IR photorefraction are usually in the top 99th intensity percentile
    min_val, max_val, _, max_loc = cv2.minMaxLoc(blur)
    if max_val < 160:
        return []

    thresh_val = max(180, int(max_val * 0.88))
    _, bright_mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    glints = []
    for c in contours:
        area = cv2.contourArea(c)
        if 2 <= area <= 400:
            m = cv2.moments(c)
            if m["m00"] > 0:
                gx = int(m["m10"] / m["m00"])
                gy = int(m["m01"] / m["m00"])
                glints.append((gx, gy, float(blur[gy, gx])))

    glints.sort(key=lambda g: g[2], reverse=True)
    return glints[:top_k]


def filter_eyelid_false_positives(
    prob_map: np.ndarray,
    gray: np.ndarray,
    threshold: float = 0.40,
    glints: list = None
):
    """
    Evaluates candidate pupil regions and rejects eyelid/eyelash artifacts.
    Returns (best_contour, best_metrics, clean_mask) or (None, None, clean_mask).
    """
    h, w = prob_map.shape
    raw_mask = (prob_map >= threshold).astype(np.uint8) * 255

    # Morphological opening with elliptical kernel:
    # Eyelashes and horizontal eyelid folds are thin and get severed/removed,
    # while the circular pupil body is preserved.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, clean_mask

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # Pupils in standard framing typically occupy 100 to 12000 pixels
        if area < 80 or area > 14000:
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        # Circularity: 4*pi*area / perimeter^2  (Circle = 1.0, horizontal slit < 0.3)
        circularity = (4.0 * np.pi * area) / (peri * peri)

        # Convex Hull Solidity: Area / HullArea (Pupil ~ 0.9+, eyelid notch < 0.7)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect_ratio = float(bw) / max(bh, 1)

        # REJECTION RULE 1: Extreme horizontal elongation (Eyelid fold)
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            continue

        # REJECTION RULE 2: Jagged non-circular perimeter
        if circularity < 0.32:
            continue

        # REJECTION RULE 3: Highly concave shape
        if solidity < 0.65:
            continue

        # Region confidence from U-Net probability
        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, -1)
        mean_prob = cv2.mean(prob_map, mask=c_mask)[0]

        # Glint proximity bonus: pupil should contain or be adjacent to the Purkinje glint
        glint_bonus = 0.0
        if glints:
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                min_dist = min(np.hypot(cx - gx, cy - gy) for gx, gy, _ in glints)
                # If glint is inside or near the pupil radius
                expected_r = np.sqrt(area / np.pi)
                if min_dist <= expected_r * 1.5:
                    glint_bonus = 1.2 * max(0.0, 1.0 - (min_dist / (expected_r * 1.5)))

        # Composite score
        score = (
            mean_prob * 2.0
            + circularity * 1.5
            + solidity * 1.0
            + glint_bonus
            - abs(aspect_ratio - 1.0) * 0.6
        )

        metrics = {
            "area": area,
            "circularity": circularity,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "mean_prob": mean_prob,
            "glint_bonus": glint_bonus,
            "score": score,
        }
        candidates.append((score, c, metrics))

    if not candidates:
        return None, None, clean_mask

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_contour, best_metrics = candidates[0]
    return best_contour, best_metrics, clean_mask


def fit_stable_ellipse(contour: np.ndarray, prev_ellipse: tuple = None, smooth: float = 0.6):
    """
    Fits an ellipse to the contour and applies temporal smoothing.
    Returns smoothed ((cx, cy), (axis1, axis2), angle).
    """
    if len(contour) < 5:
        return None

    raw_ellipse = cv2.fitEllipse(contour)
    (cx, cy), (ax1, ax2), angle = raw_ellipse

    # Safety: check axes sanity
    major = max(ax1, ax2)
    minor = min(ax1, ax2)
    if minor <= 1 or major <= 1:
        return None

    # Eyelid clipping can distort ratio; limit extreme minor/major flattening
    if minor / major < 0.45:
        minor = major * 0.75

    curr = ((cx, cy), (major, minor), angle)

    if prev_ellipse is None or smooth <= 0.0:
        return curr

    # Smooth center, axes, and angle
    (pcx, pcy), (p_maj, p_min), p_ang = prev_ellipse
    s = float(np.clip(smooth, 0.0, 0.95))

    scx = s * pcx + (1.0 - s) * cx
    scy = s * pcy + (1.0 - s) * cy
    s_maj = s * p_maj + (1.0 - s) * major
    s_min = s * p_min + (1.0 - s) * minor
    s_ang = s * p_ang + (1.0 - s) * angle

    return ((scx, scy), (s_maj, s_min), s_ang)


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
    print("E-EYE OPTIMIZED PUPIL SEGMENTATION & TRACKING")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Video       : {args.video}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Threshold   : {args.threshold}")
    print(f"Smoothing   : {args.smooth}")
    print("=" * 60)

    # Load Model
    model = UNet(in_channels=1, num_classes=1)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print("Loading checkpoint weights...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    unet_sd = {
        k[len("model."):]: v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    if not unet_sd:
        unet_sd = state_dict
    model.load_state_dict(unet_sd, strict=False)
    model = model.to(device)
    model.eval()
    print("Model initialized successfully.")

    # Image Transform
    width, height = 640, 480
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width)),
        transforms.Grayscale(),
    ])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Recording output to: {args.output}")

    frame_idx = 0
    detected_count = 0
    prev_ellipse = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (width, height))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        x = trans(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        glints = detect_corneal_glints(gray)
        best_contour, metrics, clean_mask = filter_eyelid_false_positives(
            prob_map, gray, threshold=args.threshold, glints=glints
        )

        display = frame_resized.copy()

        # Draw red segmentation overlay
        overlay = display.copy()
        overlay[clean_mask > 0] = [0, 0, 255]
        display = cv2.addWeighted(overlay, args.alpha, display, 1.0 - args.alpha, 0)

        # Draw detected glints in yellow dots
        for gx, gy, _ in glints:
            cv2.circle(display, (gx, gy), 2, (0, 255, 255), -1)

        pupil_diameter = 0.0
        if best_contour is not None:
            detected_count += 1
            # Draw green contour around pupil
            cv2.drawContours(display, [best_contour], -1, (0, 255, 0), 2)

            smoothed_ellipse = fit_stable_ellipse(best_contour, prev_ellipse, args.smooth)
            if smoothed_ellipse is not None:
                prev_ellipse = smoothed_ellipse
                (cx, cy), (maj, min_ax), ang = smoothed_ellipse
                cv2.ellipse(
                    display,
                    ((cx, cy), (maj, min_ax), ang),
                    (0, 255, 128),
                    2,
                )
                cv2.circle(display, (int(cx), int(cy)), 3, (255, 255, 255), -1)
                pupil_diameter = np.sqrt(maj * min_ax)

        # HUD Overlay
        cv2.rectangle(display, (12, 12), (320, 85), (20, 20, 20), -1)
        cv2.rectangle(display, (12, 12), (320, 85), (70, 70, 70), 1)

        status_text = (
            f"Pupil D: {pupil_diameter:.1f}px" if best_contour is not None else "Searching..."
        )
        status_color = (0, 255, 0) if best_contour is not None else (0, 165, 255)

        cv2.putText(
            display,
            f"E-Eye Status: {status_text}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA,
        )

        if metrics:
            diag_str = f"Circ: {metrics['circularity']:.2f} | Solid: {metrics['solidity']:.2f}"
            cv2.putText(
                display,
                diag_str,
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            display,
            f"Frame: {frame_idx}/{total_frames}",
            (20, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(display)

        if not args.no_show:
            cv2.imshow("E-Eye Pupil Optimization", display)
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
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Total Frames Processed : {frame_idx}")
    print(f"Pupil Detected Frames  : {detected_count} ({detected_count / max(frame_idx, 1) * 100:.1f}%)")
    if not args.no_save:
        print(f"Output Video Saved     : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
