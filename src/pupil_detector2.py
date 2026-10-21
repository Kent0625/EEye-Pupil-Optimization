"""
EXPERIMENTAL pupil detector — for footage where pupil_detector.py's plain
full-frame Hough search fails or misfires.

Why a second file: plain Hough-on-the-whole-frame has a sharp failure mode
once the frame contains much more than just skin + eye — e.g. a zoomed-out
eye-camera rig where a lot of the surrounding housing (hard edges, shadows,
corners) is visible. Hough circle detection will confidently return SOME
circle from all that clutter — scored by darkness * radius, a dark shadowed
housing corner can easily outscore the much smaller, lower-contrast real
pupil — and there is no "it found nothing" signal to fall back from, because
it always finds something. Tighter radius bounds only help if you already
know how large the eye appears in frame, which varies a lot between setups.

Approach, per-frame:
  1. Run OpenCV's built-in Haar cascade eye detector (haarcascade_eye.xml,
     shipped with opencv-python — no extra download) on the full frame. This
     is a real trained classifier for "does this look like an eye", so it's
     far more robust to surrounding clutter than any hand-tuned brightness/
     shape heuristic: it generalizes across the very different lighting and
     framing seen across test footage so far, whereas threshold-based
     heuristics broke on nearly every new example.
  2. If an eye is found, crop to it (with padding) and search for the pupil
     ONLY inside that crop — this is what actually fixes the housing-false-
     positive problem, since the clutter is simply no longer in view. The
     search radius is scaled to the detected eye's size, rather than a fixed
     pixel range, so it adapts to how zoomed-in the footage is:
       a. Try Hough circles within the crop first (accurate when the pupil
          boundary has decent contrast).
       b. If that finds nothing, fall back to the brightest-spot-in-crop
          (corneal glint) method from the previous version of this file —
          still useful when a reflection breaks up the pupil's edge.
  3. If Haar finds no eye at all (unusual angle, extreme close-up, poor
     image quality), fall back to the plain full-frame methods: Hough first,
     then the old "widest dark blob" eye-slit crop + glint. These cover the
     original two problem cases this file was first built for, as a safety
     net, but are no longer the primary path.

Usage is identical to pupil_detector.py:
    python pupil_detector_experimental.py input.mp4 output.mp4
    python pupil_detector_experimental.py image.png output.png
    python pupil_detector_experimental.py 0                 # live webcam

Tuning:
    --min-neighbors   Haar cascade strictness (default 3). Raise if it's
                       finding a false "eye" somewhere in the housing;
                       lower if it's missing a real eye.
    --eye-min-size    Minimum eye box size in px the cascade will consider
                       (default 40). Lower this for footage where the eye
                       appears small/far from the camera.
    --debug           Prints what each stage found; saves debug_eye_crop.png
                       for frame 0/the image.

Requires: pip install opencv-python numpy
"""
import argparse
import shutil
import subprocess
import sys
import cv2
import numpy as np

# Reuse the exact full-frame Hough stage and overlay styling from the main
# script, as the fallback-of-last-resort path (see module docstring, step 3).
from pupil_detector import (
    detect_pupil as detect_pupil_hough_fullframe,
    draw_overlay,
    reencode_for_compatibility,
    SAMPLE_FRAMES,
)

_EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def remove_glint(gray, kernel_size=13):
    """Suppress small bright specular highlights via grayscale morphological
    opening, so a glint doesn't distort the radius estimate afterward."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


def detect_eye_box(gray, min_neighbors=3, min_size=40):
    """Return (x, y, w, h) of the largest detected eye, or None."""
    eyes = _EYE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=min_neighbors, minSize=(min_size, min_size)
    )
    if len(eyes) == 0:
        return None
    return tuple(max(eyes, key=lambda b: b[2] * b[3]))


def _score_hough_candidates(gray_or_crop, circles):
    best, best_score = None, -1.0
    h, w = gray_or_crop.shape
    for cx, cy, r in circles[0]:
        if r <= 0 or not (0 <= cx < w and 0 <= cy < h):
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
        mean_intensity = cv2.mean(gray_or_crop, mask=mask)[0]
        score = (255.0 - mean_intensity) * float(r)
        if score > best_score:
            best, best_score = (int(cx), int(cy), int(r), float(mean_intensity)), score
    return best


def detect_pupil_in_eye_crop(gray, box, pad_frac=0.15, debug=False):
    """Search for the pupil only within the detected eye box. Returns
    (cx, cy, radius, brightness) in full-image coordinates, or None."""
    ex, ey, ew, eh = box
    pad = int(pad_frac * ew)
    x0, y0 = max(0, ex - pad), max(0, ey - pad)
    x1, y1 = min(gray.shape[1], ex + ew + pad), min(gray.shape[0], ey + eh + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    if debug:
        cv2.imwrite("debug_eye_crop.png", crop)

    # Radius search range scaled to the detected eye's own size, instead of a
    # fixed pixel range -- this is what lets the same code handle both a
    # close-up eye filling most of the frame and a small, distant one.
    min_r, max_r = max(4, int(ew * 0.08)), max(6, int(ew * 0.22))
    blur = cv2.GaussianBlur(crop, (7, 7), 1.5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(20, ew // 3),
        param1=60.0, param2=15.0, minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None:
        best = _score_hough_candidates(crop, circles)
        if best is not None:
            cx, cy, r, b = best
            if debug:
                print(f"  Eye-crop Hough: found ({cx + x0},{cy + y0}) r={r}")
            return cx + x0, cy + y0, r, b
    if debug:
        print("  Eye-crop Hough: nothing found, trying glint fallback")

    # Fallback: brightest spot in the crop (corneal glint), with a sanity
    # check so a flat, glint-less crop doesn't produce a confident-looking
    # but meaningless result.
    glint_blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, glint_val, _, glint_loc = cv2.minMaxLoc(glint_blur)
    crop_median = float(np.median(glint_blur))
    if glint_val < max(150, crop_median + 40):
        if debug:
            print(f"  Eye-crop glint: no clear glint (brightest={glint_val}, median={crop_median:.0f})")
        return None

    fx, fy = glint_loc[0] + x0, glint_loc[1] + y0
    opened = remove_glint(gray)
    center_val = float(opened[fy, fx])
    radii = []
    for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        dx, dy = np.cos(ang), np.sin(ang)
        for r in range(2, max_r + 15):
            px, py = int(round(fx + dx * r)), int(round(fy + dy * r))
            if not (0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]):
                break
            if float(opened[py, px]) > center_val + 18:
                radii.append(r)
                break
    radius = int(np.clip(np.median(radii), min_r, max_r)) if len(radii) >= 6 else (min_r + max_r) // 2

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (fx, fy), radius, 255, -1)
    brightness = float(cv2.mean(gray, mask=mask)[0])
    if debug:
        print(f"  Eye-crop glint: found ({fx},{fy}) val={glint_val} radius={radius} "
              f"from {len(radii)} samples")
    return fx, fy, radius, brightness


def crop_eye_slit(gray, debug=False):
    """Legacy heuristic (previous version of this file): the widest dark
    blob in the lit window, kept only as a last-resort fallback for frames
    where the Haar cascade finds no eye at all."""
    _, bright_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    bright_mask = cv2.erode(bright_mask, np.ones((10, 10), np.uint8))
    if cv2.countNonZero(bright_mask) < 200:
        return None

    vals = gray[bright_mask > 0]
    lash_thresh = np.percentile(vals, 8)
    _, lash_mask = cv2.threshold(gray, lash_thresh, 255, cv2.THRESH_BINARY_INV)
    lash_mask = cv2.bitwise_and(lash_mask, bright_mask)
    closed = cv2.morphologyEx(lash_mask, cv2.MORPH_CLOSE, np.ones((7, 31), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    best, best_w = None, -1
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 100:
            continue
        if w > best_w:
            best_w, best = w, (x, y, w, h)
    if best is None and debug:
        print("  Legacy fallback: no eye-slit-shaped dark region found")
    return best


def detect_pupil(gray, min_neighbors=3, eye_min_size=40, debug=False):
    """Try Haar eye detection + in-crop search first (robust to surrounding
    clutter); fall back to legacy full-frame methods only if Haar finds no
    eye at all. Returns (cx, cy, radius, brightness) or None."""
    box = detect_eye_box(gray, min_neighbors, eye_min_size)
    if box is not None:
        if debug:
            print(f"  Haar eye box: {box}")
        result = detect_pupil_in_eye_crop(gray, box, debug=debug)
        if result is not None:
            return result
        if debug:
            print("  Nothing usable in the eye crop, falling back to full-frame methods")
    elif debug:
        print("  Haar: no eye found, falling back to full-frame methods")

    result = detect_pupil_hough_fullframe(gray, min_radius=8, max_radius=20, debug=debug)
    if result is not None:
        return result

    legacy_box = crop_eye_slit(gray, debug=debug)
    if legacy_box is None:
        return None
    return detect_pupil_in_eye_crop(gray, legacy_box, pad_frac=0.1, debug=debug)


def process_video(src, dst, args):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: could not open '{src}'. Check the path/codec.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    i, hits = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dbg = args.debug and i == 0
        if dbg:
            print(f"Frame {i}:")
        result = detect_pupil(gray, args.min_neighbors, args.eye_min_size, debug=dbg)
        if result:
            hits += 1
        label_idx = min(SAMPLE_FRAMES, int(i * SAMPLE_FRAMES / max(n, 1)) + 1)
        out.write(draw_overlay(frame, result, label_idx, SAMPLE_FRAMES))
        i += 1

    cap.release()
    out.release()
    print(f"Wrote {dst} ({i} frames, pupil found in {hits}/{i})")
    reencode_for_compatibility(dst)


def process_image(src, dst, args):
    img = cv2.imread(src)
    if img is None:
        print(f"Error: could not open '{src}' as an image.")
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detect_pupil(gray, args.min_neighbors, args.eye_min_size, debug=args.debug)
    out = draw_overlay(img, result, 1, 1)
    cv2.imwrite(dst, out)
    print(f"Wrote {dst} (pupil {'found' if result else 'NOT found'})")


def live(cam_index, args):
    cap = cv2.VideoCapture(cam_index)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        i += 1
        result = detect_pupil(gray, args.min_neighbors, args.eye_min_size)
        cv2.imshow("Pupil detector (experimental)", draw_overlay(frame, result, i, i))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="Input video/image path, or a camera index (e.g. 0) for live mode")
    p.add_argument("output", nargs="?", help="Output path (omit for live camera mode)")
    p.add_argument("--min-neighbors", type=int, default=3,
                   help="Haar cascade strictness (default 3). Raise if it finds a false eye "
                        "in the housing; lower if it misses a real eye.")
    p.add_argument("--eye-min-size", type=int, default=40,
                   help="Minimum eye box size in px for the cascade to consider (default 40). "
                        "Lower this if the eye appears small/far away in your footage.")
    p.add_argument("--debug", action="store_true",
                   help="Print what each stage found; saves debug_eye_crop.png for frame 0/the image")
    args = p.parse_args()

    if args.output:
        ext = args.output.lower().rsplit(".", 1)[-1]
        if ext in ("png", "jpg", "jpeg", "bmp"):
            process_image(args.source, args.output, args)
        else:
            process_video(args.source, args.output, args)
    elif args.source.isdigit():
        live(int(args.source), args)
    else:
        print("No output path given and source isn't a camera index.\n")
        p.print_help()


if __name__ == "__main__":
    main()