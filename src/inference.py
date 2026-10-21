import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms

from unet import UNet


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Pupil segmentation inference on MP4 video."
)

parser.add_argument(
    "--device",
    "-d",
    choices=["auto", "cpu", "cuda"],
    default="auto",
    help="Inference device: auto, cpu, or cuda. Default: auto"
)

parser.add_argument(
    "video",
    help="Path to input MP4 video."
)

parser.add_argument(
    "--checkpoint",
    "-c",
    default=(
        "logs/pupil_unet/version_0/checkpoints/"
        "pupil-unet-epoch=48-val_dice=0.7170.ckpt"
    ),
    help="Path to Lightning checkpoint."
)

parser.add_argument(
    "--output",
    "-o",
    default="pupil_prediction.mp4",
    help="Output MP4 filename."
)

parser.add_argument(
    "--threshold",
    "-t",
    type=float,
    default=0.5,
    help="Pupil probability threshold. Default: 0.5"
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.4,
    help="Mask overlay opacity. Default: 0.4"
)

parser.add_argument(
    "--no-save",
    action="store_true",
    help="Display inference without saving the video."
)

parser.add_argument(
    "--no-show",
    action="store_true",
    help="Do not display GUI window during inference."
)

args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = args.video
CHECKPOINT_PATH = args.checkpoint
OUTPUT_PATH = args.output

THRESHOLD = args.threshold
MASK_ALPHA = args.alpha

SAVE_VIDEO = not args.no_save

# OpenCV uses (width, height)
WIDTH = 640
HEIGHT = 480


# ============================================================
# VALIDATE ARGUMENTS
# ============================================================

if not 0 <= THRESHOLD <= 1:
    raise ValueError(
        "Threshold must be between 0 and 1."
    )

if not 0 <= MASK_ALPHA <= 1:
    raise ValueError(
        "Alpha must be between 0 and 1."
    )


# ============================================================
# DEVICE
# ============================================================

# ============================================================
# DEVICE
# ============================================================

if args.device == "auto":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

elif args.device == "cuda":

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print(f"Device: {device}")
print("=" * 60)
print("PUPIL SEGMENTATION")
print("=" * 60)

print(f"Device     : {device}")
print(f"Video      : {VIDEO_PATH}")
print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"Threshold  : {THRESHOLD}")
print(f"Save video : {SAVE_VIDEO}")

if SAVE_VIDEO:
    print(f"Output     : {OUTPUT_PATH}")

print("=" * 60)


# ============================================================
# IMAGE TRANSFORM
#
# Must match training:
#
# transforms.ToTensor()
# transforms.Resize((480, 640))
# transforms.Grayscale()
#
# torchvision Resize uses:
#
# (height, width)
#
# ============================================================

img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.Grayscale()
])


# ============================================================
# CREATE U-NET
# ============================================================

model = UNet(
    in_channels=1,
    num_classes=1
)


# ============================================================
# LOAD LIGHTNING CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

state_dict = checkpoint["state_dict"]


# ============================================================
# EXTRACT U-NET WEIGHTS
#
# Lightning checkpoint:
#
# model.enc1.conv.0.weight
#
# U-Net expects:
#
# enc1.conv.0.weight
#
# ============================================================

unet_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("model."):

        new_key = key[len("model."):]

        unet_state_dict[new_key] = value


# Make sure weights were actually found
if len(unet_state_dict) == 0:

    raise RuntimeError(
        "No U-Net weights were found in the checkpoint. "
        "Expected checkpoint keys beginning with 'model.'."
    )


model.load_state_dict(
    unet_state_dict
)

model = model.to(device)

model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise ValueError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30.0


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print()
print(f"FPS          : {fps:.2f}")
print(f"Total frames : {total_frames}")
print(f"Input size   : {WIDTH}x{HEIGHT}")


# ============================================================
# OUTPUT VIDEO
# ============================================================

writer = None

if SAVE_VIDEO:

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (WIDTH, HEIGHT)
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Could not create output video: {OUTPUT_PATH}"
        )


# ============================================================
# DISPLAY COLORS
#
# OpenCV = BGR
# ============================================================

# Red mask
MASK_COLOR = np.array(
    [0, 0, 255],
    dtype=np.uint8
)

# Green contour
CONTOUR_COLOR = (
    0,
    255,
    0
)


# ============================================================
# VIDEO INFERENCE
# ============================================================

frame_number = 0


while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------------------------
    # RESIZE
    #
    # cv2.resize uses:
    #
    # (width, height)
    # --------------------------------------------------------

    frame = cv2.resize(
        frame,
        (WIDTH, HEIGHT)
    )


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # PREPROCESS
    #
    # [H, W, 3]
    #
    #        ↓
    #
    # [1, 480, 640]
    # --------------------------------------------------------

    x = img_trans(
        rgb_frame
    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    #
    # [1, 480, 640]
    #
    #        ↓
    #
    # [1, 1, 480, 640]
    # --------------------------------------------------------

    x = x.unsqueeze(0)

    x = x.to(device)


    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    with torch.inference_mode():

        logits = model(x)

        probability = torch.sigmoid(
            logits
        )

        prediction = (
            probability >= THRESHOLD
        )


    # ========================================================
    # TENSOR -> NUMPY MASK
    # ========================================================

    mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
        * 255
    )


    # ========================================================
    # CREATE DISPLAY FRAME
    # ========================================================

    display = frame.copy()


    # ========================================================
    # MASK OVERLAY
    # ========================================================

    pupil_pixels = (
        mask > 0
    )

    overlay = display.copy()

    overlay[pupil_pixels] = (
        MASK_COLOR
    )


    display = cv2.addWeighted(
        overlay,
        MASK_ALPHA,
        display,
        1.0 - MASK_ALPHA,
        0
    )


    # ========================================================
    # FIND CONNECTED REGIONS
    # ========================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    pupil_contour = None


    if len(contours) > 0:

        # ----------------------------------------------------
        # Largest predicted region
        # ----------------------------------------------------

        pupil_contour = max(
            contours,
            key=cv2.contourArea
        )


        # ----------------------------------------------------
        # DRAW ACTUAL SEGMENTATION BOUNDARY
        # ----------------------------------------------------

        cv2.drawContours(
            display,
            [pupil_contour],
            -1,
            CONTOUR_COLOR,
            2
        )


        # ====================================================
        # FIT ELLIPSE
        # ====================================================

        if len(pupil_contour) >= 5:

            ellipse = cv2.fitEllipse(
                pupil_contour
            )


            # Draw fitted ellipse
            cv2.ellipse(
                display,
                ellipse,
                CONTOUR_COLOR,
                2
            )


            # ------------------------------------------------
            # EXTRACT ELLIPSE INFORMATION
            # ------------------------------------------------

            (
                (cx, cy),
                (axis1, axis2),
                angle
            ) = ellipse


            major_axis = max(
                axis1,
                axis2
            )

            minor_axis = min(
                axis1,
                axis2
            )


            # ------------------------------------------------
            # EQUIVALENT DIAMETER
            #
            # sqrt(major * minor)
            # ------------------------------------------------

            equivalent_diameter = np.sqrt(
                major_axis
                *
                minor_axis
            )


            # ------------------------------------------------
            # PUPIL CENTER
            # ------------------------------------------------

            center = (
                int(cx),
                int(cy)
            )

            cv2.circle(
                display,
                center,
                3,
                (255, 255, 255),
                -1
            )


            # ------------------------------------------------
            # DISPLAY DIAMETER
            # ------------------------------------------------

            cv2.putText(
                display,
                f"Pupil D: {equivalent_diameter:.1f}px",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


    # ========================================================
    # FRAME INFORMATION
    # ========================================================

    cv2.putText(
        display,
        f"Frame: {frame_number}",
        (20, HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # SAVE FRAME
    # ========================================================

    if writer is not None:
        writer.write(display)

    # ========================================================
    # DISPLAY
    # ========================================================

    if not args.no_show:
        cv2.imshow(
            "Pupil Segmentation",
            display
        )

        delay = max(
            1,
            int(1000 / fps)
        )

        key = (
            cv2.waitKey(delay)
            & 0xFF
        )

        # Q or ESC to quit
        if (
            key == ord("q")
            or key == 27
        ):
            break


    frame_number += 1


# ============================================================
# CLEAN UP
# ============================================================

cap.release()


if writer is not None:
    writer.release()


if not args.no_show:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)

print(
    f"Frames processed: {frame_number}"
)

if SAVE_VIDEO:

    print(
        f"Saved to: {OUTPUT_PATH}"
    )