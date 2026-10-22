# E-Eye Pupil Detection and Optimization

This repository contains the video dataset, detection code, and improved pupil detection program for the E-Eye portable vision screening device.

---

## Overview

The E-Eye device captures video of a person's eye to check how well they can see. To do this measurement properly, the software must locate the center black circle of the eye (the pupil) accurately in every video frame.

In earlier versions of the software, the detector had trouble when eyelids, eyelashes, or the dark tube housing got in the way. This updated version fixes those problems by using a "Zoom In Then Capture" approach and shades the detected pupil area cleanly in red, matching the ideal capture example.

---

## What Was the Problem Before?

1. **Dark Background Housing Distractions**:
   The camera records inside a dark enclosure tube. Over 65% of the video frame is pitch-black cardboard. The old program examined the entire frame at once, which caused it to mistake dark shadows, tape edges, or tube corners for the pupil.

2. **Eyelids Mistaken for Pupils**:
   When people squinted or had drooping eyelids, the old program would often get confused. It tracked the dark fold of the upper or lower eyelid instead of the actual pupil.

3. **Pixel Loss on Small Eyes**:
   When subjects had narrow or deep-set eyes, the eye appeared very small in the middle of the frame. Because the resolution was small, the detector had trouble finding the exact edges of the pupil.

4. **Jumping and Shaking**:
   If the person blinked or the light shifted, the detection circle would suddenly jump across the frame, making size measurements unreliable.

5. **Only an Empty Line**:
   The previous test code only drew a thin outline circle instead of filling the pupil area with a clear red tint like the team's ideal sample video.

---

## What Improvements Were Made?

1. **Zoom In Then Capture Approach (Eye Only)**:
   The software now cuts out the black tube background before running detection. It focuses strictly on the illuminated opening of the eye, then magnifies that eye area into a clean, steady square image. Because the dark housing is removed immediately, the program cannot get tricked by tube corners or exterior shadows.

2. **Double the Detail for Narrow Eyes**:
   By zooming in on the eye region first, the software gains up to two times more pixel detail across the pupil. This allows clean, accurate circle measurements even on squinting or small eyes.

3. **Red Area Shading (Just Like the Ideal Capture)**:
   The pupil area is shaded with a transparent red tint, outlined with a bright green border, and marked with a clean white center dot. This matches the exact appearance shown in the ideal reference video.

4. **Smart Eyelid Filter**:
   The program searches only inside the natural central opening of the eye. It automatically rejects wide, flat horizontal shapes (eyelids) and accepts only round, circular shapes (pupils).

5. **Infrared Light Reflection Anchor**:
   The E-Eye device shines a small infrared light into the eye, creating a bright white reflection on the front of the eye. The updated program uses this reflection as an anchor point. Because eyelids do not have this reflection inside them, the detector will not jump to the eyelid.

6. **Smooth, Stable Video Tracking**:
   The program remembers the eye and pupil position from one video frame to the next. If the person blinks for a split second, the detector holds its place smoothly instead of shaking or jumping.

---

## Display View Options

The program supports three view modes using the `--view` flag:

1. **Zoomed Eye View (`--view zoomed`, Default)**:
   Shows only the magnified eye with the black background removed. The pupil is shaded in red with the green border and white center point.

2. **Side-by-Side View (`--view side-by-side`)**:
   Shows the zoomed eye on the left and the full camera frame on the right for comparison.

3. **Full Frame View (`--view full`)**:
   Shows the full camera image with a yellow box indicating the eye region and the shaded pupil.

---

## What is Inside This Repository?

- **src/**: Main Python programs.
  - `optimized_inference.py`: The improved detection program featuring the Zoom In Then Capture pipeline and red pupil shading.
  - `inference.py`: The original baseline program for comparison.
  - `pupil_detector2.py`: Helper detector using eye-box localization.
  - `train_unet.py`: Training program to retrain the neural network when new labeled images are added.
  - `export_frames.py`: Tool to save video frames as PNG images for labeling.
  - `unet.py`: The neural network model structure.
- **checkpoints/**: The saved model weights file (`pupil-unet-epoch48-val_dice0.7170.ckpt`).
- **data/**:
  - `emmetropic/`: Videos of normal eyes.
  - `hyperopia/`: Videos of farsighted eyes, including difficult videos with drooping eyelids.
  - `myopia/`: Videos of nearsighted eyes.
  - `pupil_capture_attempt/`: Test videos from earlier device builds.
  - `reference/`: Reference sample video (`ideal captured.mp4`), drawing guide (`reference image to be captured.png`), and sample outputs.
- **docs/**: Setup guides, literature, and command cheat sheets.
- **requirements.txt**: List of software packages needed to run the code.

---

## How to Run on a Computer (Windows, Mac, or Linux)

### Step 1: Download the Repository

Open a terminal or command prompt and run:

```bash
git clone https://github.com/Kent0625/EEye-Pupil-Optimization.git
cd EEye-Pupil-Optimization
```

### Step 2: Install Needed Packages

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Improved Detection

To process any video in the default Zoom-In mode (eye only) with an on-screen preview:

```bash
python src/optimized_inference.py "data/reference/ideal captured.mp4" -o "my_output.mp4"
```

To view both the zoomed eye and the full camera view side by side:

```bash
python src/optimized_inference.py "data/reference/ideal captured.mp4" --view side-by-side -o "side_by_side.mp4"
```

To run quietly in the background without opening a window (saves directly to the video file):

```bash
python src/optimized_inference.py "data/reference/ideal captured.mp4" -o "my_output.mp4" --no-show
```

---

## How to Run on the Raspberry Pi

The E-Eye device uses a Raspberry Pi to control the camera and record eye videos. You can run this improved version directly on the Raspberry Pi without needing a desktop screen or a heavy graphics card.

### Step 1: Connect to the Raspberry Pi

Make sure your computer is connected to the same Wi-Fi network as the Raspberry Pi, then open your terminal and log in using SSH:

```bash
ssh admin@eeye.local
```
*(If `eeye.local` does not work, replace it with the Pi IP address, such as `ssh admin@192.168.254.113`)*

### Step 2: Open or Create the Environment

On the Raspberry Pi, navigate to your project folder and activate your Python environment:

```bash
cd ~/eeye
source eeye_env/bin/activate
```

If you need to install the lightweight packages on the Pi, run:

```bash
pip install opencv-python-headless numpy torch torchvision
```

### Step 3: Clone or Pull the Latest Code

You can clone or update the repository on the Raspberry Pi:

```bash
cd ~
git clone https://github.com/Kent0625/EEye-Pupil-Optimization.git
cd EEye-Pupil-Optimization
```

Or if you already have the repository on the Pi:

```bash
cd ~/EEye-Pupil-Optimization
git pull origin main
```

### Step 4: Run Detection on the Raspberry Pi

Because the Raspberry Pi is usually operated without a monitor attached, always add the `--no-show` flag so it runs cleanly in the background:

```bash
python3 src/optimized_inference.py "data/reference/ideal captured.mp4" -o "pi_output.mp4" --no-show
```

### Step 5: Copy the Result Video Back to Your Computer

After the program finishes, open a command prompt on your laptop or computer and download the processed video:

```bash
scp admin@eeye.local:~/EEye-Pupil-Optimization/pi_output.mp4 C:\Users\ASUS\Downloads\
```

You can now open and watch the video on your computer.

---

## Summary of Results

- **Ideal Capture Video**: Tracks the pupil smoothly across all 147 frames (100% success rate) with a steady red shade and green outline.
- **Drooping Eyelid Video**: Successfully ignores eyelid edges, shadows, and eyelashes. The red shading stays strictly inside the pupil circle without bleeding into the skin.
- **Narrow Eye Attempt Video**: Increases effective pupil detail by two times, keeping detection steady even when the eye is partially closed.
- **Speed**: Runs efficiently on standard computers and small portable devices like the Raspberry Pi.
