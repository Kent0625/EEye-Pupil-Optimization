# E-Eye Pupil Detection and Optimization

This repository contains the video dataset, detection code, and improved pupil detection program for the E-Eye portable vision screening device.

---

## Overview

The E-Eye device captures video of a person's eye to check how well they can see. To do this measurement properly, the software must locate the center black circle of the eye (the pupil) accurately in every video frame.

In earlier versions of the software, the detector had trouble when eyelids, eyelashes, or dark corners got in the way. This updated version fixes those problems and shades the detected pupil area cleanly, matching the ideal capture example.

---

## What Was the Problem Before?

1. **Eyelids Mistaken for Pupils**:
   When people squinted or had low-hanging eyelids, the program would often get confused. It would track the dark line of the upper or lower eyelid instead of the actual pupil.

2. **Jumping and Shaking**:
   If the person blinked or the light shifted, the detection circle would suddenly jump to the cheek or the eyelid edge, making the size measurement incorrect.

3. **Missing Detections**:
   On some videos where the lighting was slightly darker, the old program could not find the pupil at all, causing the measurement to fail.

4. **Only an Empty Line**:
   The previous test code only drew an outline circle instead of filling the pupil area with a clear red tint like the team's ideal sample video.

---

## What Improvements Were Made?

1. **Red Area Shading (Just Like the Ideal Capture)**:
   The pupil area is now shaded with a transparent red tint, outlined with a bright green border, and marked with a clean white center dot. This matches the exact appearance shown in the ideal reference video.

2. **Smart Eyelid Filter**:
   The new code looks only inside the natural opening of the eye. It automatically ignores wide, flat horizontal shapes (which are eyelids) and only accepts round, circular shapes (which are pupils).

3. **Light Reflection Anchor**:
   The E-Eye device shines a small infrared light into the eye, creating a bright white dot reflection on the front of the eye. The updated program uses this reflection as an anchor point. Because eyelids do not have this reflection inside them, the detector will not jump to the eyelid.

4. **Smooth Video Tracking**:
   The program remembers the pupil position from one video frame to the next. If the person blinks for a split second, the detector stays calm and holds its place instead of jumping wildly across the screen.

5. **Ready for Training**:
   A simple training program is included so that as new eye videos are collected, the model can easily be retrained and improved further.

---

## What is Inside This Repository?

- **src/**: Contains the main Python programs.
  - `optimized_inference.py`: The improved detection program that shades the pupil and removes eyelid errors.
  - `inference.py`: The original baseline program for comparison.
  - `pupil_detector2.py`: The fast helper detector using eye-box cropping.
  - `train_unet.py`: Program to retrain the model when you have new labeled images.
  - `export_frames.py`: Tool to save video frames as PNG pictures so you can label them.
  - `unet.py`: The deep learning model structure.
- **checkpoints/**: The saved model weights file (`pupil-unet-epoch48-val_dice0.7170.ckpt`).
- **data/**:
  - `emmetropic/`: Videos of normal eyes.
  - `hyperopia/`: Videos of farsighted eyes, including difficult videos with drooping eyelids.
  - `myopia/`: Videos of nearsighted eyes.
  - `pupil_capture_attempt/`: Test videos from earlier device builds.
  - `reference/`: The target sample video (`ideal captured.mp4`), drawing guide (`reference image to be captured.png`), and improved result videos.
- **docs/**: Setup guides, literature, and command cheat sheets for the device.
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

To process any video and see the window on your screen:

```bash
python src/optimized_inference.py "data/reference/ideal captured.mp4" -o "my_output.mp4"
```

To run quietly in the background without opening a window (saves directly to the output video file):

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

On the Raspberry Pi, go to your project folder and activate your Python environment:

```bash
cd ~/eeye
source eeye_env/bin/activate
```

If you need to install the lightweight packages on the Pi, run:

```bash
pip install opencv-python-headless numpy torch torchvision
```

### Step 3: Copy the Improved Script to the Pi

You can clone this repository directly on the Raspberry Pi:

```bash
cd ~
git clone https://github.com/Kent0625/EEye-Pupil-Optimization.git
cd EEye-Pupil-Optimization
```

### Step 4: Run Detection on the Raspberry Pi

Because the Raspberry Pi is usually used without a computer monitor attached, always add the `--no-show` flag so it runs cleanly in the background:

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
- **Drooping Eyelid Video**: Rejects eyelid edges, shadows, and eyelashes. The red shading stays strictly inside the pupil circle without bleeding into the skin.
- **Speed**: Runs efficiently on both desktop computers and small devices like the Raspberry Pi.
