# E-Eye Pupil Detection & Optimization (Master Repository)

Master repository combining the E-Eye photorefraction test datasets, deep learning inference pipelines, and algorithmic optimizations for robust pupil detection with eyelid false-positive rejection.

---

## 🌟 Highlights & Architecture

- **Dual-Engine Detection**:
  - **Deep U-Net Pipeline**: End-to-end PyTorch U-Net trained on infrared ocular imagery.
  - **Optimized Hybrid Pipeline**: Glint-guided region scoring + morphological opening + geometric filtering (aspect ratio, circularity, solidity) to reject eyelids and camera housing artifacts.
  - **Haar-Cascade + Eye-Crop Hough**: Fallback detector for high-contrast ocular crops.
- **Eyelid False-Positive Rejection**: Specifically eliminates false pupil detections caused by drooping eyelids, eyelashes, and palpebral creases.
- **Complete Test Dataset**: Bundles labeled captures across emmetropic, hyperopic, and myopic eyes, along with reference video (`ideal captured.mp4`) and visual annotations.
- **Training & Retraining Scaffold**: Complete pipeline with Dice + BCE loss and hard-negative eyelid occlusion augmentation.

---

## 📁 Repository Structure

```
EEye-Pupil-Optimization/
├── checkpoints/
│   └── pupil-unet-epoch48-val_dice0.7170.ckpt  # Pre-trained U-Net weights
├── data/
│   ├── emmetropic/                             # Capture videos (emmetropic)
│   ├── hyperopia/                              # Capture videos (hyperopia)
│   ├── myopia/                                 # Capture videos (myopia)
│   ├── pupil_capture_attempt/                  # Real-world diagnostic attempts
│   └── reference/                              # Benchmark references
│       ├── ideal captured.mp4
│       ├── reference image to be captured.png
│       └── terminal.png
├── docs/
│   ├── Literatures/                            # Ophthalmic & deep learning papers
│   ├── eeye commands.txt                       # Hardware / Raspberry Pi runtime cheatsheet
│   ├── PROBLEM_STATEMENT_AND_OPTIMIZATION.md   # Mathematical & biological problem analysis
│   └── RE_guide.pdf                            # Refractive error clinical measurement guide
├── models/
│   └── unet.py                                 # PyTorch U-Net architecture definition
├── src/
│   ├── export_frames.py                        # Extract PNG dataset frames for labeling
│   ├── inference.py                            # Baseline U-Net video inference
│   ├── optimized_inference.py                  # Eyelid-rejection & glint-guided inference
│   ├── pupil_detector.py                       # Classical OpenCV Hough circle detector
│   ├── pupil_detector2.py                      # Haar cascade eye-crop detector
│   ├── train_unet.py                           # Training & fine-tuning pipeline
│   └── unet.py                                 # Symlinked model definition
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Kent0625/EEye-Pupil-Optimization.git
cd EEye-Pupil-Optimization
pip install -r requirements.txt
```

### 2. Run Baseline U-Net Inference

```bash
python src/inference.py "data/reference/ideal captured.mp4" \
    --checkpoint checkpoints/pupil-unet-epoch48-val_dice0.7170.ckpt \
    --output output_baseline.mp4
```

### 3. Run Optimized Eyelid-Rejection Inference (Recommended)

```bash
python src/optimized_inference.py "data/hyperopia/bad capture.mp4" \
    --checkpoint checkpoints/pupil-unet-epoch48-val_dice0.7170.ckpt \
    --output output_optimized.mp4 \
    --threshold 0.40 \
    --smooth 0.6
```

Options:
- `--no-show`: Run headless (no GUI popup, ideal for Raspberry Pi / servers).
- `--smooth <float>`: Exponential moving average factor ($0.0 - 0.9$) for inter-frame temporal smoothing.
- `--threshold <float>`: U-Net probability threshold (default: $0.40$).

### 4. Run Classical Haar + Eye-Crop Detector

```bash
python src/pupil_detector2.py "data/reference/ideal captured.mp4" output_haar.mp4 --debug
```

### 5. Scrape Frames for Annotation

To create new training data from your video recordings:

```bash
python src/export_frames.py \
    --input data/reference/ideal captured.mp4 \
    --output data/extracted_frames \
    --stride 5
```

### 6. Train or Fine-Tune U-Net

Prepare your labeled data under `data/annotated/` with `images/` and `masks/` subfolders:

```bash
python src/train_unet.py \
    --data-dir data/annotated \
    --checkpoint checkpoints/pupil-unet-epoch48-val_dice0.7170.ckpt \
    --epochs 50 \
    --batch-size 4 \
    --lr 0.001
```

---

## 🔬 Clinical & Optical Context

For detailed clinical math, Purkinje reflection physics, and geometric rejection rules, see [PROBLEM_STATEMENT_AND_OPTIMIZATION.md](docs/PROBLEM_STATEMENT_AND_OPTIMIZATION.md).
