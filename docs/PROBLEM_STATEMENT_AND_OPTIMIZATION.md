# E-Eye Pupil Detection Optimization & Eyelid Rejection

## 1. Problem Statement

In the E-Eye portable eccentric infrared photorefractor, video footage captured through the alignment aperture often presents difficult optical conditions:
- **Eyelid and Eyelash Clutter**: Dark horizontal folds from drooping eyelids and eyelashes exhibit sharp contrast boundaries in infrared light.
- **False-Positive Pupil Detections**:
  - Classical Hough Circle searches across the full frame frequently lock onto dark camera housing corners or eyelid folds.
  - Deep U-Net segmentation can produce elongated masks that bleed into or mistake the upper eyelid margin for the pupil boundary.
- **Distorted Biometrics**: When the eyelid partially covers the pupil, naive ellipse fitting (`cv2.fitEllipse`) flattens and tilts, resulting in invalid equivalent diameter and inaccurate refractive error estimates.

---

## 2. Visual Reference

As annotated in `data/reference/reference image to be captured.png`:
- **Pupil Target (Green)**: The true circular pupil aperture containing the central corneal glint (Purkinje reflection).
- **Iris Region (Yellow)**: The iris surrounding the pupil.
- **Eyelid & Boundary Artifacts (Red X)**: Eyelashes, upper and lower eyelid margins, and lateral canthi that must be rejected.

---

## 3. Two-Stage Optimization Strategy

### Stage A: Inference Post-Processing & Eyelid Filtering (`optimized_inference.py`)

1. **Corneal Glint Co-occurrence**:
   - The eccentric IR LED illuminator creates a sharp specular glint on the cornea.
   - Genuine pupil contours enclose or sit adjacent to this high-intensity reflection. Eyelid skin and eyelashes do not produce corneal glints.
2. **Morphological Opening**:
   - Applying an elliptical structuring element ($5\times 5$) severs thin horizontal eyelash and eyelid crease artifacts while maintaining the circular pupil body.
3. **Geometric Rejection Filters**:
   - **Aspect Ratio ($w/h$)**: Rejects candidates with aspect ratio $> 2.5$ or $< 0.4$ (eyelid folds are heavily elongated horizontally).
   - **Circularity ($\frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$)**: Pupil candidates must exceed $0.32$ (circular pupils approach $1.0$, whereas jagged eyelid margins are $< 0.3$).
   - **Solidity ($\frac{\text{Area}}{\text{ConvexHull Area}}$)**: Rejects concave contours caused by eyelid drooping or shadow notches.
4. **Temporal Exponential Moving Average (EMA)**:
   - Stabilizes frame-to-frame ellipse fitting and prevents sudden jumps during partial blinks.

---

### Stage B: Training & Retraining Scaffold (`train_unet.py`)

1. **Synthetic Hard-Negative Eyelid Augmentation**:
   - Injects random horizontal and parabolic dark occlusions across the eye aperture during training, forcing the U-Net to learn that eyelid shadows are not part of the pupil mask.
2. **Combined Dice + BCE Loss**:
   - Balances pixel-level binary cross-entropy with global overlap optimization to prevent boundary bleeding.
3. **Transfer Learning**:
   - Fine-tunes from `checkpoints/pupil-unet-epoch48-val_dice0.7170.ckpt` rather than training from scratch.
