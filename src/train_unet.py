"""
E-Eye U-Net Pupil Segmentation Training & Fine-Tuning Pipeline
=============================================================
Features:
  - Supports training from scratch or fine-tuning from existing checkpoints
  - Combined Dice + BCE Loss with customizable loss weights
  - Hard-negative eyelid occlusion data augmentation (simulates eyelid drooping
    and horizontal shadows so the network learns to exclude eyelids)
  - Validation metrics: Dice Score, Intersection-over-Union (IoU), Precision, Recall
  - Automatic checkpoint saving based on best validation Dice score
"""

import argparse
import glob
import os
import random
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from unet import UNet


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ============================================================
# DATASET & AUGMENTATION
# ============================================================

def simulate_eyelid_occlusion(image: np.ndarray, mask: np.ndarray, prob: float = 0.4):
    """
    Simulates upper/lower eyelid droop and shadow over the aperture.
    This trains the network to recognize eyelid folds and NOT classify them as pupil.
    """
    if random.random() > prob:
        return image, mask

    h, w = image.shape[:2]
    aug_img = image.copy()

    # Upper eyelid droop: dark curve/polygon across upper third of eye
    eyelid_h = random.randint(int(h * 0.1), int(h * 0.45))
    y_points = [
        int(eyelid_h + random.uniform(-10, 15) * np.sin(np.pi * x / w))
        for x in range(w)
    ]
    poly = [[0, 0], [w - 1, 0]] + [[x, y_points[x]] for x in range(w - 1, -1, -1)]
    poly = np.array(poly, dtype=np.int32)

    # Darken region to simulate eyelid shadow
    shadow_val = random.randint(20, 70)
    eyelid_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(eyelid_mask, [poly], 255)

    aug_img[eyelid_mask > 0] = np.clip(
        aug_img[eyelid_mask > 0].astype(np.int32) - shadow_val, 0, 255
    ).astype(np.uint8)

    # Any pupil mask region covered by eyelid is occluded (not visible pupil)
    aug_mask = mask.copy()
    aug_mask[eyelid_mask > 0] = 0

    return aug_img, aug_mask


class PupilDataset(Dataset):
    def __init__(self, data_dir: str, augment: bool = True, target_size: tuple = (480, 640)):
        self.target_size = target_size
        self.augment = augment

        img_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks")

        self.samples = []
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
            for img_path in img_files:
                base = os.path.basename(img_path)
                name, _ = os.path.splitext(base)
                # Look for matching mask
                for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                    mask_candidate = os.path.join(mask_dir, name + ext)
                    if os.path.exists(mask_candidate):
                        self.samples.append((img_path, mask_candidate))
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise ValueError(f"Could not load image or mask: {img_path}, {mask_path}")

        img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)

        # Binary mask (0 or 1)
        mask = (mask > 127).astype(np.uint8)

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            # Eyelid occlusion simulation
            img, mask = simulate_eyelid_occlusion(img, mask)

            # Random brightness jitter
            jitter = random.uniform(0.85, 1.15)
            img = np.clip(img * jitter, 0, 255).astype(np.uint8)

        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return img_tensor, mask_tensor


# ============================================================
# EVALUATION METRIC
# ============================================================

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    union = preds_flat.sum().item() + targets_flat.sum().item() - intersection

    dice = (2.0 * intersection) / (preds_flat.sum().item() + targets_flat.sum().item() + 1e-6)
    iou = intersection / (union + 1e-6)

    return dice, iou


# ============================================================
# TRAINING LOOP
# ============================================================

def train(args):
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    print("=" * 60)
    print("E-EYE U-NET TRAINING & FINE-TUNING")
    print("=" * 60)
    print(f"Device        : {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Epochs        : {args.epochs}")
    print("=" * 60)

    dataset = PupilDataset(args.data_dir, augment=True)
    if len(dataset) == 0:
        print(f"No image/mask pairs found in {args.data_dir}!")
        print("Expected layout:")
        print(f"  {args.data_dir}/images/*.png")
        print(f"  {args.data_dir}/masks/*.png")
        print("Extract frames first with: python src/export_frames.py")
        return

    # Train / Val split (80 / 20)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=1, num_classes=1)

    # Load existing checkpoint if requested
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint for fine-tuning: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        sd = ckpt.get("state_dict", ckpt)
        unet_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(unet_sd or sd, strict=False)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = DiceBCELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, total_dice, total_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)

                dice, iou = compute_metrics(logits, masks)
                total_dice += dice * images.size(0)
                total_iou += iou * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_dice = total_dice / len(val_loader.dataset)
        val_iou = total_iou / len(val_loader.dataset)

        scheduler.step(val_dice)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = os.path.join(args.save_dir, f"pupil-unet-best-val_dice{val_dice:.4f}.ckpt")
            torch.save({
                "epoch": epoch,
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                "val_dice": val_dice,
                "hyper_parameters": {"lr": args.lr},
            }, ckpt_path)
            print(f"  --> Saved new best checkpoint: {ckpt_path}")

    print("=" * 60)
    print(f"Training completed! Best Validation Dice: {best_dice:.4f}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="Train E-Eye U-Net Pupil Detector")
    p.add_argument("--data-dir", "-d", default="../data/annotated", help="Path to annotated dataset folder")
    p.add_argument("--checkpoint", "-c", default="../checkpoints/pupil-unet-epoch48-val_dice0.7170.ckpt",
                   help="Existing checkpoint to fine-tune")
    p.add_argument("--save-dir", "-s", default="../checkpoints", help="Directory to save checkpoints")
    p.add_argument("--epochs", "-e", type=int, default=50, help="Training epochs")
    p.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
