import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# Mapping from YOLO class IDs in dataset to Semantic Mask class IDs
# YOLO classes from data.yaml:
#   0: Line_R
#   1: Line_C
#   2: Line_L
#   3: Sideway
#   4: Lane
# Target Mask classes:
#   0: Background
#   1: Line_L
#   2: Line_C
#   3: Line_R
#   4: Lane
#   5: Sideway
YOLO_TO_MASK_CLASS = {
    0: 3,  # Line_R -> class 3
    1: 2,  # Line_C -> class 2
    2: 1,  # Line_L -> class 1
    3: 5,  # Sideway -> class 5
    4: 4   # Lane -> class 4
}

# Drawing order to ensure thin line markings are drawn crisp on top of the lane area:
# Sideway(5) -> Lane(4) -> Line_L(1) -> Line_C(2) -> Line_R(3)
DRAW_ORDER = [
    (3, 5),  # Sideway first
    (4, 4),  # Lane on top of background / sideway
    (2, 1),  # Line_L on top of lane
    (1, 2),  # Line_C on top of lane
    (0, 3)   # Line_R on top of lane
]

CLASS_NAMES = {
    0: "Background",
    1: "Line_L",
    2: "Line_C",
    3: "Line_R",
    4: "Lane",
    5: "Sideway"
}

# Distinct bright colors for clear visual separation
CLASS_COLORS_RGB = {
    0: (0, 0, 0),         # Background: Pure black
    1: (255, 0, 0),       # Line_L: Red
    2: (255, 255, 0),     # Line_C: Yellow
    3: (255, 140, 0),     # Line_R: Orange
    4: (0, 220, 60),      # Lane: Bright Green
    5: (220, 30, 220)     # Sideway: Magenta
}

def rasterize_polygons(lbl_path: str, target_h: int, target_w: int) -> np.ndarray:
    """
    Renders normalized polygon coordinates from YOLO annotation file into a 2D integer mask (H, W).
    """
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    if not os.path.exists(lbl_path):
        return mask

    polys_by_cls = {0: [], 1: [], 2: [], 3: [], 4: []}
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls_id = int(parts[0])
            if cls_id not in polys_by_cls:
                continue
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] = np.clip(coords[:, 0] * target_w, 0, target_w - 1)
            coords[:, 1] = np.clip(coords[:, 1] * target_h, 0, target_h - 1)
            pts = np.round(coords).astype(np.int32)
            polys_by_cls[cls_id].append(pts)

    # Render in DRAW_ORDER to ensure thin track lines remain on top of lane surface
    for yolo_cls, mask_cls in DRAW_ORDER:
        for pts in polys_by_cls[yolo_cls]:
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], int(mask_cls))

    return mask

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Converts a 2D class-indexed mask (H, W) into an RGB image (H, W, 3)."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_RGB.items():
        color_mask[mask == cls_id] = color
    return color_mask

class PSULaneDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train", img_size=(384, 640), augment: bool = False):
        self.root_dir = root_dir
        self.split = split
        self.target_h, self.target_w = img_size
        self.augment = augment

        img_dir = os.path.join(root_dir, "images", split)
        lbl_dir = os.path.join(root_dir, "labels", split)

        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.lbl_paths = []
        for p in self.img_paths:
            base_name = os.path.splitext(os.path.basename(p))[0]
            lbl_file = os.path.join(lbl_dir, f"{base_name}.txt")
            self.lbl_paths.append(lbl_file)

        assert len(self.img_paths) > 0, f"No images found in {img_dir}"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        lbl_path = self.lbl_paths[idx]

        # Read image (BGR -> RGB)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Rasterize polygon labels to mask at target resolution
        mask = rasterize_polygons(lbl_path, self.target_h, self.target_w)
        img_resized = cv2.resize(img_rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        # Data augmentation during training
        if self.augment:
            # Horizontal flip: swap left boundary (1) with right boundary (3)
            if np.random.rand() > 0.5:
                img_resized = np.fliplr(img_resized).copy()
                mask = np.fliplr(mask).copy()
                left_mask = (mask == 1)
                right_mask = (mask == 3)
                mask[left_mask] = 3
                mask[right_mask] = 1

            # Random brightness & contrast
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.9, 1.1)
                beta = np.random.uniform(-10, 10)
                img_resized = np.clip(alpha * img_resized + beta, 0, 255).astype(np.uint8)

        # Normalize
        img_norm = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std

        tensor_img = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy(mask).long()

        return tensor_img, tensor_mask, os.path.basename(img_path)
