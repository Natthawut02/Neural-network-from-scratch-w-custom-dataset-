import os
import sys
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import CLASS_NAMES, CLASS_COLORS_RGB, colorize_mask, rasterize_polygons
from models.custom_lane_net import ResLaneSegNet

def create_prediction_snapshot(
    model, img_path: str, lbl_path: str, output_path: str,
    device, target_size=(384, 640)
):
    target_h, target_w = target_size

    # 1. Read Original Image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 2. Ground Truth Mask (from polygon)
    gt_mask = rasterize_polygons(lbl_path, target_h, target_w)
    gt_colored = colorize_mask(gt_mask)

    # Ground truth overlay
    gt_overlay = img_resized.copy()
    non_bg_gt = gt_mask > 0
    gt_overlay[non_bg_gt] = cv2.addWeighted(img_resized[non_bg_gt], 0.45, gt_colored[non_bg_gt], 0.55, 0)

    # 3. Model Inference
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    tensor_in = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor_in)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_colored = colorize_mask(pred_mask)

    # 4. Predicted Overlay Blend
    pred_overlay = img_resized.copy()
    non_bg_pred = pred_mask > 0
    pred_overlay[non_bg_pred] = cv2.addWeighted(img_resized[non_bg_pred], 0.45, pred_colored[non_bg_pred], 0.55, 0)

    # 5. Composite 2x2 Grid Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    plt.subplots_adjust(wspace=0.04, hspace=0.12)

    # Panel 1: Original
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title("(A) Input Road Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: Ground Truth Overlay
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("(B) Ground Truth Mask Overlay", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 3: Semantic Mask on Pure Black (Crisp visual lines)
    axes[1, 0].imshow(pred_colored)
    axes[1, 0].set_title("(C) PSU-LaneNet Semantic Prediction", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 4: Prediction Overlay
    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title("(D) Lane Inference Overlay", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    # Add Class Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label=f"{name}",
                   markerfacecolor=[c / 255.0 for c in color], markersize=12)
        for cls_id, (name, color) in enumerate(zip(CLASS_NAMES.values(), CLASS_COLORS_RGB.values()))
        if cls_id > 0
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=5,
        fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Snapshot saved to: {output_path}")

def generate_snapshots(
    model_path: str = "checkpoints/best_model.pth",
    data_dir: str = "dataset",
    output_dir: str = "snapshots",
    num_samples: int = 5
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating snapshots using model: {model_path} on {device}...")

    model = ResLaneSegNet(in_channels=3, num_classes=6).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    test_img_dir = os.path.join(data_dir, "images", "test")
    test_lbl_dir = os.path.join(data_dir, "labels", "test")
    img_files = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))

    indices = np.linspace(0, len(img_files) - 1, num_samples, dtype=int)
    for i, idx in enumerate(indices, 1):
        img_path = img_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(test_lbl_dir, f"{base_name}.txt")
        out_name = f"snapshot_{i}_{base_name}.png"
        out_path = os.path.join(output_dir, out_name)
        create_prediction_snapshot(model, img_path, lbl_path, out_path, device, target_size=(384, 640))

if __name__ == "__main__":
    generate_snapshots()
