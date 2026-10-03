import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import PSULaneDataset, CLASS_NAMES
from models.custom_lane_net import ResLaneSegNet
from utils.metrics import SegmentationMetrics

def evaluate_model(model_path: str, data_dir: str, split: str = "test", batch_size: int = 8, img_size=(288, 512)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating {model_path} on {split.upper()} set ({device}) ===")

    # 1. Load Dataset
    dataset = PSULaneDataset(data_dir, split=split, img_size=img_size, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Load Model
    model = ResLaneSegNet(in_channels=3, num_classes=6).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. Evaluate Metrics
    metrics_tracker = SegmentationMetrics(num_classes=6, class_names=CLASS_NAMES)
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            metrics_tracker.update(preds, masks)

    results = metrics_tracker.compute()

    # 4. Print Formatted Table
    print("\n" + "=" * 65)
    print(f"{'Class Name':<18} | {'IoU (%)':<12} | {'Dice / F1 (%)':<15}")
    print("-" * 65)
    for c_id, c_name in CLASS_NAMES.items():
        c_stats = results["per_class"][c_name]
        iou_pct = c_stats["IoU"] * 100
        dice_pct = c_stats["Dice"] * 100
        print(f"{c_name:<18} | {iou_pct:<12.2f} | {dice_pct:<15.2f}")
    print("-" * 65)
    print(f"{'OVERALL mIoU':<18} : {results['mIoU'] * 100:.2f}%")
    print(f"{'OVERALL Mean Dice':<18} : {results['mean_Dice'] * 100:.2f}%")
    print(f"{'Pixel Accuracy':<18} : {results['Pixel_Accuracy'] * 100:.2f}%")
    print("=" * 65 + "\n")

    # Save to JSON
    out_file = f"evaluation_results_{split}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate ResLaneSegNet")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=640)
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, split=args.split, batch_size=args.batch_size, img_size=(args.img_h, args.img_w))

if __name__ == "__main__":
    main()
