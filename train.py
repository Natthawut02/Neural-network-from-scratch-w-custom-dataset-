import os
import sys
import time
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import PSULaneDataset, CLASS_NAMES
from models.custom_lane_net import ResLaneSegNet
from utils.losses import WeightedFocalDiceLoss
from utils.metrics import SegmentationMetrics

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    num_batches = len(dataloader)

    t0 = time.time()
    for batch_idx, (images, masks, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss, l_focal, l_dice = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_focal += l_focal.item()
        total_dice += l_dice.item()

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch:02d}/{total_epochs:02d} | Batch {batch_idx+1:03d}/{num_batches:03d}] "
                  f"Loss: {loss.item():.4f} (Focal: {l_focal.item():.4f}, Dice: {l_dice.item():.4f}) | "
                  f"Elapsed: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches
    avg_focal = total_focal / num_batches
    avg_dice = total_dice / num_batches
    return avg_loss, avg_focal, avg_dice

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, metrics_tracker):
    model.eval()
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    num_batches = len(dataloader)
    metrics_tracker.reset()

    for images, masks, _ in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss, l_focal, l_dice = criterion(logits, masks)

        total_loss += loss.item()
        total_focal += l_focal.item()
        total_dice += l_dice.item()

        preds = torch.argmax(logits, dim=1)
        metrics_tracker.update(preds, masks)

    avg_loss = total_loss / num_batches
    avg_focal = total_focal / num_batches
    avg_dice = total_dice / num_batches
    metrics_result = metrics_tracker.compute()

    return avg_loss, avg_focal, avg_dice, metrics_result

def plot_and_save_curves(history: list, output_path: str):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_miou = [h["val_miou"] * 100 for h in history]
    val_acc = [h["val_acc"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, 'b-o', label='Train Weighted Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, val_loss, 'r-s', label='Val Weighted Loss', linewidth=2, markersize=4)
    ax1.set_title('PSU-LaneNet Loss Convergence (From Scratch)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Weighted Focal + Dice Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True)

    ax2.plot(epochs, val_miou, 'g-^', label='Val mIoU (%)', linewidth=2, markersize=4)
    ax2.plot(epochs, val_acc, 'm-d', label='Val Pixel Acc (%)', linewidth=2, markersize=4)
    ax2.set_title('Validation Metric Convergence', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Score (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Convergence curves saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train PSU-LaneNet")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_h", type=int, default=384)
    parser.add_argument("--img_w", type=int, default=640)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Enhanced Training on {device} ({args.img_w}x{args.img_h}) ===")

    img_size = (args.img_h, args.img_w)
    train_dataset = PSULaneDataset(args.data_dir, split="train", img_size=img_size, augment=True)
    val_dataset = PSULaneDataset(args.data_dir, split="val", img_size=img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = ResLaneSegNet(in_channels=3, num_classes=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResLaneSegNet (Total params: {total_params:,})")

    criterion = WeightedFocalDiceLoss(num_classes=6, gamma=2.0, weight_dice=1.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    metrics_tracker = SegmentationMetrics(num_classes=6, class_names=CLASS_NAMES)

    history = []
    best_miou = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch [{epoch:02d}/{args.epochs:02d}] (lr: {optimizer.param_groups[0]['lr']:.6f}) ---")

        train_loss, train_focal, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        val_loss, val_focal, val_dice, val_metrics = evaluate_epoch(
            model, val_loader, criterion, device, metrics_tracker
        )

        val_miou = val_metrics["mIoU"]
        val_acc = val_metrics["Pixel_Accuracy"]
        val_dice_score = val_metrics["mean_Dice"]

        print(f"  Summary Epoch {epoch:02d}: "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val mIoU: {val_miou*100:.2f}% | Val Pixel Acc: {val_acc*100:.2f}% | Mean Dice: {val_dice_score*100:.2f}%")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_focal": train_focal,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_focal": val_focal,
            "val_dice": val_dice,
            "val_miou": val_miou,
            "val_acc": val_acc,
            "val_mean_dice": val_dice_score,
            "lr": optimizer.param_groups[0]['lr']
        }
        history.append(epoch_record)

        if val_miou > best_miou:
            best_miou = val_miou
            best_ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "metrics": val_metrics
            }, best_ckpt_path)
            print(f"  --> [*] New Best Model Saved! Val mIoU: {best_miou*100:.2f}% at {best_ckpt_path}")

        scheduler.step()

    total_training_time = time.time() - start_time
    print(f"\n=== Training Complete in {total_training_time/60:.2f} mins! Best Val mIoU: {best_miou*100:.2f}% ===")

    latest_ckpt_path = os.path.join(args.save_dir, "latest_model.pth")
    torch.save({"model_state_dict": model.state_dict(), "history": history}, latest_ckpt_path)

    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_and_save_curves(history, "loss_convergence.png")

if __name__ == "__main__":
    main()
