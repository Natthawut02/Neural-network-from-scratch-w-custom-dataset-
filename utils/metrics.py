import torch
import numpy as np

class SegmentationMetrics:
    """
    Computes confusion matrix and derives per-class IoU, mIoU, Dice (F1), and Pixel Accuracy.
    """
    def __init__(self, num_classes: int = 6, class_names: dict = None):
        self.num_classes = num_classes
        self.class_names = class_names or {i: f"Class_{i}" for i in range(num_classes)}
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds: (B, H, W) or (B, C, H, W)
        targets: (B, H, W)
        """
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        mask = (targets_np >= 0) & (targets_np < self.num_classes)
        hist = np.bincount(
            self.num_classes * targets_np[mask].astype(int) + preds_np[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += hist

    def compute(self) -> dict:
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fn = np.sum(cm, axis=1) - tp
        fp = np.sum(cm, axis=0) - tp

        denominator_iou = tp + fp + fn
        iou_per_class = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            if denominator_iou[i] > 0:
                iou_per_class[i] = tp[i] / denominator_iou[i]
            else:
                iou_per_class[i] = np.nan

        denominator_dice = 2 * tp + fp + fn
        dice_per_class = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            if denominator_dice[i] > 0:
                dice_per_class[i] = (2 * tp[i]) / denominator_dice[i]
            else:
                dice_per_class[i] = np.nan

        # Overall Pixel Accuracy
        total_pixels = np.sum(cm)
        pixel_acc = np.sum(tp) / total_pixels if total_pixels > 0 else 0.0

        # Mean IoU and Dice (ignoring NaN classes if any)
        valid_iou = iou_per_class[~np.isnan(iou_per_class)]
        miou = float(np.mean(valid_iou)) if len(valid_iou) > 0 else 0.0

        valid_dice = dice_per_class[~np.isnan(dice_per_class)]
        mean_dice = float(np.mean(valid_dice)) if len(valid_dice) > 0 else 0.0

        per_class_summary = {}
        for i in range(self.num_classes):
            c_name = self.class_names.get(i, f"Class_{i}")
            per_class_summary[c_name] = {
                "IoU": float(iou_per_class[i]) if not np.isnan(iou_per_class[i]) else 0.0,
                "Dice": float(dice_per_class[i]) if not np.isnan(dice_per_class[i]) else 0.0
            }

        return {
            "mIoU": miou,
            "mean_Dice": mean_dice,
            "Pixel_Accuracy": float(pixel_acc),
            "per_class": per_class_summary
        }
