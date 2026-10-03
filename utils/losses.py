import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    """
    Multi-Class Dice Loss:
    Directly maximizes spatial overlap (IoU) across all classes.
    """
    def __init__(self, num_classes: int = 6, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score
        return dice_loss.mean()

class WeightedFocalDiceLoss(nn.Module):
    """
    Weighted Focal Loss + Multi-Class Dice Loss:
    Specifically tackles severe class imbalance where thin lane markings represent < 2% of pixels.
    Applies heavy class weighting (up to 5x) for Track_Left, Track_Center, and Track_Right.
    """
    def __init__(self, num_classes: int = 6, gamma: float = 2.0, weight_dice: float = 1.5):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.weight_dice = weight_dice
        self.dice = MultiClassDiceLoss(num_classes=num_classes)

        # Class weights: Background=0.4, Line_L=3.5, Line_C=4.0, Line_R=3.5, Lane=1.0, Sideway=1.2
        weights = torch.tensor([0.4, 3.5, 4.0, 3.5, 1.0, 1.2], dtype=torch.float32)
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: (B, C, H, W)
        # targets: (B, H, W)
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")  # (B, H, W)
        pt = torch.exp(-ce_loss)  # Probability of target class
        focal_loss = (((1.0 - pt) ** self.gamma) * ce_loss).mean()

        dice_loss = self.dice(logits, targets)
        total_loss = focal_loss + self.weight_dice * dice_loss

        return total_loss, focal_loss, dice_loss
