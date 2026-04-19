"""Sınıf dengesizliği ve segmentasyon kayıpları."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        t = targets
        inter = (probs * t).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) + self.smooth
        dice = (2 * inter + self.smooth) / denom
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """İkili sınıflandırma (logits: N,2)."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("w", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.w, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_w: float = 1.0, dice_w: float = 1.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = DiceLoss()(logits, targets)
    return bce_w * bce + dice_w * dice


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return bce_dice_loss(logits, targets, bce_w=1.0, dice_w=1.0)
