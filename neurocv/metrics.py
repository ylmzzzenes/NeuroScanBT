"""Sınıflandırma ve segmentasyon metrikleri (PyTorch tensör)."""
from __future__ import annotations

import numpy as np
import torch


def dice_iou_from_logits(logits: torch.Tensor, masks: torch.Tensor, thr: float = 0.5) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3))
    dice_d = 2 * inter + 1e-6
    dice_n = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6
    dice = (dice_d / dice_n).mean().item()
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()
    return float(dice), float(iou)


def mask_coverage(masks: torch.Tensor) -> float:
    return float(masks.mean().item())
