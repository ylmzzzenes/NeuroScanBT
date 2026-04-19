"""İkili kanama sınıflandırması için ResNet-18."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18_binary(use_imagenet: bool = True) -> nn.Module:
    wts = ResNet18_Weights.DEFAULT if use_imagenet else None
    m = models.resnet18(weights=wts)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m
