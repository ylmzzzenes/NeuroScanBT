"""Kompakt U-Net (BT kesitleri, 3 kanal giriş, tek kanal ikili maske çıkışı)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(base, base * 2)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(base * 2, base * 4)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(base * 4, base * 8)
        self.down4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.bottleneck(self.down4(x4))

        u = self.up4(x5)
        u = self._crop_cat(u, x4)
        u = self.dec4(u)
        u = self.up3(u)
        u = self._crop_cat(u, x3)
        u = self.dec3(u)
        u = self.up2(u)
        u = self._crop_cat(u, x2)
        u = self.dec2(u)
        u = self.up1(u)
        u = self._crop_cat(u, x1)
        u = self.dec1(u)
        return self.outc(u)

    @staticmethod
    def _crop_cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        _, _, h, w = up.shape
        skip = skip[:, :, :h, :w]
        return torch.cat([up, skip], dim=1)
