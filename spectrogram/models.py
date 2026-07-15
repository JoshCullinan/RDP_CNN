"""Backbone loader + small from-scratch capacity floor."""
from __future__ import annotations
import timm
import torch
import torch.nn as nn
from spectrogram.config import BACKBONE

def in_channels_for(arm: str) -> int:
    return {"A0": 3, "A1": 3, "A2": 12, "A3": 3, "A4": 3}[arm]

def build_backbone(in_ch: int, pretrained: bool, name: str = BACKBONE, n_classes: int = 3):
    return timm.create_model(name, pretrained=pretrained, in_chans=in_ch,
                             num_classes=n_classes)

class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 3):
        super().__init__()
        chs = [in_ch, 32, 64, 128, 128]
        blocks = []
        for a, b in zip(chs[:-1], chs[1:]):
            blocks += [nn.Conv2d(a, b, 3, stride=2, padding=1),
                       nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(chs[-1], n_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=(2, 3))
        return self.head(x)
