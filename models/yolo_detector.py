import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class YOLODetector(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
        )

        # class heatmap prediction
        self.cls_head = nn.Conv2d(256, num_classes, 1)

        # box regression prediction: tx, ty, tw, th
        self.box_head = nn.Conv2d(256, 4, 1)

    def forward(self, x):
        feat = self.backbone(x)
        cls_map = self.cls_head(feat)
        box_map = self.box_head(feat)
        return cls_map, box_map
