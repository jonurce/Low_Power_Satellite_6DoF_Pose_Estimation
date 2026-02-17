# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", message="libpng warning")
warnings.filterwarnings("ignore", message=".*eXIf: duplicate.*")

import os
os.environ["LIBPNG_NO_WARNINGS"] = "1"

class SimpleConvBlock(nn.Module):
    """Basic convolutional block: Conv → BN → ReLU → optional MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EventBBNet(nn.Module):
    """
    Simple from-scratch CNN for bounding box regression on event images.
    Input: Event (1, H, W) normalized [0,1]
    Output: YOLO-like grid predictions [B, grid_h, grid_w, num_anchors, 6]
           where 6 = [cx, cy, w, h, obj_conf, class_prob] (class always 0)
    """
    def __init__(self, input_size=(720, 800), base_channels=32):
        super().__init__()

        # Event backbone (1 input channel)
        self.backbone = nn.Sequential(
            SimpleConvBlock(1,   base_channels,     pool=True),   # → /2
            SimpleConvBlock(base_channels,   base_channels*2,   pool=True),
            SimpleConvBlock(base_channels*2, base_channels*4,   pool=True),
            SimpleConvBlock(base_channels*4, base_channels*8,   pool=True),
            SimpleConvBlock(base_channels*8, base_channels*16,  pool=True),  # → /32
        )

        # Detection head (simple 1×1 conv to predict per grid cell)
        # 5 + 1 class outputs: cx, cy, w, h, obj_conf, class_prob
        self.head = nn.Conv2d(
            base_channels*16,          # input channels from backbone
            6,                         # 4 box params + obj_conf + 1 class
            kernel_size=1,             # 1x1 conv
            stride=1,
            padding=0
        )

        # Bias for obj_conf & class_prob → start with higher logit (~0.5–0.9 prob)
        nn.init.constant_(self.head.bias[4], 4.0)  # logit(0.98) ≈ 4.0
        nn.init.constant_(self.head.bias[5], 2.0)  # logit(~0.88)


    def forward(self, event):
        # Feature extraction: [B, C=1, H, W] -> [B, C*16, H/32, W/32]
        feat = self.backbone(event) 

        # Detection head: [B, C*16, H/32, W/32] -> [B, 6, H/32, W/32]
        pred = self.head(feat) 

        # Selective activation: [B, 6, H/32, W/32] -> [B, 6, H/32, W/32]
        pred[:, :2] = torch.sigmoid(pred[:, :2])          # cx, cy → [0,1]
        pred[:, 2:4] = torch.exp(pred[:, 2:4])            # w, h → positive & can be >>1
        pred[:, 4:]  = torch.sigmoid(pred[:, 4:])         # obj_conf, class_prob → [0,1]

        # Reshape from [B, 6, H/32, W/32] -> [B, grid_h, grid_w, 6]
        B, C, gh, gw = pred.shape
        pred = pred.permute(0, 2, 3, 1).contiguous()

        # Prediction: [B, gh, gw, 6] (for each grid -> 6)
        return pred


# Quick test / usage example
if __name__ == "__main__":
    model = EventBBNet(input_size=(720, 800))
    event = torch.randn(2, 1, 720, 800)
    out = model(event)
    print("Output shape:", out.shape)          # e.g. [2, 40, 40, 6]
    print("Sample output min/max:", out.min().item(), out.max().item())