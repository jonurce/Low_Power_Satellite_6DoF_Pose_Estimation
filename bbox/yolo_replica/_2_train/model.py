# model.py

import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["LIBPNG_NO_WARNINGS"] = "1"
os.environ["PNG_QUIET"] = "1"


import sys
# sys.stderr = open(os.devnull, 'w')

import warnings
# warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", message=".*duplicate.*")

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResidualBlock(nn.Module):
    """
    Current ResidualBlock → classic residual (good base)

    If I add split/concat + multiple bottlenecks inside → it becomes CSP-like (Cross Stage Partial)

    If I make the concat richer (all intermediate features) → it becomes C2f-like (YOLOv8 style, Cross Stage Partial with Feed-forward)
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels//2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels//2)
        self.conv2 = nn.Conv2d(channels//2, channels, kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)  # residual add


class EventBBNet(nn.Module):
    """
    Simple from-scratch CNN for bounding box regression on event images.
    Input: Event (1, H, W) normalized [0,1]
    Output: YOLO-like grid predictions [B, grid_h, grid_w, num_anchors, 8]
           where 8 = [cx, cy, w, h, obj_conf, class_0_prob, class_1_prob, class_2_prob],
           and class_prob {'cassini': 0, 'satty': 1, 'soho': 2}
    """
    def __init__(self, input_size=(720, 800), base_channels=32, K=3, num_classes=3):
        super().__init__()

        # Event backbone (1 input channel)
        self.backbone_simple = nn.Sequential(
            SimpleConvBlock(1,   base_channels,     pool=True),   # → /2
            SimpleConvBlock(base_channels,   base_channels*2,   pool=True),
            SimpleConvBlock(base_channels*2, base_channels*4,   pool=True),
            SimpleConvBlock(base_channels*4, base_channels*8,   pool=True),
            SimpleConvBlock(base_channels*8, base_channels*16,  pool=True),  # → /32
        )

        # Improved backbone with residuals and SPPF
        self.backbone_stages = nn.Sequential(
            # Stage 1: downsample + CSP block
            nn.Conv2d(1, base_channels, 3, stride=2, padding=1, bias=False),  # /2
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            ResidualBlock(base_channels),

            # Stage 2
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1, bias=False),  # /4
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            ResidualBlock(base_channels*2),

            # Stage 3
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1, bias=False),  # /8
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(),
            ResidualBlock(base_channels*4),

            # Stage 4 + SPPF (Spatial Pyramid Pooling - Fast)
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1, bias=False),  # /16
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(),
            ResidualBlock(base_channels*8),
        )

        self.sppf_max5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.sppf_max9 = nn.MaxPool2d(9, stride=1, padding=4)
        self.sppf_max13 = nn.MaxPool2d(13, stride=1, padding=6)
        self.sppf_conv = nn.Sequential(
            nn.Conv2d(base_channels*8 * 4, base_channels*8, 1, bias=False),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU()
        )

        self.backbone_final = nn.Sequential(
            # Final stage
            nn.Conv2d(base_channels*8, base_channels*16, 3, stride=2, padding=1, bias=False),  # /32
            nn.BatchNorm2d(base_channels*16),
            nn.ReLU(),
            ResidualBlock(base_channels*16),
        )

        # Number of predictions per grid cell
        self.K = K 
        self.num_classes = num_classes 

        # Output: a list of K elements, each being (5 + num_class): cx, cy, w, h, obj_conf, class_prob_0 ... class_prob_(num_class-1)
        self.head_out_channels = K * (4 + 1 + num_classes)  # 4 box params + 1 obj_conf + num_classes class probs
        
        # Detection head
        self.head = nn.Conv2d(
            base_channels * 16,   # input channels from backbone
            self.head_out_channels,  
            kernel_size=1,        # 1x1 conv
            stride=1,
            padding=0
        )

        # Bias for obj_conf & class_prob → start with higher logit: logit(0.99995) ≈ 4.0
        for k in range(self.K):
            nn.init.constant_(self.head.bias[4 + k * (4 + 1 + num_classes)], 10.0) 
            nn.init.constant_(self.head.bias[5 + k * (4 + 1 + num_classes)], 10.0) 
            nn.init.constant_(self.head.bias[6 + k * (4 + 1 + num_classes)], 10.0)
            nn.init.constant_(self.head.bias[7 + k * (4 + 1 + num_classes)], 10.0)

    def forward(self, event):
        # Feature extraction: [B, C=1, H, W] -> [B, C*16, H/32, W/32]
        feat = self.backbone_stages(event)  # [B, base*8, H/16, W/16]

        # SPPF parallel branches
        x1 = feat
        x2 = self.sppf_max5(feat)
        x3 = self.sppf_max9(feat)
        x4 = self.sppf_max13(feat)

        # Concat along channel dim
        concat = torch.cat([x1, x2, x3, x4], dim=1)  # [B, base*8*4, H/16, W/16]

        # Compress back
        feat = self.sppf_conv(concat)  # [B, base*8, H/16, W/16]

        feat = self.backbone_final(feat)  # [B, base*16, H/32, W/32]

        # Detection head: [B, C*16, H/32, W/32] -> [B, K*8, H/32, W/32]
        pred = self.head(feat) 

        # Get shape values from [B, K*8, H/32, W/32]
        B, C, gh, gw = pred.shape

        # Reshape to [B, K, 8, gh, gw]
        pred = pred.view(B, self.K, (4 + 1 + self.num_classes), gh, gw) 

        # Reshape from [B, K, 8, gh, gw] → [B, gh, gw, K, 8]
        pred = pred.permute(0, 3, 4, 1, 2).contiguous() 

        # Selective activation: [B, gh, gw, K, 8] -> [B, gh, gw, K, 8]
        pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])          # cx, cy → [0,1] (dim=-1)
        pred[..., 2:4] = torch.exp(pred[..., 2:4])              # w, h → positive & can be >>1
        pred[..., 4]  = torch.sigmoid(pred[..., 4])             # obj_conf → [0,1]
        pred[..., 5:] = torch.softmax(pred[..., 5:], dim=-1)    # class_prob → [0,1] sum to 1

        # Prediction: [B, gh, gw, K, 8] (for each grid -> K*8)
        return pred


# Quick test / usage example
if __name__ == "__main__":
    model = EventBBNet(input_size=(720, 800))
    event = torch.randn(2, 1, 720, 800)
    out = model(event)
    print("Output shape:", out.shape)          # e.g. [2, 40, 40, 6]
    print("Sample output min/max:", out.min().item(), out.max().item())