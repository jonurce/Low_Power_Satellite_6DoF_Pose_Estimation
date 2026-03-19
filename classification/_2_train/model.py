# model.py
import torch
import torch.nn as nn

from classification._2_train.layers_mmf_1 import MMFLinear, MMFConv2d

###################### YOLOv1 Classification ######################
class YOLOv1Classifier(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            nn.Conv2d(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            nn.Conv2d(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            nn.Conv2d(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            nn.Conv2d(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            nn.Linear(1024, num_classes)            # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs

###################### YOLOv1 Classification MMF ######################
class YOLOv1ClassifierMMF(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2d(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2d(64, 192, kernel_size=3, padding=1), #2
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2d(192, 128, kernel_size=1), #3
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(128, 256, kernel_size=3, padding=1), #4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 256, kernel_size=1), #5
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #6
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2d(512, 256, kernel_size=1), #7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #9
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #10
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #11
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #12
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #13
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 512, kernel_size=1), #15
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2d(1024, 512, kernel_size=1), #17
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #18
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024, 512, kernel_size=1), #19
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #20
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            MMFLinear(1024, num_classes),            # single fully connected layer
            nn.BatchNorm1d(num_classes),
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs
  
###################### YOLOv1 Bbox ######################
class YOLOv1Bbox(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            nn.Conv2d(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            nn.Conv2d(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            nn.Conv2d(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            nn.Conv2d(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Backbone yolo: 4 additional convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_yolo = nn.Sequential(

            # Conv21–22: 3x3x1024 + 3x3x1024-s-2
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), #21
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2), #22
            nn.LeakyReLU(0.1, inplace=True),

            # Conv23-24: 3x3x1024 + 3x3x1024
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), #23
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), #24
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # YOLO head (adapted from YOLOv1's FC layers)
        self.head_yolo = nn.Sequential(
            # will do this later
            nn.Flatten(),
            nn.Linear(1024 * 1 * 1, 4096),  # after all pools → 1×1 feature map
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        features = self.backbone_yolo(features)
        logits = self.head_yolo(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs
  
###################### YOLOv1 Bbox MMF ######################
class YOLOv1BboxMMF(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2d(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2d(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2d(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2d(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2d(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Backbone yolo: 4 additional convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_yolo = nn.Sequential(

            # Conv21–22: 3x3x1024 + 3x3x1024-s-2
            MMFConv2d(1024, 1024, kernel_size=3, padding=1), #21
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024, 1024, kernel_size=3, padding=1, stride=2), #22
            nn.LeakyReLU(0.1, inplace=True),

            # Conv23-24: 3x3x1024 + 3x3x1024
            MMFConv2d(1024, 1024, kernel_size=3, padding=1), #23
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024, 1024, kernel_size=3, padding=1), #24
            nn.LeakyReLU(0.1, inplace=True),
        )

        # YOLO head (adapted from YOLOv1's FC layers)
        self.head_yolo = nn.Sequential(
            # will do this later
            nn.Flatten(),
            MMFLinear(1024 * 1 * 1, 4096),  # after all pools → 1×1 feature map
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            MMFLinear(4096, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        features = self.backbone_yolo(features)
        logits = self.head_yolo(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs
  
   
# Quick test / usage example
if __name__ == "__main__":
    model = YOLOv1Classifier()
    image = torch.randn(512, 3, 32, 32)
    out = model(image)
    print("Output shape:", out.shape) 
    print("Sample output min/max:", out.min().item(), out.max().item())