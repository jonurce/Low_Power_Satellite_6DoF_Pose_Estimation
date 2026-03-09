# dataset.py

import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["LIBPNG_NO_WARNINGS"] = "1"
os.environ["PNG_QUIET"] = "1"


import sys
# sys.stderr = open(os.devnull, 'w')

import warnings
# warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", message=".*duplicate.*")

import cv2
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path

class AnimalsBBDataset(Dataset):
    def __init__(self, split='train', img_size=800):
        """
        Dataset from: https://universe.roboflow.com/roboflow-100/animals-ij5d2/dataset/2

        split:
            'train' -> 700 images
            'valid' -> 200 images
            'test' -> 100 images

        classes (10): ['cat', 'chicken', 'cow', 'dog', 'fox', 'goat', 'horse', 'person', 'racoon', 'skunk'] 

        """
        
        self.root_dir = "_dataset/animals"
        self.img_dir = os.path.join(self.root_dir, split, 'images')
        self.label_dir = os.path.join(self.root_dir, split, 'labels')

        # Get all image files
        self.img_files = sorted(Path(self.img_dir).glob('*.[jp][pn]g'))  # .jpg, .jpeg, .png
        
        if not self.img_files:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        
        
        # Only transform if split is train (separate for rgb and event frames)
        if split == 'train':
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            # pascal_voc: [x_min, y_min, x_max, y_max]
            # yolo: [cx, cy, w, h]
            
        else:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        img_path = self.img_files[idx]
        label_path = Path(os.path.join(self.label_dir, (img_path.stem + '.txt')))

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        boxes = []
        class_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.strip().split())
                    boxes.append([cx, cy, w, h])
                    class_labels.append(int(cls))
        
        boxes = np.array(boxes) if boxes else np.empty((0, 4))
        class_labels = np.array(class_labels) if class_labels else np.empty((0,))

        # Apply augmentations/transforms
        transformed = self.transform(
            image=img,
            bboxes=boxes,
            class_labels=class_labels
        )
        
        image = transformed['image']   # [C, H, W] tensor
        bboxes = np.array(transformed['bboxes']) if transformed['bboxes'].any() else np.empty((0, 4))
        labels = np.array(transformed['class_labels']) if transformed['class_labels'].any() else np.empty((0,))
        
        # Convert to YOLO target format: [cx, cy, w, h, class] or separate tensors
        if len(bboxes) > 0:
            target = np.hstack([bboxes, labels[:, None]])  # [N, 5]
            target = torch.from_numpy(target).float()
        else:
            target = torch.empty((0, 5), dtype=torch.float32)

        return image / 255.0, target