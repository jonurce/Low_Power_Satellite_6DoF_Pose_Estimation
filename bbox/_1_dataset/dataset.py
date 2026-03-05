# dataset.py

import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["LIBPNG_NO_WARNINGS"] = "1"

import sys
# sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore", module="PIL")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*libpng.*")
warnings.filterwarnings("ignore", message="libpng warning: eXIf: duplicate")
warnings.filterwarnings("ignore", message=".*eXIf: duplicate.*")

import cv2
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.transform import Rotation as R
import numpy as np


class SatelliteBBDataset(Dataset):
    def __init__(self, split='train'):
        """
        split:
            'train' -> synthetic 80%, with augmentations
            'val' -> synthetic 20%, no augmentations
            'test' -> real, no augmentations

        satellite: 'cassini', 'satty', 'soho'
        sequence: '1', '2', '3', or '4' (only in test with real data)
        distance: 'close' or 'far' (only in test with real data)

        """
        
        self.root_dir = "_dataset/"
        self.img_dirs = []
        self.label_dirs = []

        if split == 'train' or split == 'val':
            # Synthetic: combine all satellites
            label_file = 'train.json' if split == 'train' else 'test.json'
            for sat in ['cassini', 'satty', 'soho']:
                base = os.path.join(self.root_dir, 'synthetic', sat)
                self.img_dirs.append(os.path.join(base, 'frames'))
                self.label_dirs.append(os.path.join(base, label_file))
        else:
            # Real: combine all real folders (all sequences + distances)
            real_base = os.path.join(self.root_dir, 'real')
            for item in os.listdir(real_base):
                item_path = os.path.join(real_base, item)
                if os.path.isdir(item_path):
                    self.img_dirs.append(os.path.join(item_path, 'frames'))
                    self.label_dirs.append(os.path.join(item_path, 'test.json'))

        # Load labels
        self.labels = {"annotations": []}
        # dict {"annotations": [{"filename_rgb": 00001_rgb.png, "filename_event": 00001_event.png, 
            #       "bbox": [ x1, y1, x2, y2 ] }] }
        sat_to_class = {'cassini': 0, 'satty': 1, 'soho': 2}

        for img_dir, label_dir in zip(self.img_dirs, self.label_dirs):
            
            if not os.path.exists(label_dir):
                continue

            # Extract satellite name from path (e.g. from synthetic/cassini/...)
            # Assuming structure: .../synthetic/<sat>/... or .../real/<sat>-...
            sat = img_dir.split('/')[-2].split('-')[0] if split == 'test' else img_dir.split('/')[-2]
            
            if sat not in sat_to_class:
                continue  # skip unknown

            with open(label_dir, 'r') as f:
                data = json.load(f)
                for ann in data["annotations"]:
                    ann['filepath'] = img_dir
                    ann['class_id'] = sat_to_class[sat]
                    self.labels["annotations"].append(ann)
        
        
        # Only transform if split is train (separate for rgb and event frames)
        if split == 'train':
            # Common transform for RGB and event images: rotation, translation, gaussian blur
            self.common_transform = A.Compose([
                A.Rotate(limit=45, p=0.7, fill=128),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.6, fill=128),
                A.GaussianBlur(blur_limit=(3,7), p=0.4),
            ], 
            additional_targets={'event': 'image'},
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True),
            # pascal_voc: [x_min, y_min, x_max, y_max]
            # yolo: [cx, cy, w, h]
            )

            # RGB-specific augmentations: colojitter, uniformnoise, colornoise
            self.rgb_transform = A.Compose([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),   # uniform-like Gaussian noise
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),  # color noise proxy
                ToTensorV2() # transforms HWC image to CHW tensor for training/inference of the NN
            ])

            # Event-specific augmentations: ignore polarity, event noise, event patch noise (quadrilateral)
            self.event_transform = A.Compose([
                A.Lambda(image=lambda img,**kw: -img if np.random.rand() < 0.3 else img, p=0.3), # randomly flip sign of events (simple polarity noise)
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),   # uniform-like Gaussian noise
                A.PixelDropout(dropout_prob=0.01, p=0.4),  # drops ~1% of pixels to black
                A.OneOf([
                    A.Compose([
                        # Generate dropout mask (but don't apply fill yet)
                        A.CoarseDropout(
                            max_holes=6,
                            max_height=0.15,
                            max_width=0.15,
                            min_holes=2,
                            min_height=0.05,
                            min_width=0.05,
                            fill_value=128,           # grey mask
                            p=1.0,
                            always_apply=True       # force apply to create mask
                        ),
                        # Apply noise only where mask was dropped (i.e. inside patches)
                        A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                    ], p=0.5)
                ]),
                ToTensorV2()
            ])
        else:
            # Val/test: only normalization + ToTensor (no augmentations)
            self.common_transform = A.Compose([], 
                additional_targets={'event': 'image'},
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], clip=True),
                # pascal_voc: [x_min, y_min, x_max, y_max]
                # yolo: [cx, cy, w, h]
            )
            self.rgb_transform = A.Compose([
                ToTensorV2()
            ])
            self.event_transform = A.Compose([
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.labels["annotations"])

    def __getitem__(self, idx):

        # Get label
        label_dict = self.labels["annotations"][idx]

        # Get name, path and image for rgb
        rgb_name = label_dict["filename_rgb"]
        rgb_path = os.path.join(label_dict["filepath"], rgb_name)
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise FileNotFoundError(f"Image missing for {rgb_name}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # read image (BGR → RGB)

        # Get name, path and image for event
        event_name = label_dict["filename_event"]
        event_path = os.path.join(label_dict["filepath"], event_name)
        event_img = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)
        if event_img is None:
            raise FileNotFoundError(f"Image missing for {event_name}")
        
        # Get original bbox: [x1, y1, x2, y2] absolute pixels + check its correct
        bbox = label_dict['bbox']
        
        # Apply common augmentations
        augmented = self.common_transform(
            image=rgb_img,
            event=event_img,
            bboxes=[bbox], 
            labels=[0] # dummy class label (required by BboxParams)
        )

        # Apply RGB specific augmentations
        rgb_aug = self.rgb_transform(image=augmented['image'])
        rgb_img = rgb_aug['image']  # tensor, normalized -> rgb_img.shape = [3, H, W]

        # Apply event specific augmentations
        event_aug = self.event_transform(image=augmented['event'])
        event_img = event_aug['image']  # tensor, normalized -> event_img.shape = [1, H, W]

        # Convert augmented bbox to normalized YOLO format [0,1]
        x1, y1, x2, y2 = augmented['bboxes'][0]
        c, img_h, img_w = rgb_img.shape  # get original image size
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        bbox_yolo = np.array([cx, cy, w, h], dtype=np.float32)

        # Get class id: {'cassini': 0, 'satty': 1, 'soho': 2}
        class_id = label_dict.get('class_id', 0)  # default 0 if missing

        return rgb_img.float() / 255.0, event_img.float()/ 255.0, bbox_yolo, torch.tensor(class_id)