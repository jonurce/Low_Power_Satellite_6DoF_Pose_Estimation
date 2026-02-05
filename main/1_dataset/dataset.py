# dataset.py
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SatellitePoseDataset(Dataset):
    def __init__(self, split='train', satellite='cassini', sequence='1', distance='close'):
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
        self.split = split

        # Define paths, depending on train / val / test
        if 'train' in split:
            self.img_dir = os.path.join(self.root_dir, 'synthetic', self.satellite, 'frames')
            self.label_dir = os.path.join(self.root_dir, 'synthetic', self.satellite, 'train.json')
        else if 'val' in split:
            self.img_dir = os.path.join(self.root_dir, 'synthetic', self.satellite, 'frames')
            self.label_dir = os.path.join(self.root_dir, 'synthetic', self.satellite, 'test.json')
        else:
            self.img_dir = os.path.join(self.root_dir, 'real', f'{satellite}-{sequence}-{distance}', 'frames')
            self.label_dir = os.path.join(self.root_dir, 'real', f'{satellite}-{sequence}-{distance}', 'test.json')

        # Load labels
        with open(self.label_dir, 'r') as file:
            self.labels = json.load(file)  # dict {filename: {"q": [...], "r": [...]}} or similar

        # Load images
        self.img_files = [f for f in os.listdir(self.img_dir)
                        if f.endswith('.png') and f in self.labels]
        
        # Default transform if none provided
        if 'train' in split:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Read image (BGR → RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get label – adjust keys to match your JSON
        label_dict = self.labels[img_name]
        pose = label_dict['r'] + label_dict['q']   # [x,y,z, qw,qx,qy,qz]
        # or: pose = label_dict['pose'] if flattened

        pose = torch.tensor(pose, dtype=torch.float32)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']   # already tensor & normalized

        return img, pose