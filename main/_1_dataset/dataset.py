# dataset.py
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.transform import Rotation as R
import numpy as np

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

        # Define paths, depending on train / val / test
        if 'train' in split:
            self.img_dir = os.path.join(self.root_dir, 'synthetic', satellite, 'frames') # 00001_rgb.png , 00001_event.png, ...
            self.label_dir = os.path.join(self.root_dir, 'synthetic', satellite, 'train.json')
        elif 'val' in split:
            self.img_dir = os.path.join(self.root_dir, 'synthetic', satellite, 'frames')
            self.label_dir = os.path.join(self.root_dir, 'synthetic', satellite, 'test.json')
        else:
            self.img_dir = os.path.join(self.root_dir, 'real', f'{satellite}-{sequence}-{distance}', 'frames')
            self.label_dir = os.path.join(self.root_dir, 'real', f'{satellite}-{sequence}-{distance}', 'test.json')

        # Load labels
        with open(self.label_dir, 'r') as file:
            self.labels = json.load(file) 
            # dict {"annotations": [{"filename_rgb": 00001_rgb.png, "filename_event": 00001_event.png, 
            #       "pose": [ [0,0,0,0], [...], [...], [...] ] (rotation matrix by rows) } ]}
        
        # Only transform if split is train (separate for rgb and event frames)
        if split == 'train':
            # Common transform for RGB and event images: rotation, translation, gaussian blur
            self.common_transform = A.Compose([
                A.Rotate(limit=45, p=0.7, fill=128),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.6, fill=128),
                A.GaussianBlur(blur_limit=(3,7), p=0.4),
            ], additional_targets={'event': 'image'})

            # RGB-specific augmentations: colojitter, uniformnoise, colornoise
            self.rgb_transform = A.Compose([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),   # uniform-like Gaussian noise
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),  # color noise proxy
                ToTensorV2() # converts uint8 [0, 255] → float32 and divides by 255 [0,1], and transforms it to a tensor for training/inference of the NN
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
            self.common_transform = A.Compose([], additional_targets={'event': 'image'})
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
        rgb_path = os.path.join(self.img_dir, rgb_name)
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise FileNotFoundError(f"Image missing for {rgb_name}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # read image (BGR → RGB)

        # Get name, path and image for event
        event_name = label_dict["filename_event"]
        event_path = os.path.join(self.img_dir, event_name)
        event_img = cv2.imread(event_path, cv2.IMREAD_GRAYSCALE)
        if event_img is None:
            raise FileNotFoundError(f"Image missing for {event_name}")
        
        # Extract flattened 4x4 pose (list of 16 floats)
        pose_flat = np.array(label_dict['pose'], dtype=np.float32)

        # Reshape to 4x4 matrix (row-major)
        T = pose_flat.reshape(4, 4)

        # Extract translation (right column, first 3 rows)
        translation = T[:3, 3]   # shape (3,)

        # Extract rotation matrix (top-left 3x3)
        rot_mat = T[:3, :3]      # shape (3,3)

        # Convert rotation matrix → quaternion (order: x, y, z, w)
        rot = R.from_matrix(rot_mat)
        quat_xyzw = rot.as_quat()   # [qx, qy, qz, qw]

        # Optional: ensure unit norm (usually already is)
        quat_norm = np.linalg.norm(quat_xyzw)
        if quat_norm > 0:
            quat_xyzw /= quat_norm

        # Combine: [x, y, z, qx, qy, qz, qw]
        pose = np.concatenate([translation, quat_xyzw])
        pose = torch.tensor(pose, dtype=torch.float32)

        # Apply augmentations
        augmented = self.common_transform(image=rgb_img, event=event_img)

        rgb_aug = self.rgb_transform(image=augmented['image'])
        rgb_img = rgb_aug['image']  # tensor, normalized -> rgb_img.shape = [3, H, W]

        event_aug = self.event_transform(image=augmented['event'])
        event_img = event_aug['image']  # tensor, normalized -> event_img.shape = [1, H, W]

        return rgb_img, event_img, pose