# dataset.py

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as T

class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 classification dataset (32×32 RGB images, 10 classes).
    
    Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    
    split: 'train' (50,000 images) or 'test' (10,000 images)
    """
    def __init__(self, split='train', root='_dataset/cifar10'):
        """
        root: directory where CIFAR-10 will be downloaded/extracted
        """
        self.split = split
        self.root = root
        
        if split == 'train':

            # Original transform
            self.transform_original = T.Compose([
                # Random scaling and translation up to 20% of image size
                T.RandomAffine(
                    degrees=0,                    # no rotation
                    translate=(0.2, 0.2),         # up to ±20% shift in x and y
                    scale=(0.8, 1.2),             # scaling between 80% and 120%
                    shear=0,
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0
                ),
                
                # Random exposure (brightness) and saturation adjustment up to 1.5× in HSV
                T.ColorJitter(
                    brightness=(1/1.5, 1.5),     # exposure/brightness: 0.666× to 1.5×
                    contrast=0,                   # no contrast change
                    saturation=(1/1.5, 1.5),      # saturation: 0.666× to 1.5×
                    hue=0                         # no hue change
                ),
                
                T.ToTensor(),
            ])

            # Improved transform
            self.transform_improved = T.Compose([
                # 1. Random scaling and translation up to 20%
                T.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    shear=0,
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0
                ),
                
                # 2. Stronger exposure/saturation + add contrast & small hue
                T.ColorJitter(
                    brightness=(0.5, 1.5),     # wider range: 0.5× to 1.5×
                    contrast=(0.7, 1.3),       # add mild contrast jitter
                    saturation=(0.5, 1.5),     # wider saturation range
                    hue=(-0.05, 0.05)          # small hue shift (safe on CIFAR)
                ),
                
                # 3. Add random horizontal flip (very effective on CIFAR)
                T.RandomHorizontalFlip(p=0.5),

                T.ToTensor(),
                
                # 4. Add random erasing (cutout) – huge gain on CIFAR
                T.RandomErasing(
                    p=0.4,                     # probability
                    scale=(0.02, 0.25),        # erase 2–25% of area
                    ratio=(0.3, 3.3),          # aspect ratio range
                    value="random"             # random pixel values
                ),

                # 5. Normalization (must be last)
                T.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                ),
                
            ])

            # Even more improved transform
            self.transform= T.Compose([
                # 1. Geometric augmentations (stronger than before)
                T.RandomAffine(
                    degrees=15,                   # ← added mild rotation (±15°)
                    translate=(0.2, 0.2),
                    scale=(0.75, 1.25),           # slightly wider: 75–125%
                    shear=10,                     # ← added small shear (±10°)
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=0
                ),
                
                # 2. Strong color jitter (wider ranges)
                T.ColorJitter(
                    brightness=(0.4, 1.6),        # wider exposure
                    contrast=(0.5, 1.5),
                    saturation=(0.4, 1.6),
                    hue=(-0.08, 0.08)             # a bit more hue variation
                ),
                
                # 3. Flip + very effective on CIFAR
                T.RandomHorizontalFlip(p=0.5),

                # 4. AutoAugment or RandAugment policy (very strong gain on CIFAR)
                T.RandAugment(                    # ← new: RandAugment (default policy)
                    num_ops=2,
                    magnitude=9
                ),
                
                T.ToTensor(),
                
                # 5. Cutout / erasing — stronger probability & range
                T.RandomErasing(
                    p=0.5,                        # ↑ higher probability
                    scale=(0.02, 0.4),            # larger possible erase area
                    ratio=(0.3, 3.3),
                    value="random"
                ),
                
                
                # 6. Normalization (must be last)
                T.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                ),
            ])

        else:
            self.transform = T.Compose([
                T.ToTensor(),
                # 6. Normalization (must be last)
                T.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                ),
            ])
        
        # Load CIFAR-10 (downloads automatically if missing)
        self.dataset = datasets.CIFAR10(
            root=root,
            train=(split == 'train'),
            download=True,
            transform=self.transform
        )
        
        print(f"Loaded CIFAR-10 {split} split: {len(self.dataset)} images")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]  # img is already tensor [3,32,32], label is int 0-9
        return img, label