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
            self.transform = T.Compose([
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
        else:
            self.transform = T.Compose([
                T.ToTensor(),
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