# test_dataset.py
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import CIFAR10Dataset  # your new CIFAR-10 class

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 1. Create dataset instances
train_ds = CIFAR10Dataset(split='train')
test_ds  = CIFAR10Dataset(split='test')  

print(f"Train samples: {len(train_ds)}, Val/Test samples: {len(test_ds)}")

for idx in range(10):  # show first 5 samples
    # 2. Pick one sample from each
    img_train, label_train = train_ds[idx]
    img_test,  label_test  = test_ds[idx]

    # 3. Print shapes and values
    print(f"\nSample {idx}:")
    print(f"Train image shape: {img_train.shape}")  # [3, 32, 32]
    print(f"Train label: {label_train} ({class_names[label_train]})")

    print(f"Test image shape: {img_test.shape}")
    print(f"Test label: {label_test} ({class_names[label_test]})")

    print(f"Train min/max: {img_train.min().item():.3f} / {img_train.max().item():.3f}")

    # 4. Visualize
    def chw_to_hwc(img_tensor):
        return img_tensor.permute(1, 2, 0).cpu().numpy()  # [32,32,3]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].imshow(chw_to_hwc(img_train))
    axs[0].set_title(f"Train: {class_names[label_train]}")
    axs[0].axis('off')

    axs[1].imshow(chw_to_hwc(img_test))
    axs[1].set_title(f"Test: {class_names[label_test]}")
    axs[1].axis('off')

    plt.tight_layout()

    # Save
    save_dir = 'classification/_1_dataset/samples'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/sample_{idx}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)