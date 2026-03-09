# test_dataset.py
import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import AnimalsBBDataset  
import torch

# 1. Create dataset instances
train_ds = AnimalsBBDataset(split='train')  
val_ds = AnimalsBBDataset(split='valid')
test_ds = AnimalsBBDataset(split='test')

print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")

for idx in range (5):
    # 2. Pick one sample from each
    rgb_train,  target_train = train_ds[idx]
    rgb_val, target_valid = val_ds[idx]
    rgb_test, target_test = test_ds[idx]

    # Train
    if target_train.numel() > 0:
        bbox_train = target_train[0, :4].numpy()  # take first box (or loop if multi-object)
        class_id_train = target_train[0, 4].long()  # class of first box
    else:
        bbox_train = None
        class_id_train = torch.tensor(-1)

    # Valid
    if target_valid.numel() > 0:
        bbox_val = target_valid[0, :4].numpy()  # take first box (or loop if multi-object)
        class_id_val = target_valid[0, 4].long()  # class of first box
    else:
        bbox_val = None
        class_id_val = torch.tensor(-1)

    # Test
    if target_test.numel() > 0:
        bbox_test = target_test[0, :4].numpy()  # take first box (or loop if multi-object)
        class_id_test = target_test[0, 4].long()  # class of first box
    else:
        bbox_test = None
        class_id_test = torch.tensor(-1)

    # 3. Print shapes and values to check everything loaded correctly
    print("Train sample:")
    print(f"RGB shape: {rgb_train.shape}")  # expected: torch.Size([3, H, W])
    print(f"BBox: {bbox_train}") 
    print(f"Class ID: {class_id_train}")

    print("\nVal sample:")
    print(f"RGB shape: {rgb_val.shape}")
    print(f"BBox: {bbox_val}")
    print(f"Class ID: {class_id_val}")

    print("\nTest sample:")
    print(f"RGB shape: {rgb_test.shape}")
    print(f"BBox: {bbox_test}")
    print(f"Class ID: {class_id_test}")

    print("\nRGB traing min/max before imshow:", rgb_train.min().item(), rgb_train.max().item())
    print("\nRGB val min/max before imshow:", rgb_val.min().item(), rgb_val.max().item())
    print("\nRGB test min/max before imshow:", rgb_test.min().item(), rgb_test.max().item())

    # 4. Visualize the first few images
    def chw_to_hwc(img_tensor):
        img = img_tensor.permute(1, 2, 0).cpu().numpy() # CHW → HWC for plt
        # img = np.clip(img, 0, 255).astype(np.uint8)
        return img 

    def remove_channel(event_img):
        return event_img.squeeze(0).cpu().numpy()  # remove channel dim [1, H, W] → [H, W]

    def draw_bbox(ax, img_np, bbox, color='red', linewidth=2):
        if bbox is None or len(bbox) != 4:
            return
        h, w = img_np.shape[:2]
        cx, cy, bw, bh = bbox # (expects [cx, cy, w, h] normalized)
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=linewidth,
                        edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    class_names = {0: 'cat', 1: 'chicken', 2: 'cow', 3: 'dog', 4: 'fox', 5: 'goat',
                   6: 'horse', 7: 'person', 8: 'racoon', 9: 'skunk'}
    # ['cat', 'chicken', 'cow', 'dog', 'fox', 'goat', 'horse', 'person', 'racoon', 'skunk'] 


    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].imshow(chw_to_hwc(rgb_train))
    draw_bbox(axs[0], chw_to_hwc(rgb_train), bbox_train)
    axs[0].set_title(f"Train RGB - {class_names.get(class_id_train.item(), 'Unknown')}")

    axs[1].imshow(chw_to_hwc(rgb_val))
    draw_bbox(axs[1], chw_to_hwc(rgb_val), bbox_val)
    axs[1].set_title(f"Val RGB - {class_names.get(class_id_val.item(), 'Unknown')}")

    axs[2].imshow(chw_to_hwc(rgb_test))
    draw_bbox(axs[2], chw_to_hwc(rgb_test), bbox_test)
    axs[2].set_title(f"Test RGB - {class_names.get(class_id_test.item(), 'Unknown')}")

    plt.tight_layout()
    # plt.show()

    # Save instead of show
    model_dir = 'bbox/_1_dataset_animals/samples'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    plt.savefig(f"{model_dir}/{idx}.png", dpi=300, bbox_inches='tight')

    # Optional: close figure to free memory
    plt.close(fig)