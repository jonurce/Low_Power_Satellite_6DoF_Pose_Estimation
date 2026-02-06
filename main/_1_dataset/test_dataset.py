# test_dataset.py
import matplotlib.pyplot as plt
from dataset import SatellitePoseDataset  # adjust import if needed

# 1. Create dataset instances (use your real parameters)
train_ds = SatellitePoseDataset(
    split='train',
    satellite='cassini', 
    sequence='1',
    distance='close'
)

val_ds = SatellitePoseDataset(
    split='val',
    satellite='cassini',
    sequence='1',
    distance='close'
)

test_ds = SatellitePoseDataset(
    split='test',
    satellite='cassini',
    sequence='1',
    distance='close'
)

# 2. Pick one sample from train and one from val
idx = 32
rgb_train, event_train, pose_train = train_ds[idx]
rgb_val, event_val, pose_val = val_ds[idx]
rgb_test, event_test, pose_test = test_ds[idx]

# 3. Print shapes and values to check everything loaded correctly
print("Train sample:")
print(f"RGB shape: {rgb_train.shape}")         # expected: torch.Size([3, H, W])
print(f"Event shape: {event_train.shape}")
print(f"Pose: {pose_train}")                   # 7 values: [x,y,z, qx,qy,qz,qw]

print("\nVal sample:")
print(f"RGB shape: {rgb_val.shape}")
print(f"Event shape: {event_val.shape}")
print(f"Pose: {pose_val}")

print("\nTest sample:")
print(f"RGB shape: {rgb_test.shape}")
print(f"Event shape: {event_test.shape}")
print(f"Pose: {pose_test}")

print("\nEvent val min/max before imshow:", event_val.min().item(), event_val.max().item())
#print("Mean value:", event_val.mean().item())

print("\nEvent test min/max before imshow:", event_test.min().item(), event_test.max().item())
#print("Mean value:", event_test.mean().item())

# 4. Visualize the first few images
def chw_to_hwc(img_tensor):
    return img_tensor.permute(1, 2, 0).numpy()  # CHW → HWC for plt

def remove_channel(event_img):
    return event_img.squeeze(0).cpu().numpy()  # remove channel dim [1, H, W] → [H, W]

fig, axs = plt.subplots(3, 2, figsize=(10, 8))

axs[0,0].imshow(chw_to_hwc(rgb_train))
axs[0,0].set_title("Train RGB")
axs[0,1].imshow(remove_channel(event_train), cmap='gray', vmin=0, vmax=255)
axs[0,1].set_title("Train Event")

axs[1,0].imshow(chw_to_hwc(rgb_val))
axs[1,0].set_title("Val RGB")
axs[1,1].imshow(remove_channel(event_val), cmap='gray', vmin=0, vmax=255)
axs[1,1].set_title("Val Event")

axs[2,0].imshow(chw_to_hwc(rgb_test))
axs[2,0].set_title("Test RGB")
axs[2,1].imshow(remove_channel(event_test), cmap='gray', vmin=0, vmax=255)
axs[2,1].set_title("Test Event")

plt.tight_layout()
plt.show()