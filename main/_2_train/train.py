# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

from main._1_dataset.dataset import SatellitePoseDataset
from model import RGBEventPoseNet

def pose_loss(pred, target):
    """
    Combined pose loss:
    - MSE on translation (position)
    - Geodesic loss on unit quaternions (orientation)
    """
    # Translation error (Euclidean distance)
    pos_loss = nn.MSELoss()(pred[:, :3], target[:, :3])

    # Quaternion geodesic loss
    dot_product = torch.sum(pred[:, 3:] * target[:, 3:], dim=1)    
    dot_abs = torch.abs(dot_product)  # symmetry (q and -q represent same rotation)
    dot_clamped = torch.clamp(dot_abs, 0.0, 1.0) # numerical stability, avoid NaN in acos
    angle_rad = 2 * torch.acos(dot_clamped) # radians [0, π]
    angle_deg = angle_rad * (180.0 / torch.pi) # degrees (0, 180)
    rot_loss = angle_deg.pow(2).mean()

    # Combine
    total_loss = pos_loss + rot_loss

    return total_loss

def train_one_epoch(model, epoch, writer, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)
    running_loss = 0.0

    for batch_idx, (rgb, event, pose) in enumerate(tqdm(loader, desc="Training")):

        rgb, event, pose = rgb.to(device), event.to(device), pose.to(device)
        
        optimizer.zero_grad()
        pred = model(rgb, event)
        loss = criterion(pred, pose)
        loss.backward()
        optimizer.step()
        
        # Average losses per sample in the batch <- loss.item()

        # Tensorboard log loss
        global_step = epoch * len(loader) + batch_idx
        writer.add_scalar("Train/loss_batch", loss.item(), global_step)

        total_loss += loss.item()
        running_loss += loss.item()

        # Print running average every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_running = running_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx+1}/{num_batches} | Running Train Loss: {avg_running:.6f}")
    
    epoch_avg_loss = total_loss / num_batches
    return epoch_avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for rgb, event, pose in tqdm(loader, desc="Validation"):
            rgb, event, pose = rgb.to(device), event.to(device), pose.to(device)
            pred = model(rgb, event)
            loss = criterion(pred, pose)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets & Loaders
    train_ds = SatellitePoseDataset(split='train', satellite=args.satellite)
    val_ds   = SatellitePoseDataset(split='val',   satellite=args.satellite)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model
    model = RGBEventPoseNet().to(device)

    # Tensorboard logging
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs"))

    # Log model architecture in Tensorboard
    dummy_rgb   = torch.randn(1, 3, 720, 800).to(device)
    dummy_event = torch.randn(1, 1, 720, 800).to(device)
    writer.add_graph(model, (dummy_rgb, dummy_event))

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = pose_loss

    # Train with early stopping
    os.makedirs(args.save_dir, exist_ok=True)
    patience = 10   # number of epochs to wait after val loss stops improving
    min_delta_pct = 0.005   # 0.5% = minimum relative improvement required to reset patience
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = args.epochs    # safety upper limit (fallback if no convergence)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, epoch, writer, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)

        # Print loss
        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Log to Tensorboard
        writer.add_scalar("Train/loss_epoch", train_loss, epoch)
        writer.add_scalar("Val/loss_epoch",   val_loss,   epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss * (1 - min_delta_pct):
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"→ Improved! Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"→ No improvement ({epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs (no {min_delta_pct*100:.2f}%+ improvement for {patience} epochs)")
                break

        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth"))

    writer.close()


# python train.py --batch_size 8 --epochs 50 --lr 0.001 --satellite cassini
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RGB-Event 6D Pose Network")
    parser.add_argument("--batch_size",   type=int,   default=8,       help="Batch size")
    parser.add_argument("--epochs",       type=int,   default=50,      help="Number of epochs")
    parser.add_argument("--lr",           type=float, default=1e-3,     help="Learning rate")
    parser.add_argument("--satellite",    type=str,   default="cassini", help="Satellite name")
    parser.add_argument("--save_dir",     type=str,   default="checkpoints", help="Save directory")
    
    args = parser.parse_args()
    main(args)