# train.py

####################### CONTINUE HERE #######################


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import argparse

from main._1_dataset.dataset import SatellitePoseDataset
from model import RGBEventPoseNet

def pose_loss(pred, target):
    """Combined loss: MSE for translation + L1 for quaternion (or geodesic if needed)"""
    pos_loss = nn.MSELoss()(pred[:, :3], target[:, :3])
    quat_loss = nn.L1Loss()(pred[:, 3:], target[:, 3:])
    return pos_loss + quat_loss  # can weight: pos_loss + 0.5 * quat_loss

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for rgb, event, pose in tqdm(loader, desc="Training"):
        rgb, event, pose = rgb.to(device), event.to(device), pose.to(device)
        
        optimizer.zero_grad()
        pred = model(rgb, event)
        loss = criterion(pred, pose)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

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
    model = RGBEventPoseNet(input_size=(720, 800)).to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = pose_loss

    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"→ Saved best model (val_loss: {val_loss:.6f})")

        # Periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RGB-Event 6D Pose Network")
    parser.add_argument("--batch_size",   type=int,   default=8,       help="Batch size")
    parser.add_argument("--epochs",       type=int,   default=50,      help="Number of epochs")
    parser.add_argument("--lr",           type=float, default=1e-3,     help="Learning rate")
    parser.add_argument("--satellite",    type=str,   default="cassini", help="Satellite name")
    parser.add_argument("--save_dir",     type=str,   default="checkpoints", help="Save directory")
    
    args = parser.parse_args()
    main(args)