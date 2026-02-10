# train.py
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from boundingbox._1_dataset.dataset import SatelliteBBDataset
from model import EventBBNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
warnings.filterwarnings("ignore", message="libpng warning")
warnings.filterwarnings("ignore", message=".*eXIf: duplicate.*")

import os
os.environ["LIBPNG_NO_WARNINGS"] = "1"

##################### YOLO-style loss (simplified for single-class) #####################
def yolo_loss(pred, target, lambda_box=5.0, lambda_obj=1.0, lambda_cls=0.5):
    """
    Simplified YOLO loss for single-class detection.
    pred: [B, gh, gw, 6] → cx, cy, w, h, obj_conf, class_prob
    target: [B, 4] → normalized [cx, cy, w, h] (one bbox per image for simplicity)
    """
    B, gh, gw, _ = pred.shape
    device = pred.device

    # Create grid offsets
    grid_x = torch.arange(gw, device=device).view(1, 1, gw, 1).expand(B, gh, gw, 1)
    grid_y = torch.arange(gh, device=device).view(1, gh, 1, 1).expand(B, gh, gw, 1)

    # Sigmoid activations for cx/cy/obj/class
    pred[..., :2] = torch.sigmoid(pred[..., :2])   # cx, cy offsets
    pred[..., 4:] = torch.sigmoid(pred[..., 4:])   # obj_conf, class_prob

    # Add grid offsets to cx/cy
    pred[..., 0] += grid_x / gw     # absolute cx
    pred[..., 1] += grid_y / gh     # absolute cy

    # Flatten predictions for easier matching
    pred = pred.view(B, -1, 6)      # [B, gh*gw, 6]

    # For simplicity: assume one ground truth bbox per image (common for satellite)
    # In real YOLO you'd match multiple GTs per grid cell
    gt_cx, gt_cy, gt_w, gt_h = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # Compute IoU (simplified CIoU or GIoU can be used later)
    # Here: simple MSE on box + BCE on obj_conf + BCE on class
    box_loss = nn.MSELoss()(pred[..., :4], target.unsqueeze(1).expand(-1, pred.size(1), -1))
    obj_loss = nn.BCELoss()(pred[..., 4], torch.ones_like(pred[..., 4]))  # assume object exists
    cls_loss = nn.BCELoss()(pred[..., 5], torch.zeros_like(pred[..., 5]))  # class=0

    total_loss = lambda_box * box_loss + lambda_obj * obj_loss + lambda_cls * cls_loss

    return total_loss, box_loss, obj_loss, cls_loss


##################### Train one epoch #####################
def train_one_epoch(model, epoch, writer, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_box = 0.0
    total_obj = 0.0
    total_cls = 0.0
    num_batches = len(loader)
    running_loss = 0.0

    for batch_idx, (rgb, event, bbox) in enumerate(tqdm(loader, desc="Training")):
        event, bbox = event.to(device), bbox.to(device)
        
        optimizer.zero_grad()
        pred = model(event)
        loss, box_loss, obj_loss, cls_loss = criterion(pred, bbox)
        loss.backward()
        optimizer.step()
        
        # TensorBoard batch logging
        global_step = epoch * len(loader) + batch_idx
        writer.add_scalar("Train/loss_batch", loss.item(), global_step)
        writer.add_scalar("Train/loss_batch/box", box_loss.item(), global_step)
        writer.add_scalar("Train/loss_batch/obj", obj_loss.item(), global_step)
        writer.add_scalar("Train/loss_batch/cls", cls_loss.item(), global_step)

        total_loss += loss.item()
        total_box += box_loss.item()
        total_obj += obj_loss.item()
        total_cls += cls_loss.item()
        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            avg_running = running_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx+1}/{num_batches} | Running Loss: {avg_running:.6f}")

    epoch_avg_loss = total_loss / num_batches
    return epoch_avg_loss, total_box / num_batches, total_obj / num_batches, total_cls / num_batches


##################### Validate #####################
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_box = 0.0
    total_obj = 0.0
    total_cls = 0.0
    with torch.no_grad():
        for rgb, event, bbox in tqdm(loader, desc="Validation"):
            event, bbox = event.to(device), bbox.to(device)
            pred = model(event)
            loss, box_loss, obj_loss, cls_loss = criterion(pred, bbox)
            total_loss += loss.item()
            total_box += box_loss.item()
            total_obj += obj_loss.item()
            total_cls += cls_loss.item()
    
    n = len(loader)
    return total_loss / n, total_box / n, total_obj / n, total_cls / n


##################### Main #####################
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model directory with sequential numbering
    counter = 0
    while True:
        model_subdir = f"{counter}"
        model_dir = os.path.join(args.save_dir, model_subdir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            break
        counter += 1

    # Save run details
    details_path = os.path.join(model_dir, "details.txt")
    with open(details_path, "w") as f:
        f.write(f"Bounding Box Training Run\n")
        f.write(f"-------------------------\n")
        f.write(f"Satellite:       {args.satellite}\n")
        f.write(f"Sequence:       {args.sequence}\n")
        f.write(f"Distance:       {args.distance}\n")
        f.write(f"Batch size:      {args.batch_size}\n")
        f.write(f"Max epochs:      {args.epochs}\n")
        f.write(f"Learning rate:   {args.lr}\n")
        f.write(f"Device:          x2 GPUs\n")
        f.write(f"Notes:           Single-class (satellite), event-only input\n")

    # Datasets & Loaders
    train_ds = SatelliteBBDataset(split='train', satellite=args.satellite, sequence=args.sequence, distance=args.distance)
    val_ds   = SatelliteBBDataset(split='val',   satellite=args.satellite, sequence=args.sequence, distance=args.distance)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model
    model = EventBBNet().to(device)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # TensorBoard
    writer = SummaryWriter(log_dir=model_dir)

    # Log model graph
    dummy_event = torch.randn(1, 1, 720, 800).to(device)
    writer.add_graph(model, dummy_event)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = yolo_loss

    # Early stopping
    patience = 10
    min_delta_pct = 0.005
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = args.epochs

    for epoch in range(1, max_epochs + 1):
        train_loss, train_box, train_obj, train_cls = train_one_epoch(
            model, epoch, writer, train_loader, optimizer, criterion, device
        )
        val_loss, val_box, val_obj, val_cls = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # TensorBoard epoch logging
        writer.add_scalar("Train/loss_epoch/total", train_loss, epoch)
        writer.add_scalar("Val/loss_epoch/total", val_loss, epoch)
        writer.add_scalar("Train/loss_epoch/box", train_box, epoch)
        writer.add_scalar("Val/loss_epoch/box", val_box, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss * (1 - min_delta_pct):
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print(f"→ Improved! Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"→ No improvement ({epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pth"))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Event-only Bounding Box Network")
    parser.add_argument("--batch_size",   type=int,   default=32,       help="Batch size")
    parser.add_argument("--epochs",       type=int,   default=100,      help="Number of epochs")
    parser.add_argument("--lr",           type=float, default=1e-3,     help="Learning rate")
    parser.add_argument("--satellite",    type=str,   default="cassini", help="Satellite name")
    parser.add_argument("--sequence",     type=str,   default="1",       help="Sequence number")
    parser.add_argument("--distance",    type=str,   default="close",    help="Distance")
    parser.add_argument("--save_dir",     type=str,   default="boundingbox/_2_train/runs", help="Save directory")
    
    args = parser.parse_args()
    main(args)