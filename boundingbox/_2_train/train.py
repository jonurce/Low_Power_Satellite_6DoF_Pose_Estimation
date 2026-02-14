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
def yolo_loss(pred, target):
    """
    YOLO-style loss for single-class, single-object-per-image detection.
    
    pred:   [B, gh, gw, 6]    [cx_offset, cy_offset, w, h] normalized [0,1] relative to each cell + obj_conf, class_prob

    target: [B, 4]            [cx, cy, w, h] normalized [0,1] relative to image

    """
    B, gh, gw, _ = pred.shape
    device = pred.device

    # ── 1. Prepare grid coordinates ──
    grid_y, grid_x = torch.meshgrid(
        torch.arange(gh, device=device, dtype=torch.float32),
        torch.arange(gw, device=device, dtype=torch.float32),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).unsqueeze(-1)   # [1, gh, gw, 1]
    grid_y = grid_y.unsqueeze(0).unsqueeze(-1)   # [1, gh, gw, 1]

    # ── 2. Decode predictions ──
    pred_cx = pred[..., 0] # normalized offset [0,1] within cell
    pred_cy = pred[..., 1] # normalized offset [0,1] within cell
    pred_w  = pred[..., 2] # normalized width [0,1] relative to cell
    pred_h  = pred[..., 3] # normalized height [0,1] relative to cell
    pred_obj = pred[..., 4] # objectness confidence [0,1]
    pred_cls = pred[..., 5]  # class prob (single class) [0,1]

    # Reshape to [B, gh*gw, 6] for easier matching
    pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)  # [B, gh, gw, 4]
    pred_boxes = pred_boxes.view(B, -1, 4)                                # [B, num_cells, 4]
    pred_obj   = pred_obj.view(B, -1)                                     # [B, num_cells]
    pred_cls   = pred_cls.view(B, -1)                                     # [B, num_cells]

    # ── 3. Find responsible grid cell (cell containing GT center) ──
    cell_x = (target[:, 0] * gw).floor().long().clamp(0, gw - 1)   # grid column index [0, gw-1] [B]
    cell_y = (target[:, 1] * gh).floor().long().clamp(0, gh - 1)   # grid row index [0, gh-1] [B]

    # ── 4. Prepare target (from image to grid) ──
    gt_cx = target[:, 0] * gw  - cell_x
    gt_cy = target[:, 1] * gh  - cell_y
    gt_w = target[:, 2] * gw    
    gt_h = target[:, 3] * gh    

    # Create target boxes (broadcast to all cells, but loss only on responsible)
    target_boxes = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1).unsqueeze(1)  # [B, 1, 4]
    target_boxes = target_boxes.expand(-1, gh*gw, -1)                           # [B, num_cells, 4]

    # Create target tensors [B, gh*gw]
    target_obj = torch.zeros(B, gh*gw, device=device)   # 1 only for responsible cell
    target_cls = torch.zeros(B, gh*gw, device=device)   # 1 for correct class

    # Set responsible cells
    batch_idx = torch.arange(B, device=device)
    flat_idx = cell_y * gw + cell_x
    target_obj[batch_idx, flat_idx] = 1.0
    target_cls[batch_idx, flat_idx] = 1.0  # class = 1 (object present)

    # ── 5. Compute losses ──
    # Box loss — only on responsible cells (where target_obj == 1)
    box_mask = target_obj > 0.5
    box_loss = 0.0

    # Option 1: MSE on box parameters (simple but less effective)
    # Box loss is measured in [0,1] normalized grid units
    # box_loss of 1: entire cell off (720x800 image → 22.5x25 pixels)
    # Aim for box_loss ~< 0.2 (4.5x5 pixels)
    # if box_mask.any():
    #     box_loss = nn.MSELoss(reduction='none')(pred_boxes, target_boxes)
    #     box_loss = box_loss.mean(dim=-1)                # [B, num_cells]
    #     box_loss = (box_loss * box_mask.float()).sum() / box_mask.sum().clamp(min=1)

    # Option 2: CIoU loss (more complex but better convergence)
    # CIoU is in [0,1] (higher is better, 1 is perfect overlap)
    # Box loss is 1 - CIoU, so 0 is perfect, 1 is no overlap very bad
    # Aim for box_loss ~< 0.2
    if box_mask.any():
        # CIoU loss (standard implementation)
        pred_xy = pred_boxes[..., :2]           # [B, num_cells, 2]
        pred_wh = pred_boxes[..., 2:4]          # [B, num_cells, 2]
        gt_xy = target_boxes[..., :2]           # [B, num_cells, 2]
        gt_wh = target_boxes[..., 2:4]          # [B, num_cells, 2]

        # Intersection area
        inter_wh = torch.min(pred_xy + pred_wh/2, gt_xy + gt_wh/2) - \
                   torch.max(pred_xy - pred_wh/2, gt_xy - gt_wh/2)
        inter_wh = torch.clamp(inter_wh, min=0.0)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]

        # Union area
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        gt_area   = gt_wh[..., 0] * gt_wh[..., 1]
        union_area = pred_area + gt_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-6)

        # Center distance squared
        c2 = ((pred_xy - gt_xy)**2).sum(dim=-1)  # [B, num_cells]

        # Smallest enclosing box diagonal squared
        enclose_wh = torch.max(pred_xy + pred_wh/2, gt_xy + gt_wh/2) - \
                     torch.min(pred_xy - pred_wh/2, gt_xy - gt_wh/2)
        enclose_wh = torch.clamp(enclose_wh, min=0.0)
        enclose_diag = (enclose_wh[..., 0]**2 + enclose_wh[..., 1]**2) + 1e-6

        # Aspect ratio consistency (v term)
        v = (4 / (torch.pi**2)) * (torch.atan(pred_wh[..., 0]/pred_wh[..., 1].clamp(min=1e-6)) - \
                                   torch.atan(gt_wh[..., 0]/gt_wh[..., 1].clamp(min=1e-6)))**2

        # CIoU
        alpha = v / ((1 - iou + v) + 1e-6)
        ciou = iou - (c2 / enclose_diag) - alpha * v

        # Mean over responsible cells
        ciou_loss = (1 - ciou) * box_mask.float()
        box_loss = ciou_loss.sum() / box_mask.sum().clamp(min=1)

    # Objectness loss — BCE everywhere
    obj_loss = nn.BCELoss()(pred_obj, target_obj)

    # Class loss — BCE on responsible cells (or everywhere if you want)
    cls_loss = nn.BCELoss()(pred_cls, target_cls)

    # Total loss (average per sample in the batch)
    total_loss = 20.0 * box_loss + 1.0 * obj_loss + 0.5 * cls_loss

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
    counter = 3
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
        f.write(f"Notes: Notes: total_loss = 20 * box_loss (CIoU) + 1 * p_obj_loss (BCE) + 0.5 * p_class_loss (BCE)\n")
        f.write(f"Notes: patience 60 + max epochs 500 (keep it high for convergence)\n")
        f.write(f"longer training: min_delta_pct from 0.005 (0.5%) to 0.0005 (0.05%)\n")
        f.write(f"increase weight decay: from 1e-5 to 3e-5\n")
        f.write(f"Results: .....\n")

        

    # Datasets & Loaders
    train_ds = SatelliteBBDataset(split='train', satellite=args.satellite, sequence=args.sequence, distance=args.distance)
    val_ds   = SatelliteBBDataset(split='val',   satellite=args.satellite, sequence=args.sequence, distance=args.distance)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Model
    model = EventBBNet().to(device)

    # TensorBoard
    writer = SummaryWriter(log_dir=model_dir)

    # Log model graph
    dummy_event = torch.randn(1, 1, 720, 800).to(device)
    writer.add_graph(model, dummy_event)

    # Multi-GPU (after graph, so it does not crash)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = yolo_loss

    # Early stopping
    patience = 60
    min_delta_pct = 0.0005
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
    parser.add_argument("--batch_size",   type=int,   default=64,       help="Batch size")
    parser.add_argument("--epochs",       type=int,   default=500,      help="Number of epochs")
    parser.add_argument("--lr",           type=float, default=5e-4,     help="Learning rate")
    parser.add_argument("--satellite",    type=str,   default="cassini", help="Satellite name")
    parser.add_argument("--sequence",     type=str,   default="1",       help="Sequence number")
    parser.add_argument("--distance",    type=str,   default="close",    help="Distance")
    parser.add_argument("--save_dir",     type=str,   default="boundingbox/_2_train/runs", help="Save directory")
    
    args = parser.parse_args()
    main(args)