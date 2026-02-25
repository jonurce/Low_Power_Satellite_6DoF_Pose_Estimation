# train.py
import datetime
import sys
import signal
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

# Global variables to access last known values (update them in loop)
global_last_epoch = 0
global_best_val_loss = float('inf')
global_last_val_loss = float('inf')
global_last_train_box = 0.0
global_last_val_box = 0.0
global_last_model_state = None
global_model_dir = ""



##################### YOLO-style loss (simplified for single-class) #####################
def yolo_loss(pred, target, w_box, w_obj, w_cls, gamma_obj, gamma_cls, sigma):
    """
    YOLO-style loss for single-class, single-object-per-image detection.
    
    pred:   [B, gh, gw, K, 6]    [cx_offset, cy_offset, w, h] normalized [0,1] relative to each cell + obj_conf + class_prob

    target: [B, 4]            [cx, cy, w, h] normalized [0,1] relative to image

    """
    B, gh, gw, K, _ = pred.shape
    device = pred.device

    # ── 1. Decode predictions ──
    pred_cx = pred[..., :, 0] # normalized offset [0,1] within cell
    pred_cy = pred[..., :, 1] # normalized offset [0,1] within cell
    pred_w  = pred[..., :, 2] # normalized width [0,1] relative to cell
    pred_h  = pred[..., :, 3] # normalized height [0,1] relative to cell
    pred_obj = pred[..., :, 4] # objectness confidence [0,1]
    pred_cls = pred[..., :, 5]  # class prob (single class) [0,1]

    # Reshape to [B, gh*gw, K, 6] for easier matching
    pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)  # [B, gh, gw, K, 4]
    pred_boxes = pred_boxes.view(B, -1, K, 4)                                # [B, num_cells, K, 4]
    pred_obj   = pred_obj.view(B, -1, K)                                     # [B, num_cells, K]
    pred_cls   = pred_cls.view(B, -1, K)                                     # [B, num_cells, K]




    # ── 2. Prepare target_boxes ──

    # Target center cell indexes (which cell contains GT center)
    cell_x = (target[:, 0] * gw).floor().long().clamp(0, gw - 1)   # [0, gw-1] [B]
    cell_y = (target[:, 1] * gh).floor().long().clamp(0, gh - 1)   # [0, gh-1] [B]

    gt_cx = target[:, 0] * gw  - cell_x
    gt_cy = target[:, 1] * gh  - cell_y
    gt_w = target[:, 2] * gw    
    gt_h = target[:, 3] * gh   

    # Create target boxes (broadcast to all cells, but loss only on responsible)
    target_boxes = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)  # [B, 4]
    target_boxes = target_boxes.unsqueeze(1).expand(-1, gh*gw, -1)  # [B, num_cells, 4]
    target_boxes = target_boxes.unsqueeze(2).expand(-1, -1, K, -1)  # [B, num_cells, K, 4]



    # ── 3. Prepare target_obj and target_cls ──

    # Create target tensors [B, gh*gw, K]
    target_obj = torch.zeros(B, gh*gw, K, device=device)   # 1 only for responsible cell
    target_cls = torch.zeros(B, gh*gw, K, device=device)   # 1 for correct class

    # Grid cell centers in relative image coords
    grid_x_centers = torch.arange(gw, device=device, dtype=torch.float32) + 0.5  # [gw]
    grid_y_centers = torch.arange(gh, device=device, dtype=torch.float32) + 0.5  # [gh]

    # Expand to [B, gh, gw]
    grid_x_centers = grid_x_centers[None, None, :]  # [1, 1, gw]
    grid_y_centers = grid_y_centers[None, :, None]  # [1, gh, 1]

    # - 3.1 - Only 1 responsible cell (the one containing GT center)
    # Set responsible cells
    # batch_idx = torch.arange(B, device=device)
    # flat_idx = cell_y * gw + cell_x
    # target_obj[batch_idx, flat_idx] = 1.0
    # target_cls[batch_idx, flat_idx] = 1.0  # class = 1 (object present)

    # - 3.2 - Set all cells that overlap GT box as responsible (all to 1)

    # GT box corners in relative image coords
    # gt_x1 = (target[:, 0] - target[:, 2]/2) * gw   # left edge
    # gt_y1 = (target[:, 1] - target[:, 3]/2) * gh   # top edge
    # gt_x2 = (target[:, 0] + target[:, 2]/2) * gw   # right edge
    # gt_y2 = (target[:, 1] + target[:, 3]/2) * gh   # bottom edge 

    # Check which cells overlap GT box (center inside box)
    # inside_x = (grid_x_centers >= gt_x1[:, None, None]) & (grid_x_centers <= gt_x2[:, None, None])
    # inside_y = (grid_y_centers >= gt_y1[:, None, None]) & (grid_y_centers <= gt_y2[:, None, None])
    # inside = inside_x & inside_y  # [B, gh, gw] boolean

    # Flatten and set 1.0 for overlapping cells
    # flat_inside = inside.view(B, -1)  # [B, num_cells]
    # target_obj[flat_inside] = 1.0 # [B, num_cells]
    # target_cls[flat_inside] = 1.0 # [B, num_cells]

    # - 3.3 - Set all cells that overlap GT box as responsible (gradual with distance to GT center)

    # GT center in relative image coords
    gt_center_x = target[:, 0] * gw  # [B]
    gt_center_y = target[:, 1] * gh  # [B]

    # Half-size of GT box in grid units
    half_w = target[:, 2] * gw / 2  # [B]
    half_h = target[:, 3] * gh / 2  # [B]

    # Distance from GT center to each cell center (L2 distance in relative image coords)
    dx = grid_x_centers - gt_center_x[:, None, None]      # [B, gh, gw]
    dy = grid_y_centers - gt_center_y[:, None, None]      # [B, gh, gw]
    distance = torch.sqrt(dx**2 + dy**2)                  # [B, gh, gw]

    # Gaussian weight: 1.0 at center, falls off with distance
    # sigma = 3  # smaller sigma = sharper peak, larger = broader
    gaussian_weight = torch.exp(-distance**2 / (2 * sigma**2))  # [B, gh, gw]

    # Scale down toward box edges (linear falloff from center to edge)
    # Distance to box edge (approximate Manhattan distance to nearest edge)
    edge_dist_x = torch.abs(torch.abs(dx) - torch.abs(half_w[:, None, None]))
    edge_dist_y = torch.abs(torch.abs(dy) - torch.abs(half_h[:, None, None]))
    max_edge_dist_x = torch.abs(half_w[:, None, None]) # [B, gh, gw]
    max_edge_dist_y = torch.abs(half_h[:, None, None]) # [B, gh, gw]

    # [B, gh, gw], normalized to [0,1] where 0 at center, 1 at farthest edge
    edge_dist = torch.min(edge_dist_x/max_edge_dist_x+1e-6, edge_dist_y/max_edge_dist_y+1e-6) 

    # [B, gh, gw] Switch to 1 at center, 0 at farthest edge
    edge_falloff = torch.clamp(1.0 - edge_dist, min=0.0, max=1.0)

    # Final soft target: Gaussian peak × edge falloff
    soft_target = gaussian_weight * edge_falloff  # [B, gh, gw]
    soft_target = soft_target[..., None].expand(-1, -1, -1, K)  # [B, gh, gw, K]
    target_obj = soft_target.view(B, -1, K)  # [B, gh*gw, K]
    target_cls = soft_target.view(B, -1, K)  # [B, gh*gw, K]

    # Flatten and assign
    # flat_soft = soft_target.view(B, -1)  # [B, num_cells]
    # target_obj = flat_soft # [B, num_cells]
    # target_cls = flat_soft  # same for class (single-class) [B, num_cells]

    # Optional: force 1.0 exactly at center cell (strongest supervision)
    center_idx = cell_y * gw + cell_x
    batch_idx = torch.arange(B, device=device)
    target_obj[batch_idx, center_idx, :] = 1.0
    target_cls[batch_idx, center_idx, :] = 1.0


    # ── 4. Compute losses ──
    # Box loss — only on responsible cells (where target_obj == 1)
    box_mask = target_obj > 0.5 # [B, num_cells, K] 
    box_loss = 0.0

    # - 4.1 - MSE on box parameters (simple but less effective)
    # Box loss is measured in [0,1] normalized grid units
    # box_loss of 1: entire cell off (720x800 image → 22.5x25 pixels)
    # Aim for box_loss ~< 0.2 (4.5x5 pixels)
    # if box_mask.any():
    #     box_loss = nn.MSELoss(reduction='none')(pred_boxes, target_boxes)
    #     box_loss = box_loss.mean(dim=-1)                # [B, num_cells]
    #     box_loss = (box_loss * box_mask.float()).sum() / box_mask.sum().clamp(min=1)

    # - 4.2 - CIoU loss on boxes
    # CIoU is in [0,1] (higher is better, 1 is perfect overlap)
    # Box loss is 1 - CIoU, so 0 is perfect, 1 is no overlap very bad
    # Aim for box_loss ~< 0.2
    if box_mask.any():
        # CIoU loss (standard implementation)
        pred_xy = pred_boxes[..., :2]           # [B, num_cells, K, 2]
        pred_wh = pred_boxes[..., 2:4]          # [B, num_cells, K, 2]
        gt_xy = target_boxes[..., :2]           # [B, num_cells, K, 2]
        gt_wh = target_boxes[..., 2:4]          # [B, num_cells, K, 2]

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
        c2 = ((pred_xy - gt_xy)**2).sum(dim=-1)  # [B, num_cells, K]

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
        ciou_loss = (1 - ciou) * box_mask.float() # [B, num_cells, K]
        box_loss = ciou_loss.sum() / box_mask.sum().clamp(min=1)

    # - 4.3 - Focal loss on objectness confidence
    # far_mask = distance > 4.0  # [B, gh, gw]
    obj_loss = focal_loss(pred_obj, target_obj, gamma=gamma_obj)
    # obj_loss[far_mask] = 0.0   # ignore very distant negatives
    # obj_loss = obj_loss.mean() * remove mean from focal_loss and apply after cls_loss

    # - 4.4 - Focal loss on class probability (same target as objectness for single-class)
    cls_loss = focal_loss(pred_cls, target_cls, gamma=gamma_cls)



    # ── 5. Compute final loss: weight each component ──

    # Total loss (average per sample in the batch)
    total_loss = w_box * box_loss + w_obj * obj_loss + w_cls * cls_loss

    return total_loss, box_loss, obj_loss, cls_loss

##################### Focal Loss #####################
def focal_loss(inputs, targets, gamma):

    # bce: 0 for perfect prediction, inf for worst prediction
    bce = nn.BCELoss(reduction='none')(inputs, targets)

    # pt: 1 for perfect prediction, 0 for worst prediction
    pt = torch.exp(-bce)

    # Focal loss: modulate BCE with (1-pt)^gamma
    return ((1 - pt) ** gamma * bce).mean()

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
        loss, box_loss, obj_loss, cls_loss = criterion(pred, bbox, w_box=args.w_box, w_obj=args.w_obj, w_cls=args.w_cls, gamma_obj=args.gamma_obj, gamma_cls=args.gamma_cls, sigma=args.sigma)
        loss.backward()
        optimizer.step()
        
        # TensorBoard batch logging
        global_step = epoch * len(loader) + batch_idx
        writer.add_scalar("Train/loss_batch", loss.item(), global_step)
        writer.add_scalar("Train/loss_batch/box", box_loss.item(), global_step)
        writer.add_scalar("Train/loss_batch/obj", obj_loss.item(), global_step)
        writer.add_scalar("Train/loss_batch/cls", cls_loss.item(), global_step)

        # TensorBoard log: max objectness confidence and class probability
        max_obj = pred[..., 4].max().item()
        max_cls = pred[..., 5].max().item()
        writer.add_scalar("Train/batch/max_obj_conf", max_obj, global_step)
        writer.add_scalar("Train/batch/max_class_prob", max_cls, global_step)

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
def validate(model, epoch, writer, loader, criterion, device):
    
    model.eval()
    total_loss = 0.0
    total_box = 0.0
    total_obj = 0.0
    total_cls = 0.0
    with torch.no_grad():
        for rgb, event, bbox in tqdm(loader, desc="Validation"):
            event, bbox = event.to(device), bbox.to(device)
            pred = model(event)
            loss, box_loss, obj_loss, cls_loss = criterion(pred, bbox, w_box=args.w_box, w_obj=args.w_obj, w_cls=args.w_cls, gamma_obj=args.gamma_obj, gamma_cls=args.gamma_cls, sigma=args.sigma)
            total_loss += loss.item()
            total_box += box_loss.item()
            total_obj += obj_loss.item()
            total_cls += cls_loss.item()

            # TensorBoard log: max objectness confidence and class probability
            max_obj = pred[..., 4].max().item()
            max_cls = pred[..., 5].max().item()
            writer.add_scalar("Val/max_obj_conf", max_obj, epoch)
            writer.add_scalar("Val/max_class_prob", max_cls, epoch)
    
    n = len(loader)
    return total_loss / n, total_box / n, total_obj / n, total_cls / n

##################### Save on Interrupt #####################
def save_on_interrupt(signal_received, frame):
    print("\nInterrupt received — saving final results...")

    # Save model state_dict
    if global_last_model_state is not None:
        interrupt_path = os.path.join(global_model_dir, f"interrupted_epoch_{global_last_epoch}.pth")
        torch.save(global_last_model_state, interrupt_path)
        print(f"Saved interrupted model: {interrupt_path}")
    else:
        print("No model state to save (interrupted before first epoch)")
    
    with open(os.path.join(global_model_dir, "details.txt"), "a") as f:
        f.write(f"-------------------------\n")
        f.write(f"Results (interrupted):\n")
        f.write(f"Final epoch: {global_last_epoch}\n")
        f.write(f"Best val loss: {global_best_val_loss:.6f}\n")
        f.write(f"Last val loss: {global_last_val_loss:.6f}\n")
        f.write(f"Last train box loss: {global_last_train_box:.6f}\n")
        f.write(f"Last val box loss: {global_last_val_box:.6f}\n")
        f.write(f"Stopped at epoch {global_last_epoch}\n")
    
    sys.exit(0)
    
##################### Main #####################
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model directory with sequential numbering
    counter = args.start_count
    while True:
        model_subdir = f"{counter}"
        global_model_dir = os.path.join(args.save_dir, model_subdir)
        if not os.path.exists(global_model_dir):
            os.makedirs(global_model_dir, exist_ok=True)
            break
        counter += 1

    # Save run details
    details_path = os.path.join(global_model_dir, "details.txt")
    with open(details_path, "w") as f:
        f.write(f"Bounding Box Training Run\n")
        f.write(f"Single-class (satellite), event-only input\n")
        f.write(f"-------------------------\n")
        f.write(f"Satellite:       {args.satellite}\n")
        f.write(f"Sequence:       {args.sequence}\n")
        f.write(f"Distance:       {args.distance}\n")
        f.write(f"Batch size:      {args.batch_size}\n")
        f.write(f"Max epochs:      {args.epochs}\n")
        f.write(f"Learning rate:   {args.lr}\n")
        f.write(f"Device:          x{torch.cuda.device_count()} {torch.cuda.get_device_name(0)}\n")
        f.write(f"Loss weights:    w_box={args.w_box}, w_obj={args.w_obj}, w_cls={args.w_cls}\n")
        f.write(f"Focal loss gamma: gamma_obj={args.gamma_obj}, gamma_cls={args.gamma_cls}\n")
        f.write(f"Sigma for soft targets: {args.sigma}\n")
        f.write(f"Notes: Giving more importance to prob_obj and prob_cls (ensure they don't go to 0)\n")

        

    # Register handler
    signal.signal(signal.SIGINT, save_on_interrupt)

    
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
    writer = SummaryWriter(log_dir=global_model_dir)

    # Log model graph
    dummy_event = torch.randn(1, 1, 720, 800).to(device)
    writer.add_graph(model, dummy_event)

    # Multi-GPU (after graph, so it does not crash)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=6e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = yolo_loss

    # Early stopping
    patience = 100
    min_delta_pct = 0.00005
    best_val_loss = float('inf')
    epochs_no_improve = 0
    max_epochs = args.epochs

    for epoch in range(1, max_epochs + 1):
        train_loss, train_box, train_obj, train_cls = train_one_epoch(
            model, epoch, writer, train_loader, optimizer, criterion, device)
        val_loss, val_box, val_obj, val_cls = validate(
            model, epoch, writer, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # TensorBoard epoch logging
        writer.add_scalar("Train/loss_epoch/total", train_loss, epoch)
        writer.add_scalar("Val/loss_epoch/total", val_loss, epoch)
        writer.add_scalar("Train/loss_epoch/box", train_box, epoch)
        writer.add_scalar("Val/loss_epoch/box", val_box, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch)

        # Update global variables for interrupt handler
        global_last_epoch = epoch
        global_last_val_loss = val_loss
        global_last_train_box = train_box
        global_last_val_box = val_box
        global_last_model_state = model.state_dict()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss * (1 - min_delta_pct):
            best_val_loss = val_loss
            global_best_val_loss = best_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(global_model_dir, "best_model.pth"))
            print(f"→ Improved! Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"→ No improvement ({epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(global_model_dir, f"checkpoint_epoch_{epoch}.pth"))
            prev_path = os.path.join(global_model_dir, f"checkpoint_epoch_{epoch - 5}.pth")
            if os.path.exists(prev_path):
                os.remove(prev_path)

    with open(os.path.join(details_path), "a") as f:
        f.write(f"-------------------------\n")
        f.write(f"Results:\n")
        f.write(f"Final epoch: {epoch}\n")
        f.write(f"Best val loss: {best_val_loss:.6f}\n")
        f.write(f"Last val loss: {val_loss:.6f}\n")
        f.write(f"Last train box loss: {train_box:.6f}\n")
        f.write(f"Last val box loss: {val_box:.6f}\n")
        if epochs_no_improve >= patience:
            f.write(f"Early stopping triggered\n")
        elif epoch >= max_epochs:
            f.write(f"Reached max epochs\n")
        else:
            f.write(f"Stopped at epoch {epoch}\n")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Event-only Bounding Box Network")
    parser.add_argument("--start_count",   type=int,   default=20,       help="Starting count for model directory naming")
    parser.add_argument("--save_dir",     type=str,   default="boundingbox/_2_train/runs", help="Save directory")
    parser.add_argument("--batch_size",   type=int,   default=64,       help="Batch size")
    parser.add_argument("--epochs",       type=int,   default=500,      help="Number of epochs")
    parser.add_argument("--lr",           type=float, default=3e-4,     help="Learning rate")
    parser.add_argument("--satellite",    type=str,   default="cassini", help="Satellite name")
    parser.add_argument("--sequence",     type=str,   default="1",       help="Sequence number")
    parser.add_argument("--distance",    type=str,   default="close",    help="Distance")
    parser.add_argument("--w_box",        type=float, default=10.0,      help="Weight for box loss")
    parser.add_argument("--w_obj",        type=float, default=30.0,    help="Weight for objectness loss")
    parser.add_argument("--w_cls",        type=float, default=30.0,    help="Weight for class loss")
    parser.add_argument("--gamma_obj",    type=float, default=1.0,     help="Focal loss gamma for objectness")
    parser.add_argument("--gamma_cls",    type=float, default=1.0,     help="Focal loss gamma for class")
    parser.add_argument("--sigma",        type=float, default=1.0,     help="Sigma for Gaussian soft targets")
    
    args = parser.parse_args()
    main(args)