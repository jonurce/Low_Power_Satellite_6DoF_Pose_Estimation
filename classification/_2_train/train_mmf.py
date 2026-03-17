# train_mmf.py (adapted for CIFAR-10 classification with YOLOv1Classifier)

import sys
import datetime
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, LambdaLR, OneCycleLR
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
scaler = GradScaler()

from classification._1_dataset.dataset import CIFAR10Dataset
from model import YOLOv1ClassifierMMF  

GLOBAL_LAST_EPOCH = 0
GLOBAL_BEST_VAL_LOSS = float('inf')
GLOBAL_LAST_VAL_LOSS = float('inf')
GLOBAL_LAST_TRAIN_LOSS = 0.0
GLOBAL_LAST_MODEL_STATE = None
GLOBAL_LAST_OPTIMIZER_STATE = None
# GLOBAL_LAST_WARMUP_SCHEDULER_STATE = None
GLOBAL_LAST_SCHEDULER_STATE = None
GLOBAL_MODEL_DIR = ""
GLOBAL_WARMUP_EPOCHS = 10


##################### Train one epoch #####################
def train_one_epoch(model, epoch, writer, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training")):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            logits = model(images)  # [B, 10]
            loss = criterion(logits.float(), labels)

        # loss.backward()
        # optimizer.step()
        
        # Backward and step with gradient clipping (uses scaler for this) 
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # important before clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step per batch for OneCycleLR
        # scheduler.step()

        # Metrics
        pred = logits.argmax(dim=1)
        correct = (pred == labels).sum().item()
        total = labels.size(0)

        total_loss += loss.item()
        total_correct += correct
        total_samples += total

        # TensorBoard batch logging
        global_step = epoch * num_batches + batch_idx
        writer.add_scalar("Train/batch/loss", loss.item(), global_step)
        writer.add_scalar("Train/batch/accuracy", correct / total, global_step)

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

##################### Validate #####################
def validate(model, epoch, writer, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n = len(loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Validation")):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits.float(), labels)

            pred = logits.argmax(dim=1)
            correct = (pred == labels).sum().item()
            total = labels.size(0)

            total_loss += loss.item()
            total_correct += correct
            total_samples += total

            global_step = epoch * n + batch_idx
            writer.add_scalar("Val/batch/loss", loss.item(), global_step)
            writer.add_scalar("Val/batch/accuracy", correct / total, global_step)

    avg_loss = total_loss / n
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

##################### Save on Interrupt #####################
def save_on_interrupt():
    print("\nInterrupt received — saving final results...")

    # Save model state_dict
    if GLOBAL_LAST_MODEL_STATE is not None:
        interrupt_path = os.path.join(GLOBAL_MODEL_DIR, f"interrupted_epoch_{GLOBAL_LAST_EPOCH}.pth")
        torch.save({
            'epoch': GLOBAL_LAST_EPOCH,
            'model_state_dict': GLOBAL_LAST_MODEL_STATE,
            'optimizer_state_dict': GLOBAL_LAST_OPTIMIZER_STATE,
            # 'warmup_scheduler_state_dict': GLOBAL_LAST_WARMUP_SCHEDULER_STATE,
            'scheduler_state_dict': GLOBAL_LAST_SCHEDULER_STATE,
            'best_val_loss': GLOBAL_LAST_VAL_LOSS,
            'hyperparameters': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr,
                'wd': args.wd,
            }
        }, interrupt_path)
        print(f"Saved interrupted model: {interrupt_path}")
    else:
        print("No model state to save (interrupted before first epoch)")
    
    with open(os.path.join(GLOBAL_MODEL_DIR, "details.txt"), "a") as f:
        f.write(f"-------------------------\n")
        f.write(f"Results (interrupted):\n")
        f.write(f"Final epoch: {GLOBAL_LAST_EPOCH}\n")
        f.write(f"Best val loss: {GLOBAL_BEST_VAL_LOSS:.6f}\n")
        f.write(f"Last val loss: {GLOBAL_LAST_VAL_LOSS:.6f}\n")
        f.write(f"Last train loss: {GLOBAL_LAST_TRAIN_LOSS:.6f}\n")
        f.write(f"Stopped at epoch {GLOBAL_LAST_EPOCH}\n")
    
    sys.exit(0)

###################### Register handler #####################
signal.signal(signal.SIGINT, save_on_interrupt)

###################### Warmup Lambda #####################
def warmup_lambda(epoch):
    if epoch < GLOBAL_WARMUP_EPOCHS:
        return float(epoch) / float(max(1, GLOBAL_WARMUP_EPOCHS))
    return 1.0

##################### Main #####################
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = YOLOv1ClassifierMMF(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Simple classification loss: expects raw logits, internally applies softmax
    criterion = nn.CrossEntropyLoss()

    # Early stopping for training
    patience = 100
    min_delta_pct = 0.00005
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 1

    # Datasets
    train_ds = CIFAR10Dataset(split='train')
    val_ds   = CIFAR10Dataset(split='test')

    # Handle resume training
    global GLOBAL_MODEL_DIR, GLOBAL_BEST_VAL_LOSS, GLOBAL_WARMUP_EPOCHS
    GLOBAL_WARMUP_EPOCHS = args.warmup_epochs
    if args.resume is None:

        # Create model directory with sequential numbering
        counter = args.start_count
        while True:
            model_subdir = f"{counter}"
            GLOBAL_MODEL_DIR = os.path.join(args.save_dir, model_subdir)
            if not os.path.exists(GLOBAL_MODEL_DIR):
                os.makedirs(GLOBAL_MODEL_DIR, exist_ok=True)
                break
            counter += 1

        # Save run details
        details_path = os.path.join(GLOBAL_MODEL_DIR, "details.txt")
        with open(details_path, "w") as f:
            f.write(f"Classification Training Run\n")
            f.write(f"CIFAR10 32x32px RGB input\n")
            f.write(f"-------------------------\n")
            f.write(f"Model Info:\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"-------------------------\n")
            f.write(f"Batch size:      {args.batch_size}\n")
            f.write(f"Max epochs:      {args.epochs}\n")
            f.write(f"Learning rate:   {args.lr}\n")
            f.write(f"Weight decay:   {args.wd}\n")
            f.write(f"Device:          x{torch.cuda.device_count()} {torch.cuda.get_device_name(0)}\n")

        # TensorBoard
        writer = SummaryWriter(log_dir=GLOBAL_MODEL_DIR)

        # Log model graph
        dummy_event = torch.randn(1, 3, 32, 32).to(device)
        writer.add_graph(model, dummy_event)

        # Data Loaders
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Multi-GPU (after graph, so it does not crash)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
            model = torch.compile(model)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
            # warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=args.lr/20)
            # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - GLOBAL_WARMUP_EPOCHS, eta_min=1e-6)
            # scheduler = OneCycleLR(optimizer, max_lr=args.lr * 1.1, total_steps=len(train_loader) * args.epochs,
            #     pct_start=0.03, anneal_strategy='cos', div_factor=2, final_div_factor=1e5)

    else:

        # Define GLOBAL_MODEL_DIR from args.resume
        GLOBAL_MODEL_DIR = os.path.dirname(args.resume)  # .../runs/5/

        # Define details path
        details_path = os.path.join(GLOBAL_MODEL_DIR, "details.txt")

        # Tensorboard writer: user already existing log inside GLOBAL_MODEL_DIR
        writer = SummaryWriter(log_dir=GLOBAL_MODEL_DIR)

        # Load model
        checkpoint = torch.load(args.resume, map_location=device)

        # model.load_state_dict(checkpoint['model_state_dict'])
        state_dict = checkpoint['model_state_dict']

        # Remove "_orig_mod.module." prefix from all keys
        cleaned_state = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith("_orig_mod.module."):
                new_k = k.replace("_orig_mod.module.", "")
            elif k.startswith("module."):
                new_k = k.replace("module.", "")
            cleaned_state[new_k] = v

        model.load_state_dict(cleaned_state)

        # Load best val loss
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            GLOBAL_BEST_VAL_LOSS = best_val_loss

        # Load hyperparameters
        if 'hyperparameters' in checkpoint:
            loaded_hp = checkpoint['hyperparameters']
            print("Loaded hyperparameters from checkpoint:")
            for k, v in loaded_hp.items():
                print(f"  {k}: {v}")
                setattr(args, k, v)  # update args with loaded values
        else:
            print("No hyperparameters found in checkpoint — using current args")

        # Data Loaders
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Multi-GPU (after graph, so it does not crash)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
            model = torch.compile(model)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
            
            # GLOBAL_WARMUP_EPOCHS = args.warmup_epochs
            # warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, min_lr=args.lr/20)
            # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - GLOBAL_WARMUP_EPOCHS, eta_min=1e-6)
            # scheduler = OneCycleLR(optimizer, max_lr=args.lr * 1.1, total_steps=len(train_loader) * args.epochs,
            #     pct_start=0.03, anneal_strategy='cos', div_factor=2, final_div_factor=1e5)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.6f}")

    # Training loop
    for epoch in range(args.epochs):

        train_loss, train_acc = train_one_epoch(model, epoch, writer, train_loader, optimizer, scheduler, criterion, device)
       
        val_loss, val_acc = validate(model, epoch, writer, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        writer.add_scalar("Train/epoch/loss", train_loss, epoch)
        writer.add_scalar("Train/epoch/accuracy", train_acc, epoch)
        writer.add_scalar("Val/epoch/loss", val_loss, epoch)
        writer.add_scalar("Val/epoch/accuracy", val_acc, epoch)
        writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch)

        # Update global variables for interrupt handler
        global GLOBAL_LAST_EPOCH, GLOBAL_LAST_TRAIN_LOSS, GLOBAL_LAST_VAL_LOSS, GLOBAL_LAST_MODEL_STATE, GLOBAL_LAST_OPTIMIZER_STATE, GLOBAL_LAST_SCHEDULER_STATE
        GLOBAL_LAST_EPOCH = epoch
        GLOBAL_LAST_TRAIN_LOSS = train_loss
        GLOBAL_LAST_VAL_LOSS = val_loss

        GLOBAL_LAST_MODEL_STATE = model.state_dict()
        GLOBAL_LAST_OPTIMIZER_STATE = optimizer.state_dict()
        # GLOBAL_LAST_WARMUP_SCHEDULER_STATE = warmup_scheduler.state_dict()
        GLOBAL_LAST_SCHEDULER_STATE = scheduler.state_dict()

        # Scheduler step per epoch for ReduceLROnPlateau
        scheduler.step(val_loss)

        # Scheduler step per epoch with warmup for warmup + cosine scheduler
        # if epoch < GLOBAL_WARMUP_EPOCHS:
        #     warmup_scheduler.step()
        # else:
        #     scheduler.step()

        # Early stopping
        if val_loss < best_val_loss * (1 - min_delta_pct):
            best_val_loss = val_loss
            GLOBAL_BEST_VAL_LOSS = best_val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'hyperparameters': {
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'wd': args.wd,
                }
            }, os.path.join(GLOBAL_MODEL_DIR, "best_model.pth"))
            print(f"→ Improved! Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"→ No improvement ({epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': val_loss,
                'hyperparameters': {
                    'batch_size': args.batch_size,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'wd': args.wd,
                }
            }, os.path.join(GLOBAL_MODEL_DIR, f"checkpoint_epoch_{epoch}.pth"))
            prev_path = os.path.join(GLOBAL_MODEL_DIR, f"checkpoint_epoch_{epoch - 5}.pth")
            if os.path.exists(prev_path):
                os.remove(prev_path)

    with open(os.path.join(details_path), "a") as f:
        f.write(f"-------------------------\n")
        f.write(f"Results:\n")
        f.write(f"Final epoch: {epoch}\n")
        f.write(f"Best val loss: {best_val_loss:.6f}\n")
        f.write(f"Last val loss: {val_loss:.6f}\n")
        if epochs_no_improve >= patience:
            f.write(f"Early stopping triggered\n")
        elif epoch >= args.epochs:
            f.write(f"Reached max epochs\n")
        else:
            f.write(f"Stopped at epoch {epoch}\n")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv1-style Classifier on CIFAR-10")

    # Save directory
    parser.add_argument("--start_count",   type=int,   default=0,       help="Starting count for model directory naming")
    parser.add_argument("--save_dir",     type=str,   default="classification/_2_train/runs_mmf", help="Save directory")

    # Resume directory: resume_path or None
    resume_path = "classification/_2_train/runs_mmf/XX/interrupted_epoch_XX.pth"
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--wd", type=float, default=6e-5)
    args = parser.parse_args()
    main(args)