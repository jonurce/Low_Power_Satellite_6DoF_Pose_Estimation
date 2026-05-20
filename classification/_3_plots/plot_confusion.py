
from csv import writer
from csv import writer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import argparse

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from classification._1_dataset.dataset import CIFAR10Dataset
from classification._2_train.model import YOLOv1ClassifierMMFv7, YOLOv1Classifier

##################### Validate #####################
def validate(model, loader, criterion, device):
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

    avg_loss = total_loss / n
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

##################### Get Predictions for Confusion Matrix #####################
def get_predictions(model, loader, device):
    """Return all predictions and true labels"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting predictions"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)

##################### Main #####################
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model weights
    best_model = torch.load(args.model_path, map_location=device)
    print(f"Loaded model checkpoint from {args.model_path}")

    # 2. Define model structure
    if ("MMF" in args.model_path):
        print("Model is MMF, using YOLOv1ClassifierMMFv7")
        model = YOLOv1ClassifierMMFv7(num_classes=10, weight_init_scale=0.2, quantization_levels=5).to(device)

    else:
        print("Model is original, using YOLOv1Classifier")
        model = YOLOv1Classifier(num_classes=10).to(device)

    # model.load_state_dict(best_model['model_state_dict'])
    state_dict = best_model['model_state_dict']

    # Remove "_orig_mod.module." prefix from all keys
    cleaned_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("_orig_mod.module."):
            new_k = k.replace("_orig_mod.module.", "")
        elif k.startswith("module."):
            new_k = k.replace("module.", "")
        cleaned_state[new_k] = v

    # 3. Load model weights into the model structure
    model.load_state_dict(cleaned_state)

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Datasets
    train_ds = CIFAR10Dataset(split='train')
    val_ds   = CIFAR10Dataset(split='test')

    # Data Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Simple classification loss: expects raw logits, internally applies softmax
    criterion = nn.CrossEntropyLoss()

    # Validate best model
    train_loss, train_acc = validate(model, train_loader, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Print final results
    print(f"Test Results for Model")
    print(f"Model: {args.model_path}\n")
    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    
    # === Confusion Matrix ===
    print("\nGenerating Confusion Matrix...")
    true_labels, predictions = get_predictions(model, val_loader, device)
    
    cm = confusion_matrix(true_labels, predictions)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1000)
    plt.title(f'Confusion Matrix - {model_name} Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save confusion matrix
    plt.savefig(args.cm_save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix saved")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv1-style Classifier on CIFAR-10")

    # Model directory: model_path
    model_name = "Baseline" # "Baseline" or "Best MMF"
    model_path = f"classification/_2_train/runs/best/{model_name}/best_model.pth"
    cm_save_path = f"classification/_3_plots/runs/best/{model_name.replace(' ', '_')}_confusion_matrix.png"
    parser.add_argument("--model_path", type=str, default=model_path, help="Path to checkpoint to resume from")
    parser.add_argument("--cm_save_path", type=str, default=cm_save_path, help="Path to save confusion matrix plot")


    # Parameters
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    main(args)