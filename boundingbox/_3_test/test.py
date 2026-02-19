# test.py
import subprocess
import torch
import time
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torchvision.ops import nms

from boundingbox._1_dataset.dataset import SatelliteBBDataset
from boundingbox._2_train import model
from boundingbox._2_train.model import EventBBNet

import argparse

##################### Postprocess #####################
def postprocess(pred, conf_thresh=0.2, iou_thresh=0.4):
    """
    Simple NMS + filtering for single-class YOLO output.
    pred: [B, gh, gw, 6] → [cx, cy, w, h] normalized [0,1] relative to each cell + obj_conf, class_prob
    Returns list of [x1, y1, x2, y2, conf] per image, normalized [0,1] relative to image
    """
    B, gh, gw, C = pred.shape
    device = pred.device

    # Sigmoid on conf and class
    # pred[..., 4:] = torch.sigmoid(pred[..., 4:])

    # grid_x, grid_y: [gh, gw] with values 0,1,...,gw-1 and 0,1,...,gh-1
    grid_y, grid_x = torch.meshgrid(
        torch.arange(gh, device=device, dtype=torch.float32),
        torch.arange(gw, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # [1, gh, gw]
    grid_x = grid_x[None, :, :]   
    grid_y = grid_y[None, :, :]   

    # Decode cx, cy, w, h to normalized [0,1] relative to image
    pred[..., 0] = (pred[..., 0] + grid_x) / gw 
    pred[..., 1] = (pred[..., 1] + grid_y) / gh  
    pred[..., 2] = pred[..., 2] / gw 
    pred[..., 3] = pred[..., 3] / gh 

    # Flatten to [B, num_cells, 6] where num_cells = gh*gw
    pred = pred.view(B, -1, 6)

    # Extract boxes and scores
    boxes = pred[..., :4]  # [B, num_cells, 4] cx, cy, w, h normalized [0,1] relative to image
    scores = pred[..., 4] * pred[..., 5]  # obj_conf * class_prob [B, num_cells]

    # Filter by confidence
    keep = scores > conf_thresh # [B, num_cells] boolean mask

    final_boxes_list = [] # list of N [x1,y1,x2,y2] normalized [0,1] relative to image
    final_scores_list = [] # list of N confidence scores for the final boxes

    for b in range(B):
        keep_b = keep[b]  # [num_cells]
        if not keep_b.any():
            continue

        boxes_b = boxes[b][keep_b]      # [num_cells, 4]
        scores_b = scores[b][keep_b]    # [num_cells]

        # Convert to [x1,y1,x2,y2] normalized [0,1] relative to image
        x1 = boxes_b[:, 0] - boxes_b[:, 2] / 2
        y1 = boxes_b[:, 1] - boxes_b[:, 3] / 2
        x2 = boxes_b[:, 0] + boxes_b[:, 2] / 2
        y2 = boxes_b[:, 1] + boxes_b[:, 3] / 2

        # [num_cells, 4] x1,y1,x2,y2 normalized [0,1] relative to image
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # Greedy Non Maximum Suppression NMS NMS (single-class)
        # keep_idx = []
        # while scores_b.numel() > 0:
        #     max_idx = scores_b.argmax()
        #     keep_idx.append(max_idx.item())

        #     if scores_b.numel() == 1:
        #         break

            # Compute IoU of the best box with the rest: [1, 4] vs [num_cells, 4]
        #     iou = compute_iou(boxes_xyxy[max_idx:max_idx+1], boxes_xyxy)  # [1, num_cells]

            # Keep boxes with IoU < thresh (exclude self by >= thresh or slice)
        #     keep = iou.squeeze(0) < iou_thresh
        #     keep[max_idx] = False

        #     boxes_xyxy = boxes_xyxy[keep]
        #     scores_b = scores_b[keep]

        keep_idx = nms(boxes_xyxy, scores_b, iou_threshold=iou_thresh)

        # Collect final boxes & scores for this image
        final_boxes_b = boxes_xyxy[torch.tensor(keep_idx, device=device)]
        final_scores_b = scores_b[torch.tensor(keep_idx, device=device)]

        final_boxes_list.append(final_boxes_b.cpu().numpy())
        final_scores_list.append(final_scores_b.cpu().numpy())


    # print(f"DEBUG: Final boxes list: {final_boxes_list}")
    # print(f"DEBUG: Final scores list: {final_scores_list}")

    # List of [x1,y1,x2,y2] normalized [0,1] relative to image + list of confidence scores
    return final_boxes_list, final_scores_list

##################### Compute IoU #####################
def compute_iou(box1, box2):
    """
    Compute IoU between box1 [N1, 4] and box2 [N2, 4] in [x1,y1,x2,y2] format (normalized [0,1]).
    Returns IoU matrix of shape [N1, N2].
    """
    # box1: [N1, 4]  → expand to [N1, N2, 4] by adding dim 1
    # box2: [N2, 4]  → expand to [N1, N2, 4] by adding dim 0
    box1 = box1[:, None, :]   # [N1, 1, 4]
    box2 = box2[None, :, :]   # [1, N2, 4]

    # Intersection coordinates
    x1 = torch.max(box1[..., 0], box2[..., 0])   # [N1, N2]
    y1 = torch.max(box1[..., 1], box2[..., 1])   # [N1, N2]
    x2 = torch.min(box1[..., 2], box2[..., 2])   # [N1, N2]
    y2 = torch.min(box1[..., 3], box2[..., 3])   # [N1, N2]

    # Intersection area
    inter_w = torch.clamp(x2 - x1, min=0.0)      # [N1, N2]
    inter_h = torch.clamp(y2 - y1, min=0.0)      # [N1, N2]
    inter_area = inter_w * inter_h               # [N1, N2]

    # Areas of each box
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])  # [N1, 1]
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])  # [1, N2]

    # Union area [N1, N2] (broadcasting)
    union_area = area1 + area2 - inter_area + 1e-6 

    # IoU [N1, N2]
    iou = inter_area / union_area
    return iou

##################### Main #####################
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (from parallel trained to single GPU inference)
    model = EventBBNet().to(device)
    model_path = os.path.join(args.model_path, args.model_name, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    # model.load_state_dict(checkpoint)
    model.eval()

    # Test dataset
    test_ds = SatelliteBBDataset(split='test', satellite=args.satellite,
                                 sequence=args.sequence, distance=args.distance)
    # test_ds.labels["annotations"] = test_ds.labels["annotations"][:10]  # only first 10 samples
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    total_iou = 0.0
    num_samples = 0
    latencies = []
    power_readings = []

    pro_obj_max = 0.0
    prob_class_max = 0.0
    n_samples = 0

    with torch.no_grad():
        for rgb, event, bbox in tqdm(test_loader, desc="Testing"):
            event = event.to(device)

            # Start timing (seconds)
            start_time = time.time()

            # Power measurement start (Watts) - NVIDIA GPU only
            power_start = float(subprocess.check_output(
                "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", shell=True
            ).decode().strip().split('\n')[0])  # GPU 0 power

            # Inference → pred shape [1, gh, gw, 6]
            pred = model(event) 

            # Postprocess → list of [x1, y1, x2, y2, conf]
            final_boxes, final_scores = postprocess(pred)

            # End timing (inference + postprocess)
            latency = time.time() - start_time
            latencies.append(latency)

            # Sample power during inference (average of start and end)
            power_end = float(subprocess.check_output(
                "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", shell=True
            ).decode().strip().split('\n')[0])
            avg_power = (power_start + power_end) / 2
            power_readings.append(avg_power)

            # print(f"DEBUG: Obj_conf max: {pred[..., 4].max().item():.4f}")
            # print(f"DEBUG: Class_prob max: {pred[..., 5].max().item():.4f}")
            pro_obj_max += pred[..., 4].max().item()
            prob_class_max += pred[..., 5].max().item()

            # GT [cx, cy, w, h] normalized [0,1] relative to image (assuming single GT bbox per image)
            gt_bbox = bbox.numpy()[0]  

            # GT [x1, y1, x2, y2] normalized [0,1] relative to image
            gt_xyxy = np.array([
                gt_bbox[0] - gt_bbox[2]/2,
                gt_bbox[1] - gt_bbox[3]/2,
                gt_bbox[0] + gt_bbox[2]/2,
                gt_bbox[1] + gt_bbox[3]/2
            ])

            # Compute IoU
            if len(final_boxes) > 0:
            
                # final_boxes[0] is the array for the only image [N, 4]
                pred_boxes_np = final_boxes[0]  # [N, 4]
                pred_scores_np = final_scores[0]  # [N]
                
                if len(pred_boxes_np) == 0:
                    iou = 0.0
                else:
                    # Pick highest score box
                    best_idx = np.argmax(pred_scores_np)
                    pred_xyxy = pred_boxes_np[best_idx]  # [x1,y1,x2,y2]

                    # [1, 4] vs [1, 4] -> IoU scalar [1, 1]
                    iou = compute_iou(torch.tensor([pred_xyxy]), torch.tensor([gt_xyxy])).max().item()
            else:
                iou = 0.0

            total_iou += iou
            num_samples += 1

    avg_iou = total_iou / num_samples
    avg_latency = np.mean(latencies) * 1000  # ms
    fps = 1 / np.mean(latencies)
    avg_power_w = np.mean(power_readings)
    energy_per_sample_mj = avg_power_w * avg_latency  # mJ per sample

    avg_max_obj = pro_obj_max / num_samples
    avg_max_class = prob_class_max / num_samples

    # Save results
    save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{args.satellite}_{args.sequence}_{args.distance}.txt"), "w") as f:
        f.write(f"Evaluated Model: {args.model_path}/{args.model_name}/best_model.pth\n")
        f.write(f"Test Dataset: {args.satellite}_{args.sequence}_{args.distance} \n")
        f.write(f"Samples evaluated: {num_samples}\n")
        f.write(f"\n")
        f.write(f"Mean IoU: {avg_iou:.4f}\n")
        f.write(f"Avg Latency: {avg_latency:.2f} ms\n")
        f.write(f"Avg FPS: {fps:.2f}\n")
        f.write(f"Avg Power: {avg_power_w:.1f} W\n")
        f.write(f"Estimated energy per sample: {energy_per_sample_mj:.1f} mJ\n")
        f.write(f"\n")
        f.write(f"Avg max obj_conf: {avg_max_obj:.4f}\n")
        f.write(f"Avg max class_prob: {avg_max_class:.4f}\n")



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Test Event-based Bounding Box Model")
    parser.add_argument("--model_path", type=str, default="boundingbox/_2_train/runs/", help="Path to model folder")
    parser.add_argument("--model_name", type=str, default="13", help="Model name (subfolder in runs)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (keep 1 for accurate timing)")
    parser.add_argument("--satellite", type=str, default="cassini", help="Satellite name")
    parser.add_argument("--sequence", type=str, default="1", help="Sequence for real data")
    parser.add_argument("--distance", type=str, default="close", help="Distance for real data")
    parser.add_argument("--save_dir", type=str, default="boundingbox/_3_test/results", help="Save directory")

    args = parser.parse_args()
    main(args)