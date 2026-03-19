import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import subprocess
from typing import Union
import mlflow


# from earlier
def check_device():
    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def log_gpu_metrics(device: Union[torch.device, str], step: int):
    device_type = device.type if isinstance(device, torch.device) else str(device)
    if device_type == "cuda":
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            util_s, used_s, total_s = [x.strip() for x in out.split(",")]
            mlflow.log_metric("gpu_util_percent", float(util_s), step=step)
            mlflow.log_metric("gpu_mem_used_mb", float(used_s), step=step)
            mlflow.log_metric("gpu_mem_total_mb", float(total_s), step=step)
        except Exception:
            pass

    elif device_type == "mps":
        try:
            alloc = torch.mps.current_allocated_memory()
            mlflow.log_metric("mps_allocated_mb", float(alloc) / (1024**2), step=step)
        except Exception:
            pass


# dataset classes
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# function to compute IoU
def compute_iou(box1, box2):
    box1 = box1.unsqueeze(1)
    box2 = box2.unsqueeze(0)

    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    intersection = inter_w * inter_h  # [N, M]

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])  # [N, 1]
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])  # [1, M]

    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-6)
    return iou


# mps doesnt like torch, so made own nms for testing code
def nms(boxes, scores, iou_threshold=0.5):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    order = scores.argsort(descending=True)

    keep = []

    # loop until no boxes left
    while order.numel() > 0:
        best = order[0].item()
        keep.append(best)

        if order.numel() == 1:
            break

        best_box = boxes[best].unsqueeze(0)
        rest_boxes = boxes[order[1:]]
        iou = compute_iou(best_box, rest_boxes)
        iou = iou.squeeze(0)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long)


# convert yolo bbox to pixel
def yolo_to_xyxy(bbox, img_w, img_h):
    single = bbox.dim() == 1
    if single:
        bbox = bbox.unsqueeze(0)

    cx, cy, bw, bh = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h

    # save result and return
    result = torch.stack([x1, y1, x2, y2], dim=1)
    return result.squeeze(0) if single else result


# function to denormalise image tensor for inspection
def denormalise_image(tensor):
    img = tensor.cpu().clone()
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0)
    return img.numpy().clip(0, 1)


# inspect predictions
def show_predictions(
    model, val_loader, device, num_examples=8, save_path="custom_model/predictions.png"
):
    model.eval()
    collected = {
        "images": [],
        "gt_cls": [],
        "gt_bbox": [],
        "pred_cls": [],
        "pred_bbox": [],
    }

    num_classes = model.num_classes

    # disable gradient calculation
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            pred_logits, pred_boxes = model(images)  # [B,K,C+1], [B,K,4]

            for i in range(images.size(0)):
                t = targets[i]
                gt_label = t["labels"][0].item() if len(t["labels"]) > 0 else 0
                gt_box = t["boxes"][0] if len(t["boxes"]) > 0 else torch.zeros(4)

                collected["images"].append(images[i].cpu())
                collected["gt_cls"].append(gt_label)
                collected["gt_bbox"].append(gt_box.cpu())

                logits_i = pred_logits[i]
                real_scores = logits_i[:, :num_classes]
                best_query = real_scores.max(dim=1).values.argmax().item()
                collected["pred_cls"].append(real_scores[best_query].argmax().item())
                collected["pred_bbox"].append(pred_boxes[i, best_query].cpu())

                if len(collected["images"]) >= num_examples:
                    break
            if len(collected["images"]) >= num_examples:
                break

    # plot predictions
    n = len(collected["images"])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    img_size = collected["images"][0].shape[-1]  # assumes square

    # Plot each image
    for i in range(n):
        ax = axes[i]
        img_np = denormalise_image(collected["images"][i])
        ax.imshow(img_np)

        # Ground-truth box (pink)
        gt_box = yolo_to_xyxy(collected["gt_bbox"][i], img_size, img_size)
        x1, y1, x2, y2 = gt_box.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="pink",
            facecolor="none",
            label="GT",
        )
        ax.add_patch(rect)

        # Predicted box (blue)
        pred_box = yolo_to_xyxy(collected["pred_bbox"][i], img_size, img_size)
        px1, py1, px2, py2 = pred_box.tolist()
        rect_pred = patches.Rectangle(
            (px1, py1),
            px2 - px1,
            py2 - py1,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
            linestyle="--",
            label="Pred",
        )
        ax.add_patch(rect_pred)

        gt_name = (
            COCO_CLASSES[collected["gt_cls"][i]]
            if collected["gt_cls"][i] < len(COCO_CLASSES)
            else str(collected["gt_cls"][i])
        )
        pred_name = (
            COCO_CLASSES[collected["pred_cls"][i]]
            if collected["pred_cls"][i] < len(COCO_CLASSES)
            else str(collected["pred_cls"][i])
        )
        ax.set_title(f"GT: {gt_name}\nPred: {pred_name}", fontsize=9)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="pink", linewidth=2, label="Ground truth"),
        Line2D([0], [0], color="blue", linewidth=2, linestyle="--", label="Predicted"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction examples saved to {save_path}")


# trainnig and val curves — returns fig so caller can log with mlflow.log_figure
def plot_training_curves(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train acc", linewidth=2)
    ax.plot(epochs, history["val_acc"], label="Val acc", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification accuracy over epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


@torch.no_grad()
def greedy_match(pred_boxes, tgt_boxes):
    """
    pred_boxes: [K,4]
    tgt_boxes:  [N,4]
    Returns:
      matched_pred_idx: [M]
      matched_tgt_idx:  [M]
    Greedy 1-1 matching by L1 distance.
    """
    K = pred_boxes.size(0)
    N = tgt_boxes.size(0)
    if N == 0:
        return (
            torch.empty(0, dtype=torch.long, device=pred_boxes.device),
            torch.empty(0, dtype=torch.long, device=pred_boxes.device),
        )

    # cost [K,N]
    cost = torch.cdist(pred_boxes, tgt_boxes, p=1)

    matched_p = []
    matched_t = []
    used_p = torch.zeros(K, dtype=torch.bool, device=pred_boxes.device)

    for t in range(N):
        # pick closest unused prediction for this target
        c = cost[:, t].clone()
        c[used_p] = 1e9
        p = int(torch.argmin(c).item())
        if c[p].item() >= 1e8:
            break
        used_p[p] = True
        matched_p.append(p)
        matched_t.append(t)

    return (
        torch.tensor(matched_p, dtype=torch.long, device=pred_boxes.device),
        torch.tensor(matched_t, dtype=torch.long, device=pred_boxes.device),
    )


def detection_loss_set(
    pred_logits, pred_boxes, targets, num_classes, noobj_weight=0.2, l1_weight=5.0
):
    """
    pred_logits: [B,K,C+1] (last class = no-object)
    pred_boxes:  [B,K,4]
    targets: list of dicts with "labels":[N], "boxes":[N,4]
    """
    device = pred_logits.device
    B, K, Cp1 = pred_logits.shape
    noobj = num_classes  # index of no-object class

    # default: all slots are no-object
    tgt_cls = torch.full((B, K), noobj, dtype=torch.long, device=device)

    l1 = torch.tensor(0.0, device=device)
    matched = 0
    correct = 0

    for b in range(B):
        t_labels = targets[b]["labels"].to(device)
        t_boxes = targets[b]["boxes"].to(device)

        mp, mt = greedy_match(pred_boxes[b], t_boxes)
        if mp.numel() == 0:
            continue

        tgt_cls[b, mp] = t_labels[mt]
        l1 = l1 + F.l1_loss(pred_boxes[b, mp], t_boxes[mt], reduction="sum")
        matched += mp.numel()

        pred_cls = pred_logits[b, mp, :num_classes].argmax(dim=-1)
        correct += (pred_cls == t_labels[mt]).sum().item()

    # classification loss over all slots, downweight no-object
    weight = torch.ones(num_classes + 1, device=device)
    weight[noobj] = noobj_weight
    ce = F.cross_entropy(pred_logits.transpose(1, 2), tgt_cls, weight=weight)

    if matched > 0:
        l1 = l1 / matched

    total = ce + l1_weight * l1
    return total, {
        "loss_ce": ce.item(),
        "loss_l1": float(l1.detach().cpu()),
        "acc": correct / matched if matched > 0 else 0.0,
    }
