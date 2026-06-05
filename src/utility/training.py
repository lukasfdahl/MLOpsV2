import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import subprocess
from typing import Union
import mlflow


def log_gpu_metrics(device: Union[torch.device, str], step: int):
    device_type = device.type if isinstance(
        device, torch.device) else str(device)
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
            mlflow.log_metric("mps_allocated_mb", float(
                alloc) / (1024**2), step=step)
        except Exception:
            pass

# trainnig and val curves — returns fig so caller can log with mlflow.log_figure


def plot_training_curves(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"],
            label="Val loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train acc", linewidth=2)
    ax.plot(epochs, history["val_acc"],
            label="Val acc", linewidth=2, linestyle="--")
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

    # cost [K,N] — cdist doesn't support BFloat16, cast to float32
    cost = torch.cdist(pred_boxes.float(), tgt_boxes.float(), p=1)

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
