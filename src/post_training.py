# This script handles optimizing the model after training
from config import config
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from model import CustomCNN
from utility.training import detection_loss_set
import time
import copy
import matplotlib
matplotlib.use("Agg")   # headless — no display needed on AI-LAB / Jenkins
import matplotlib.pyplot as plt

import yaml as _yaml

def _load_pt_config():
    path = os.environ.get("POST_TRAINING_CONFIG", "config/post_training.config.yaml")
    with open(path) as f:
        return _yaml.safe_load(f)

_pt_cfg = _load_pt_config()
pruning_percentages = _pt_cfg["pruning"]["test_prune_amounts"]
models_path = os.path.join(config["path"]["run_base_dir"], "models", "optimized")
os.makedirs(models_path, exist_ok=True)

def quantize_model(model: CustomCNN) -> CustomCNN:
    # quantize_dynamic only supports CPU — move the whole model to CPU first
    # so Conv layers (backbone) and the new quantized Linear layers are on the same device.
    model_cpu = copy.deepcopy(model).cpu()
    quantized_model: CustomCNN = torch.ao.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )

    torch.save(model.state_dict(), os.path.join(models_path, "original_weights.pth"))
    torch.save(quantized_model.state_dict(), os.path.join(models_path, "quantized_weights.pth"))
    orig_size_mb = os.path.getsize(os.path.join(models_path, "original_weights.pth")) / (1024 * 1024)
    quant_size_mb = os.path.getsize(os.path.join(models_path, "quantized_weights.pth")) / (1024 * 1024)

    os.remove(os.path.join(models_path, "original_weights.pth"))

    print(f"Original Size: {orig_size_mb:.2f} MB")
    print(f"Quantized Size: {quant_size_mb:.2f} MB")
    return quantized_model


def benchmark(model : CustomCNN, val_loader, device): # model name is just the name used when printing to the colsole
    start_time = time.perf_counter()
    model.eval()
    val_acc, val_total = 0, 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            pred_logits, pred_boxes = model(images)

            _, metrics = detection_loss_set(pred_logits, pred_boxes, targets, num_classes=model.num_classes)

            bs = images.size(0)
            val_acc += metrics["acc"] * bs
            val_total += bs

    end_time = time.perf_counter()
    return 100.0 * val_acc / val_total, end_time - start_time


def prune_model(model: CustomCNN, val_loader, device, amount=0.2) -> CustomCNN:
    """
    runs on cpu
    """
    pruned = copy.deepcopy(model).cpu()

    for _name, module in pruned.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

    pruned = pruned.to(device)            # move back for benchmarking
    acc, inference_time = benchmark(pruned, val_loader, device)
    print(f"Pruned {amount * 100:.0f}% | Acc: {acc:.6f} | Time: {inference_time:.4f}s")
    return pruned


def _model_size_mb(model: CustomCNN) -> float:
    """Disk size of the model weights in MB (via a temp save)."""
    tmp = os.path.join(models_path, "_tmp_size_check.pth")
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    return size


def test_prunes(model: CustomCNN, val_loader, device):
    """Sweep pruning percentages, print results, and save a pruning curve plot.
    """
    baseline_acc, _ = benchmark(model, val_loader, device)
    baseline_size = _model_size_mb(model)
    print(f"Baseline | Acc: {baseline_acc:.6f} | Size: {baseline_size:.2f} MB")

    amounts, accs, sizes = [], [], []
    for amount in pruning_percentages:
        pruned = prune_model(model, val_loader, device, amount)
        acc, _ = benchmark(pruned, val_loader, device)
        size = _model_size_mb(pruned)
        amounts.append(amount * 100)
        accs.append(acc)
        sizes.append(size)

    # Plot to see the impact!
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("Pruning Impact: Accuracy & Model Size vs. Sparsity", fontsize=13)

    # Accuracy panel
    ax1.axhline(baseline_acc, color="steelblue", linestyle="--",
                linewidth=1.2, label=f"Baseline ({baseline_acc:.2f}%)")
    ax1.plot(amounts, accs, marker="o", color="tomato", linewidth=2, label="Pruned")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_ylim(0, max(baseline_acc, max(accs)) * 1.1 + 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Size panel
    ax2.axhline(baseline_size, color="steelblue", linestyle="--",
                linewidth=1.2, label=f"Baseline ({baseline_size:.1f} MB)")
    ax2.plot(amounts, sizes, marker="s", color="mediumseagreen", linewidth=2, label="Pruned")
    ax2.set_xlabel("Pruning Sparsity (%)")
    ax2.set_ylabel("Model Size (MB)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(models_path, "pruning_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Pruning curve saved to {out_path}")