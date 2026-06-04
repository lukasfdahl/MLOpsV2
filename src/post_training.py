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

pruning_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
models_path = os.path.join(config["path"]["run_base_dir"], "models", "optimized")
os.makedirs(models_path, exist_ok=True)

def quantize_model(model : CustomCNN) -> CustomCNN:
    quantized_model : CustomCNN = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8) #if someone needs to just qunatazie and nothing else, this line is all that is needed

    torch.save(model.state_dict(), os.path.join(models_path, "original_weights.pth"))
    torch.save(quantized_model.state_dict(), os.path.join(models_path, "quantized_weights.pth"))
    orig_size_mb = os.path.getsize(os.path.join(models_path, "original_weights.pth")) / (1024 * 1024)
    quant_size_mb = os.path.getsize(os.path.join(models_path, "quantized_weights.pth")) / (1024 * 1024)

    os.remove(os.path.join(models_path, "original_weights.pth")) # best model should alreayd be saved elsewhere, no point in saving it twice (outside of just measuing size)

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


def prune_model(model : CustomCNN, val_loader, device, amount = 0.2):
    test_model = copy.deepcopy(model)

    # Scan through all layers
    for name, module in test_model.named_modules():
        if isinstance(module, nn.Linear): # if layer is linear, then prune it
            # Check the weights of the current module
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, 'weight')

    acc, inference_time = benchmark(test_model, val_loader, device)
    print(f"Pruned {amount*100}% | Acc: {acc:.6f} | Time: {inference_time:.4f}s")


def test_prunes(model : CustomCNN, val_loader, device):
    for amount in pruning_percentages:
        prune_model(model, val_loader, device, amount)
