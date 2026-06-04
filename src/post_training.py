# This script handles optimizing the model after training
from config import config
import os
import torch
import torch.nn as nn
from model import CustomCNN
from utility.training import detection_loss_set
import time

models_path = os.path.join(config["path"]["run_base_dir"], "models", "optimized")
os.makedirs(models_path, exist_ok=True)

def quantize_model(model : CustomCNN) -> CustomCNN:
    quantized_model : CustomCNN = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

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
