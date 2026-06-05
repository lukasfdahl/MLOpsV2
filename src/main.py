import train as train
from utility.hardware import check_device
from post_training import quantize_model, benchmark, test_prunes, prune_model
import copy
import torch
import torch.nn as nn
import os
from train import CustomCNN
from config import config
from dataloader import get_dataloaders

def run_training():
    train.train_model()
    print("Testing complete.")


# test function to you all!
def run_inference():
    print("Inference not yet implimented")

def optimize_model(device):
    # Load the current best model to use for optimization
    models_path = os.path.join(config["path"]["run_base_dir"], "models")
    CHECKPOINT = os.path.join(models_path, "best_model.pth")
    NUM_CLASSES = 80
    _, val_loader, _ = get_dataloaders()
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Optimize model
    print("-------------------------")
    quantized_model = quantize_model(model)
    original_acc, original_time = benchmark(model, val_loader, device)
    quantized_acc, quantized_time = benchmark(quantized_model, val_loader, device)
    print(f"Original model acc: {original_acc:.6f}, time: {original_time:.4f}")
    print(f"Quantized model acc: {quantized_acc:.6f}, time: {quantized_time:.4f}")
    print("-------------------------")
    test_prunes(model, val_loader, device) # ONLY PRUNE ORIGINAL MODEL (qunatatization changes model layer types) (quantatize after pruning)
    print("-------------------------")

    # Generate final model (no most optimal prune is auto selected, look for tradeoffs on accurcy that the tests show)
    optimized_model = copy.deepcopy(model)
    prune_model(optimized_model, val_loader, device, 0.5)
    optimized_model : CustomCNN = torch.ao.quantization.quantize_dynamic(optimized_model, {nn.Linear}, dtype=torch.qint8)
    torch.save(optimized_model.state_dict(), os.path.join(config["path"]["run_base_dir"], "models", "optimized", "optimized_weights.pth"))

    # Fine tune optimized model (just reruns training with the new pruned quantazied model as a base)
    train.train_model(optimized_model)

if __name__ == "__main__":
    device = check_device()
    run_training()
    optimize_model(device)
    run_inference()  # Not implimented
