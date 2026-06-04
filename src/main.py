import train as train
from utility.hardware import check_device
from post_training import quantize_model, benchmark

import torch
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
    print(f"Original model acc: {original_acc}, time: {original_time}")
    print(f"Quantized model acc: {quantized_acc}, time: {quantized_time}")
    print("-------------------------")

if __name__ == "__main__":
    device = check_device()
    run_training()
    optimize_model(device)
    run_inference()  # Not implimented
