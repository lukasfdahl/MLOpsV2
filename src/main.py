import train as train
from utility.hardware import check_device
from post_training import quantize_model, benchmark, test_prunes, prune_model
import copy
import shutil
import yaml
import torch
import torch.nn as nn
import os
from model import CustomCNN
from config import config
from dataloader import get_dataloaders
from inference import start_inference

def _load_pt_config():
    path = os.environ.get("POST_TRAINING_CONFIG", "config/post_training.config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

def run_training():
    train.train_model()
    print("Testing complete.")


# test function to you all!
def main_inference():
    start_inference()

def optimize_model(device):
    pt_cfg = _load_pt_config()
    models_path = os.path.join(config["path"]["run_base_dir"], "models")
    CHECKPOINT = os.path.join(models_path, "best_model.pth")
    NUM_CLASSES = 80
    cpu = torch.device("cpu")

    _, val_loader, _ = get_dataloaders()
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Quantization
    print("-------------------------")
    if pt_cfg["quantization"]["enabled"]:
        quantized_model = quantize_model(model)
        original_acc, original_time = benchmark(model, val_loader, device)
        # quantize_dynamic is CPU-only — no CUDA kernel for quantized linears
        quantized_acc, quantized_time = benchmark(quantized_model, val_loader, cpu)
        print(f"Original  model | acc: {original_acc:.6f}, time: {original_time:.4f}s (device={device})")
        print(f"Quantized model | acc: {quantized_acc:.6f}, time: {quantized_time:.4f}s (device=cpu)")

    # prune sweep
    # Pruning runs on CPU inside prune_model
    print("-------------------------")
    test_prunes(model, val_loader, device)
    print("-------------------------")

    # Final optimized model (prune → quantize)
    prune_amount = pt_cfg["pruning"]["prune_amount"]
    optimized_model = prune_model(model, val_loader, device, prune_amount)
    os.makedirs(os.path.join(models_path, "optimized"), exist_ok=True)
    torch.save(optimized_model.state_dict(),
               os.path.join(models_path, "optimized", "optimized_weights.pth"))

    # Fine-tune from pruned weights
    del model, optimized_model
    if pt_cfg["quantization"]["enabled"]:
        del quantized_model
    torch.cuda.empty_cache()

    pruned_path = os.path.join(models_path, "pruned_model.pth")
    # Re-build pruned model for saving (prune_model already returned it above)
    pruned_for_save = prune_model(
        CustomCNN(num_classes=NUM_CLASSES).to(device), val_loader, device, prune_amount)
    # Reload fresh checkpoint weights into it
    fresh = CustomCNN(num_classes=NUM_CLASSES)
    fresh.load_state_dict(torch.load(CHECKPOINT, map_location="cpu")["model_state"])
    pruned_for_save = prune_model(fresh, val_loader, cpu, prune_amount)

    torch.save(
        {"epoch": 0, "model_state": pruned_for_save.state_dict(),
         "val_loss": 999.0, "val_acc": 0.0},
        pruned_path,
    )
    best_path = os.path.join(models_path, "best_model.pth")
    shutil.copy2(pruned_path, best_path)
    print(f"Pruned model saved to {pruned_path}, fine-tuning from it...")

    # Fine-tune with reduced epochs and lower LR from post_training config
    train.train_model(
        override_epochs=pt_cfg["post_training"]["finetune_epochs"],
        override_lr=pt_cfg["post_training"]["finetune_learning_rate"],
        override_warmup=pt_cfg["post_training"]["finetune_warmup_epochs"],
    )

if __name__ == "__main__":
    device = check_device()
    run_training()
    optimize_model(device)
    main_inference()  # Not implimented