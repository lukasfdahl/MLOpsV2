import os
import copy
import torch
import torch.nn as nn

from model import CustomCNN
from config import config
from dataloader import get_dataloaders
from post_training import quantize_model, benchmark, test_prunes, prune_model
import train

models_path = os.path.join(config["path"]["run_base_dir"], "models")
CHECKPOINT = os.path.join(models_path, "best_model.pth")
NUM_CLASSES = 80


def main():
    # Quantization (quantize_dynamic) is CPU-only, but benchmarking and
    # pruning benefit from GPU. Use CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Post-training optimization starting (device={device})")

    # Load best model
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT} — run training first.")

    _, val_loader, _ = get_dataloaders(
        num_workers=config["dataloader"].get("num_workers", 4),
        prefetch_factor=config["dataloader"].get("prefetch_factor", 2),
    )
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    # Quantization benchmark
    print("\n--- Quantization ---")
    quantized_model = quantize_model(model)
    orig_acc,  orig_time  = benchmark(model,           val_loader, device)
    quant_acc, quant_time = benchmark(quantized_model, val_loader, device)
    print(f"Original  | acc={orig_acc:.4f}%  time={orig_time:.4f}s")
    print(f"Quantized | acc={quant_acc:.4f}%  time={quant_time:.4f}s")

    # Pruning sweep + graph
    print("\n--- Pruning sweep ---")
    test_prunes(model, val_loader, device)

    # Build final optimized model (prune 50% → quantize)
    print("\n--- Final optimized model (50% prune + int8 quant) ---")
    optimized_model = prune_model(copy.deepcopy(model), val_loader, device, 0.5)
    optimized_model = torch.ao.quantization.quantize_dynamic(
        optimized_model, {nn.Linear}, dtype=torch.qint8)

    opt_dir = os.path.join(models_path, "optimized")
    os.makedirs(opt_dir, exist_ok=True)
    torch.save(optimized_model.state_dict(),
               os.path.join(opt_dir, "optimized_weights.pth"))
    print(f"Optimized weights saved to {opt_dir}/optimized_weights.pth")

    # Fine-tune from pruned (non-quantized) weights
    print("\n--- Fine-tuning from pruned weights ---")
    pruned_path = os.path.join(models_path, "pruned_model.pth")
    torch.save(
        {"epoch": 0, "model_state": prune_model(
            copy.deepcopy(model), val_loader, device, 0.5).state_dict(),
         "val_loss": 999.0, "val_acc": 0.0},
        pruned_path,
    )
    # Copy to best_model.pth so train_model() starts from the pruned base
    import shutil
    shutil.copy2(pruned_path, CHECKPOINT)
    print(f"Pruned model saved as {pruned_path}, starting fine-tune...")
    train.train_model()

    print("\n=== Post-training optimization complete ===")


if __name__ == "__main__":
    main()