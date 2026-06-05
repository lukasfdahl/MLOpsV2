import os
import time
import copy
import torch
import torch.nn as nn

from config import config
from model import CustomCNN
from dataloader import get_dataloaders
from utility.training import detection_loss_set

# Config 
CHECKPOINT  = os.path.join(config["path"]["run_base_dir"], "models", "best_model.pth")
NUM_CLASSES = 80
BATCH_SIZE  = 16
CONF_THRESH = 0.5


# Helpers
def load_model(device):
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"No checkpoint at '{CHECKPOINT}' — train the model first.")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    return model


def run_inference(model, val_loader, device, label):
    """Run one full pass over val_loader, print per-batch and summary stats."""
    print(f"\n{'='*55}")
    print(f"  {label}  (device={device})")
    print(f"{'='*55}")

    total_correct = total_samples = 0
    batch_times = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)

            t0 = time.perf_counter()
            pred_logits, pred_boxes = model(images)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            _, metrics = detection_loss_set(
                pred_logits, pred_boxes, targets,
                num_classes=NUM_CLASSES
            )

            bs = images.size(0)
            total_correct  += metrics["acc"] * bs
            total_samples  += bs
            batch_times.append(elapsed_ms)

            throughput = bs / (elapsed_ms / 1000)
            print(f"  Batch {batch_idx+1:>3} | {bs:>3} imgs | "
                  f"{elapsed_ms:>7.1f} ms | {throughput:>6.1f} img/s | "
                  f"acc={metrics['acc']*100:.1f}%")

    avg_ms      = sum(batch_times) / len(batch_times)
    avg_img_ms  = avg_ms / BATCH_SIZE
    total_acc   = 100.0 * total_correct / total_samples

    print(f"\n  Summary [{label}]")
    print(f"    Accuracy          : {total_acc:.2f}%")
    print(f"    Avg batch latency : {avg_ms:.1f} ms")
    print(f"    Avg per-image     : {avg_img_ms:.2f} ms")
    print(f"    Throughput        : {1000/avg_img_ms:.1f} img/s")

    return total_acc, avg_ms, avg_img_ms


def start_inference():
    dataset_path = (
        config["path"]["sample_dataset_path"]
        if config["settings"]["use_sample_dataset"]
        else config["path"]["full_dataset_path"]
    )

    _, val_loader, _ = get_dataloaders(
        data_dir=dataset_path,
        batch_size=BATCH_SIZE,
    )

    # Original model (fp32, CUDA if available)
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fp32 = load_model(cuda)
    acc_fp32, batch_ms_fp32, img_ms_fp32 = run_inference(
        model_fp32, val_loader, cuda, "Original fp32")

    # Quantized model (int8, CPU only)
    # deepcopy so fp32 model stays intact for the comparison above
    cpu = torch.device("cpu")
    model_int8 = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model_fp32).cpu(), {nn.Linear}, dtype=torch.qint8
    )
    model_int8.eval()
    acc_int8, batch_ms_int8, img_ms_int8 = run_inference(
        model_int8, val_loader, cpu, "Quantized int8 (CPU)")

    # Side-by-side comparison
    print(f"\n{'='*55}")
    print("  Comparison")
    print(f"{'='*55}")
    print(f"  {'':25s} {'fp32':>10} {'int8':>10}  {'diff':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Accuracy (%)':25s} {acc_fp32:>10.2f} {acc_int8:>10.2f}"
          f"  {acc_int8-acc_fp32:>+10.2f}")
    print(f"  {'Avg batch latency (ms)':25s} {batch_ms_fp32:>10.1f} {batch_ms_int8:>10.1f}"
          f"  {batch_ms_int8-batch_ms_fp32:>+10.1f}")
    print(f"  {'Avg per-image (ms)':25s} {img_ms_fp32:>10.2f} {img_ms_int8:>10.2f}"
          f"  {img_ms_int8-img_ms_fp32:>+10.2f}")
    print(f"\n  Note: fp32 runs on {cuda}, int8 runs on CPU.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_inference()