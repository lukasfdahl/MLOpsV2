#!/usr/bin/env python3
"""Batch-size benchmark for CustomCNN: latency + throughput per batch size.

Run (GPU container):
    docker run --rm --gpus all -v "$PWD:/app" --workdir /app \
        kaspersiebrands/mlops-kls-container:latest python src/benchmark.py
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CustomCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
IMG = 64
ITERS = 30


def main():
    model = CustomCNN(num_classes=80).to(DEVICE).eval()
    print(f"device = {DEVICE}\n")
    print(f"{'batch':>6} {'ms/batch':>10} {'ms/img':>8} {'img/s':>10}")

    results = []
    for b in BATCH_SIZES:
        x = torch.randn(b, 3, IMG, IMG, device=DEVICE)
        with torch.no_grad():
            for _ in range(5):                       # warmup
                model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                model(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        ms = dt / ITERS * 1000
        thr = b * ITERS / dt
        results.append((b, thr))
        print(f"{b:>6} {ms:>10.2f} {ms / b:>8.3f} {thr:>10.1f}")

    peak_b, peak_thr = max(results, key=lambda r: r[1])
    print(f"\nPeak throughput: {peak_thr:.0f} img/s at batch {peak_b}")
    print(f"Speedup vs batch 1: {peak_thr / results[0][1]:.1f}x")
    print("Throughput climbs as batches fill the GPU, then flattens out (saturation);")
    print("per-batch latency keeps rising past that point. If it flattens while GPU")
    print("utilisation is high it is compute-bound, if util stays low it is memory/latency-bound.")


if __name__ == "__main__":
    main()
