# (Recommended, but not mandatory: Consider writing your inference script in another language than Python e.g. C++)
# inference.py
import os
import sys
import time
import torch
import torch.nn as nn
import mlflow
from PIL import Image
from torchvision import transforms

from config import config


# ── Model definition (copied from train.py to avoid importing train.py side-effects) ──

class CustomCNN(nn.Module):
    """
    Simple CNN with two prediction heads:
      - cls_head  : classifies the dominant object (num_classes logits)
      - bbox_head : regresses the bounding box [cx, cy, w, h] normalised to [0,1]
    """

    def __init__(self, num_classes: int, num_queries: int = 50):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.query_embed = nn.Embedding(num_queries, 256)
        self.cls_head = nn.Linear(256, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b = x.size(0)
        feat = self.trunk(self.features(x))
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        h = feat.unsqueeze(1) + q
        return self.cls_head(h), self.bbox_head(h)


# ── Preprocessing ──

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Helpers ──

def get_device():
    """Pick the best available device — works on Mac (MPS), Linux (CUDA) and CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(num_classes=80):
    """Load best_model.pth from the path defined in config."""
    checkpoint_path = os.path.join(
        config["path"]["run_base_dir"], "models", "best_model.pth"
    )
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: No checkpoint found at '{checkpoint_path}'")
        print("Train the model first: python src/train.py")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CustomCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model — epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    return model


def get_image_paths():
    """Collect image paths from the dataset directory defined in config."""
    if config["settings"]["use_sample_dataset"]:
        dataset_path = config["path"]["sample_dataset_path"]
    else:
        dataset_path = config["path"]["full_dataset_path"]

    images_dir = os.path.join(dataset_path, "images", "train")

    if not os.path.isdir(images_dir):
        print(f"ERROR: Image directory not found: '{images_dir}'")
        print("Check your config paths and run from the project root directory.")
        sys.exit(1)

    paths = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not paths:
        print(f"ERROR: No images found in '{images_dir}'")
        sys.exit(1)

    return paths


def preprocess_images(image_paths):
    """Load and preprocess a list of image paths into a batch tensor."""
    tensors = []
    for p in image_paths:
        try:
            tensors.append(TRANSFORM(Image.open(p).convert("RGB")))
        except Exception as e:
            print(f"  Warning: skipping '{p}' — {e}")
    if not tensors:
        print("ERROR: No images could be loaded in this batch.")
        sys.exit(1)
    return torch.stack(tensors)


def postprocess(pred_logits, pred_boxes, conf_threshold=0.5):
    """Convert raw model output to a list of detections per image."""
    probs = torch.softmax(pred_logits, dim=-1)
    results = []
    for b in range(probs.size(0)):
        detections = []
        for q in range(probs.size(1)):
            class_id = probs[b, q].argmax().item()
            confidence = probs[b, q, class_id].item()
            if class_id == 0 or confidence < conf_threshold:
                continue
            detections.append({
                "class_id": class_id,
                "confidence": round(confidence, 3),
                "box": [round(v, 3) for v in pred_boxes[b, q].tolist()],
            })
        results.append(sorted(detections, key=lambda d: d["confidence"], reverse=True))
    return results


# ── Main ──

def run_batch_inference(image_paths, batch_size=8, conf_threshold=0.5):
    """
    Run batch inference and log throughput + results to MLflow.

    Args:
        image_paths:    List of image file paths.
        batch_size:     Images per forward pass.
        conf_threshold: Minimum confidence to keep a detection.
    """
    device = get_device()
    print(f"Device: {device}")

    model = load_model()
    model.to(device)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment("custom_model_inference")

    all_detections = []
    total_time_ms = 0.0

    print(f"\nRunning inference on {len(image_paths)} images (batch_size={batch_size})\n")

    with mlflow.start_run(run_name="batch_inference"):
        mlflow.log_params({
            "num_images": len(image_paths),
            "batch_size": batch_size,
            "conf_threshold": conf_threshold,
            "device": str(device),
        })

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensor = preprocess_images(batch_paths).to(device)

            start = time.perf_counter()
            with torch.no_grad():
                pred_logits, pred_boxes = model(batch_tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000

            detections = postprocess(
                pred_logits.cpu(), pred_boxes.cpu(), conf_threshold
            )
            all_detections.extend(detections)
            total_time_ms += elapsed_ms

            throughput = len(batch_paths) / (elapsed_ms / 1000)
            print(f"Batch {i // batch_size + 1}: "
                  f"{len(batch_paths)} images | "
                  f"{elapsed_ms:.1f} ms | "
                  f"{throughput:.1f} img/s")
            for path, dets in zip(batch_paths, detections):
                print(f"  {os.path.basename(path)}: {len(dets)} detection(s)")

        total_detections = sum(len(d) for d in all_detections)
        avg_ms = total_time_ms / len(image_paths)
        print(f"\nTotal detections : {total_detections}")
        print(f"Avg time/image   : {avg_ms:.1f} ms")
        print(f"Total time       : {total_time_ms:.1f} ms")

        mlflow.log_metrics({
            "total_detections": float(total_detections),
            "avg_detections_per_image": total_detections / len(image_paths),
            "avg_inference_time_ms": avg_ms,
            "total_inference_time_ms": total_time_ms,
        })


if __name__ == "__main__":
    run_batch_inference(get_image_paths(), batch_size=8, conf_threshold=0.5)
