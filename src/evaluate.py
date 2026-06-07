# evaluate.py
import os
import sys
import torch
import mlflow
import mlflow.pytorch

from config import config
from train import CustomCNN
from utility.hardware import check_device
from utility.testing import show_predictions

if config["settings"]["use_sample_dataset"]:
    from dataloader_sample import get_dataloaders
else:
    from dataloader_full import get_dataloaders

from utility.training import detection_loss_set

# Performance threshold — model must beat this val_loss to be registered
VAL_LOSS_THRESHOLD = 7.5
REGISTERED_MODEL_NAME = "CustomCNN"

models_path = os.path.join(config["path"]["run_base_dir"], "models")
checkpoint_path = os.path.join(models_path, "best_model.pth")


def evaluate_model():
    """
    Load the best saved model, evaluate it on the validation set,
    and register it in the MLflow model registry if it meets the
    performance threshold.
    """
    # ── Setup ──
    device = check_device()

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment("custom_model_evaluation")

    # ── Load checkpoint ──
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: No checkpoint found at '{checkpoint_path}'")
        print("Train the model first: python src/train.py")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint — epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    # ── Load data ──
    if config["settings"]["use_sample_dataset"]:
        dataset_path = config["path"]["sample_dataset_path"]
    else:
        dataset_path = config["path"]["full_dataset_path"]

    _, val_loader, num_classes = get_dataloaders(
        data_dir=dataset_path,
        batch_size=config["training"]["batch_size"],
    )

    # ── Build model ──
    model = CustomCNN(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ── Evaluate on validation set ──
    val_loss = 0.0
    val_acc = 0.0
    val_total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            pred_logits, pred_boxes = model(images)
            loss, metrics = detection_loss_set(
                pred_logits, pred_boxes, targets, num_classes=model.num_classes
            )
            bs = images.size(0)
            val_loss += loss.item() * bs
            val_acc += metrics["acc"] * bs
            val_total += bs

    val_loss /= val_total
    val_acc = 100.0 * val_acc / val_total

    print(f"Evaluation — val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")

    # ── Save prediction visualisation ──
    pred_path = os.path.join(models_path, "eval_predictions.png")
    show_predictions(model, val_loader, device, num_examples=8, save_path=pred_path)

    # ── Log to MLflow and register if good enough ──
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_params({
            "checkpoint_epoch": checkpoint["epoch"],
            "val_loss_threshold": VAL_LOSS_THRESHOLD,
            "num_classes": num_classes,
        })
        mlflow.log_metrics({
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        mlflow.log_artifact(pred_path, artifact_path="predictions")
        mlflow.log_artifact(checkpoint_path, artifact_path="model_checkpoint")
        mlflow.log_artifact("README.md", artifact_path="model_card")

        if val_loss < VAL_LOSS_THRESHOLD:
            print(f"Model passed threshold (val_loss={val_loss:.4f} < {VAL_LOSS_THRESHOLD}) — registering in MLflow.")
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
            print(f"Model registered as '{REGISTERED_MODEL_NAME}' in MLflow registry.")
        else:
            print(f"Model did NOT pass threshold (val_loss={val_loss:.4f} >= {VAL_LOSS_THRESHOLD}) — skipping registration.")

    return val_loss, val_acc


if __name__ == "__main__":
    evaluate_model()