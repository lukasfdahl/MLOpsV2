# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import mlflow
import yaml
from config import config

from dataloader_sample import get_dataloaders
from helpers import (
    check_device,
    plot_training_curves,
    log_gpu_metrics,
    detection_loss_set,
)

device = check_device()

# mlflow ui --backend-store-uri sqlite:////Users/kaspe/Desktop/kls_repo/mlflow.db

# model configureation
EPOCHS = config["training"]["epochs"]
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
WEIGHT_DECAY = config["training"]["weight_decay"]

# Make sure the output dir exists
models_path = os.path.join(config["path"]["run_base_dir"], "models")
os.makedirs(models_path, exist_ok=True)


# Simple CNN as custom model for object detection,
# could be expanded / more fancy later
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

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 → 8
        )

        # Shared fully-connected trunk
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Query embedding
        self.query_embed = nn.Embedding(num_queries, 256)

        # Classification head
        self.cls_head = nn.Linear(256, num_classes + 1)

        # Bounding-box regression head
        # Sigmoid keeps the output in (0, 1) which matches normalised YOLO coords
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Query embedding
        b = x.size(0)
        feat = self.trunk(self.features(x))

        # add unsqueezed query embedding to each feature vector
        # to create num_queries parallel heads
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        h = feat.unsqueeze(1) + q

        # Pass through heads
        pred_logits = self.cls_head(h)
        pred_boxes = self.bbox_head(h)

        return pred_logits, pred_boxes


# train function
def train_model():

    print(f"starting training loop")
    # mlflow setup
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment("custom_model")

    with mlflow.start_run(run_name="custom_cnn_detection"):
        mlflow.log_params(
            {
                "model_type": "CustomCNN",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "loss_cls": "CrossEntropy",
                "loss_bbox": "SmoothL1",
            }
        )

        # To use smaller sample dataset if enabled
        if config["settings"]["use_sample_dataset"]:
            dataset_path = config["path"]["sample_dataset_path"]
        else:
            dataset_path = config["path"]["full_dataset_path"]

        train_loader, val_loader, num_classes = get_dataloaders(
            data_dir=dataset_path,
            batch_size=BATCH_SIZE,
        )

        model = CustomCNN(num_classes=num_classes).to(device)

        # Model summary (torchsummary struggles with MPS, so temporarily use CPU/CUDA)
        summary_device = "cuda" if device == "cuda" else "cpu"
        print("\nModel summary:")
        summary(model.to(summary_device), (3, 64, 64), device=summary_device)
        model.to(device)

        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Reduce LR when validation loss plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        best_val_loss = float("inf")
        best_epoch = 0

        print(f"\nTraining for {EPOCHS} epochs on {device}...\n")
        print(
            f"{'Epoch':>6}"
            f"{'Train Loss':>10}"
            f"{'Train Acc':>10}"
            f"{'Val Loss':>9}"
            f"{'Val Acc':>8}"
        )
        print("-" * 55)

        for epoch in range(1, EPOCHS + 1):

            #  training loop
            model.train()

            # reset
            train_loss = 0.0
            train_cls_loss = 0.0
            train_reg_loss = 0.0
            train_acc = 0.0
            train_total = 0

            for images, targets in train_loader:
                images = images.to(device)

                optimizer.zero_grad()
                pred_logits, pred_boxes = model(images)

                loss, metrics = detection_loss_set(
                    pred_logits, pred_boxes, targets, num_classes=model.num_classes
                )

                loss.backward()
                optimizer.step()

                bs = images.size(0)
                train_loss += loss.item() * bs
                train_cls_loss += metrics["loss_ce"] * bs
                train_reg_loss += metrics["loss_l1"] * bs
                train_acc += metrics["acc"] * bs
                train_total += bs

            # safety guard as had error earlier
            if train_total == 0:
                raise RuntimeError("Train dataloader yielded 0 samples.")

            train_loss /= train_total
            train_cls_loss /= train_total
            train_reg_loss /= train_total
            train_acc = 100.0 * train_acc / train_total

            # Validation loop
            model.eval()
            # reset (val)
            val_loss = 0.0
            val_cls_loss = 0.0
            val_reg_loss = 0.0
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
                    val_cls_loss += metrics["loss_ce"] * bs
                    val_reg_loss += metrics["loss_l1"] * bs
                    val_acc += metrics["acc"] * bs
                    val_total += bs

            val_loss /= val_total
            val_cls_loss /= val_total
            val_reg_loss /= val_total
            val_acc = 100.0 * val_acc / val_total

            scheduler.step(val_loss)

            # log in mlflow
            current_lr = optimizer.param_groups[0]["lr"]

            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            mlflow.log_metric("train_cls_loss", train_cls_loss, step=epoch)
            mlflow.log_metric("train_reg_loss", train_reg_loss, step=epoch)
            mlflow.log_metric("val_cls_loss", val_cls_loss, step=epoch)
            mlflow.log_metric("val_reg_loss", val_reg_loss, step=epoch)

            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            log_gpu_metrics(device, step=epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(
                f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  "
                f"{val_loss:>9.4f}  {val_acc:>7.2f}%"
            )

            # save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    os.path.join(models_path, "best_model.pth"),
                )
                print(f"          ↳ New best model saved (val_loss={val_loss:.4f})")

            # also save last model if fails to complete or more training
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                os.path.join(models_path, "last_model.pth"),
            )

        print("\n" + "=" * 55)
        print("Training complete.")
        print(f"  Best model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
        print(f"  Saved → {os.path.join(models_path, "best_model.pth")}")
        print(f"  Saved → {os.path.join(models_path, "last_model.pth")}")

        # figures / graphs — log directly to MLflow so they render in the UI
        fig = plot_training_curves(history)
        mlflow.log_figure(fig, "training_curves.png")

        mlflow.log_artifact(os.path.join(models_path, "best_model.pth"))


if __name__ == "__main__":
    train_model()
