# train.py
import os
import torch
import torch.optim as optim
from torchsummary import summary
import mlflow
from config import config
from model import CustomCNN
from dataloader import get_dataloaders
from utility.training import plot_training_curves, log_gpu_metrics, detection_loss_set
from utility.hardware import check_device
from carbontracker.tracker import CarbonTracker
device = check_device()

# Training hyperparameters from config
EPOCHS = config["training"]["epochs"]
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
WEIGHT_DECAY = config["training"]["weight_decay"]

models_path = os.path.join(config["path"]["run_base_dir"], "models")
os.makedirs(models_path, exist_ok=True)

# Main training loop


def train_model():

    print("Starting training loop")

    # MLflow setup
    mlflow.set_tracking_uri(os.environ.get(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("custom_model")

    # Log hyperparameters and dataset info to MLflow
    with mlflow.start_run(run_name="custom_cnn_detection"):
        mlflow.log_params({
            "model_type": "CustomCNN",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": "Adam", "scheduler": "ReduceLROnPlateau",
            "loss_cls": "CrossEntropy", "loss_bbox": "SmoothL1",
        })

        # Config decides which dataset path to use — dataloader figures out the format
        dataset_path = (
            config["path"]["sample_dataset_path"]
            if config["settings"]["use_sample_dataset"]
            else config["path"]["full_dataset_path"]
        )

        # Get dataloaders from dataloader.py
        train_loader, val_loader, num_classes = get_dataloaders(
            data_dir=dataset_path, batch_size=BATCH_SIZE
        )

        # from model.py
        model = CustomCNN(num_classes=num_classes).to(device)

       # Print model summary using torchsummary (handles CUDA/non-CUDA devices)
        summary_device = "cuda" if str(device) == "cuda" else "cpu"
        print("\nModel summary:")
        summary(model.to(summary_device), (3, 64, 64), device=summary_device)
        model.to(device)

        # optim and scheduler setup
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3)

        # Train the model
        history = {"train_loss": [], "val_loss": [],
                  
                   "train_acc": [], "val_acc": []}
        best_val_loss, best_epoch = float("inf"), 0

        print(f"\nTraining for {EPOCHS} epochs on {device}...\n")
        print(
            f"{'Epoch':>6}{'Train Loss':>12}{'Train Acc':>11}{'Val Loss':>10}{'Val Acc':>9}")
        print("-" * 55)

        # Carbon tracking — tracks energy and CO2 for the full training run
        # Gracefully skips on environments without supported hardware (e.g. CI/CD Docker)
        try:
            tracker = CarbonTracker(
                epochs=EPOCHS,
                log_dir=config["path"]["run_base_dir"],
                components="gpu",  # GPU only — skip CPU (no RAPL permissions on AI-LAB)
            )
            carbon_available = True
            print("CarbonTracker: GPU tracking enabled")
        except Exception as e:
            print(f"CarbonTracker: hardware unavailable, skipping ({e})")
            carbon_available = False

        # simple training loop with train/val phases and MLflow logging for now
        for epoch in range(1, EPOCHS + 1):

            # start tracking carbon for this epoch
            if carbon_available:
                tracker.epoch_start()

            # Training
            model.train()
            train_loss = train_acc = train_total = 0
            for images, targets in train_loader:
                images = images.to(device)
                optimizer.zero_grad()
                pred_logits, pred_boxes = model(images)
                loss, metrics = detection_loss_set(
                    pred_logits, pred_boxes, targets, num_classes=model.num_classes)
                loss.backward()
                optimizer.step()
                bs = images.size(0)
                train_loss += loss.item() * bs
                train_acc += metrics["acc"] * bs
                train_total += bs

            if train_total == 0:
                raise RuntimeError("Train dataloader yielded 0 samples.")

            train_loss /= train_total
            train_acc = 100.0 * train_acc / train_total

            # Validation
            model.eval()
            val_loss = val_acc = val_total = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    pred_logits, pred_boxes = model(images)
                    loss, metrics = detection_loss_set(
                        pred_logits, pred_boxes, targets, num_classes=model.num_classes)
                    bs = images.size(0)
                    val_loss += loss.item() * bs
                    val_acc += metrics["acc"] * bs
                    val_total += bs

            val_loss /= val_total
            val_acc = 100.0 * val_acc / val_total

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # log mletrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc,   "val_acc": val_acc,
                "learning_rate": current_lr,
            }, step=epoch)
            log_gpu_metrics(device, step=epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(
                f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  {val_loss:>9.4f}  {val_acc:>7.2f}%")

            # save best model based on val_loss, and also save last model each epoch
            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(
                    ), "val_loss": val_loss, "val_acc": val_acc},
                    os.path.join(models_path, "best_model.pth"),
                )
                print(
                    f"          ↳ New best model saved (val_loss={val_loss:.4f})")

            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(
                ), "val_loss": val_loss, "val_acc": val_acc},
                os.path.join(models_path, "last_model.pth"),
            )

            # end carbon tracking for this epoch    
            if carbon_available:
                tracker.epoch_end()

        print("\n" + "=" * 55)
        print(
            f"Training complete. Best epoch {best_epoch}, val_loss={best_val_loss:.4f}")

        # plot training curves and log to MLflow, along with the best model checkpoint
        fig = plot_training_curves(history)
        mlflow.log_figure(fig, "training_curves.png")
        mlflow.log_artifact(os.path.join(models_path, "best_model.pth"))

        if carbon_available:
            tracker.stop()

        # Log carbon footprint to MLflow
        carbon_log = os.path.join(config["path"]["run_base_dir"], "carbontracker")
        if carbon_available and os.path.exists(carbon_log):
            mlflow.log_artifacts(carbon_log, name="carbontracker")
            print("Carbon footprint logged to MLflow")

        # Register model in MLflow model registry if it meets performance criteria
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="CustomCNN",
        )
        print(f"Model registered in MLflow registry (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    train_model()
