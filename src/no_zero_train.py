# train.py
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary
import mlflow
from config import config
from model import CustomCNN
from dataloader import get_dataloaders
from utility.training import plot_training_curves, log_gpu_metrics, detection_loss_set
from utility.testing import show_predictions
from utility.hardware import check_device
from carbontracker.tracker import CarbonTracker
from tqdm import tqdm

# Training hyperparameters from config
EPOCHS = config["training"]["epochs"]
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
WEIGHT_DECAY = config["training"]["weight_decay"]
NUM_GPUS = config["training"].get("multi_gpu", 1)
NUM_WORKERS = config["dataloader"].get("num_workers", 4)
EARLY_STOPPING_PATIENCE = config["training"].get("early_stopping_patience", 10)
WARMUP_EPOCHS = config["training"].get("warmup_epochs", 5)

models_path = os.path.join(config["path"]["run_base_dir"], "models")
os.makedirs(models_path, exist_ok=True)


def _is_main_process():
    """Returns True if this is the main process (rank 0 in DDP, or single GPU)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _run_one_epoch_train(model, train_loader, optimizer, scaler, device, use_amp, epoch, is_main):
    model.train()
    train_loss = train_acc = train_total = 0

    device_type = "cuda" if device.type == "cuda" else "cpu"
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False, disable=not is_main)

    for images, targets in pbar:
        images = images.to(device)
        optimizer.zero_grad()

        # Automatic Mixed Precision (AMP) forward pass
        with torch.autocast(device_type=device_type, enabled=use_amp):
            pred_logits, pred_boxes = model(images)
            loss, metrics = detection_loss_set(
                pred_logits, pred_boxes, targets,
                num_classes=model.module.num_classes if hasattr(model, "module") else model.num_classes
            )

        # AMP backwards pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        train_loss += loss.item() * bs
        train_acc += metrics["acc"] * bs
        train_total += bs

        pbar.set_postfix(loss=f"{train_loss / train_total:.4f}", acc=f"{100.0 * train_acc / train_total:.2f}%")

    if train_total == 0:
        raise RuntimeError("Train dataloader yielded 0 samples.")

    return train_loss / train_total, 100.0 * train_acc / train_total


def _run_one_epoch_val(model, val_loader, device, use_amp, epoch, is_main):
    model.eval()
    val_loss = val_acc = val_total = 0

    device_type = "cuda" if device.type == "cuda" else "cpu"
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [val]  ", leave=False, disable=not is_main)

    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)

            with torch.autocast(device_type=device_type, enabled=use_amp):
                pred_logits, pred_boxes = model(images)
                loss, metrics = detection_loss_set(
                    pred_logits, pred_boxes, targets,
                    num_classes=model.module.num_classes if hasattr(model, "module") else model.num_classes
                )

            bs = images.size(0)
            val_loss += loss.item() * bs
            val_acc += metrics["acc"] * bs
            val_total += bs

            pbar.set_postfix(loss=f"{val_loss / val_total:.4f}", acc=f"{100.0 * val_acc / val_total:.2f}%")

    return val_loss / val_total, 100.0 * val_acc / val_total


# Main training loop
def train_model(rank=None, world_size=None, override_epochs=None, override_lr=None, override_warmup=None):
    """
    Main training function. Works for both single-GPU and DDP multi-GPU.
    rank: process rank (0 = main process). Set automatically by torchrun.
    world_size: total number of processes (= number of GPUs).
    """
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Allow fine-tuning phase to override key hyperparams without changing config
    global EPOCHS, LEARNING_RATE, WARMUP_EPOCHS
    if override_epochs  is not None: EPOCHS        = override_epochs
    if override_lr      is not None: LEARNING_RATE = override_lr
    if override_warmup  is not None: WARMUP_EPOCHS = override_warmup

    # DDP setup — initialize process group when running with multiple GPUs
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(f"[Rank {rank}] DDP initialized on {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AMP scaler
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    is_main = _is_main_process()

    # MLflow setup, only on rank 0 to avoid duplicate logging
    if is_main:
        print("Starting training loop")
        mlflow.set_tracking_uri(os.environ.get(
            "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("custom_model")
        run = mlflow.start_run(run_name="custom_cnn_detection")
        mlflow.log_params({
            "model_type": "CustomCNN",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": "Adam", "scheduler": "ReduceLROnPlateau",
            "loss_cls": "CrossEntropy", "loss_bbox": "SmoothL1",
            "num_gpus": world_size, "amp": use_amp,
        })

    # Config decides which dataset path to use, dataloader figures out the format
    dataset_path = (
        config["path"]["sample_dataset_path"]
        if config["settings"]["use_sample_dataset"]
        else config["path"]["full_dataset_path"]
    )

    # Get dataloaders, use DistributedSampler in DDP mode so each GPU gets different data
    train_loader, val_loader, num_classes = get_dataloaders(
        data_dir=dataset_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        distributed=is_ddp,
        rank=rank,
        world_size=world_size,
    )

    # Build model and move to device
    model = CustomCNN(num_classes=num_classes).to(device)

    # Print model summary only on rank 0
    if is_main:
        summary_device = "cuda" if str(device).startswith("cuda") else "cpu"
        print("\nModel summary:")
        summary(model.to(summary_device), (3, 64, 64), device=summary_device)
        model.to(device)
    if is_ddp:
        # DDP wraps the model — gradients are averaged across GPUs automatically
        model = DDP(model, device_ids=[rank])
        if is_main:
            print(f"DDP enabled across {world_size} GPUs")

    # optim and scheduler setup
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Linear warmup for first WARMUP_EPOCHS, then hand off to ReduceLROnPlateau
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    # Train the model
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_epoch = float("inf"), 0
    early_stopping_counter = 0

    if is_main:
        print(f"\nTraining for {EPOCHS} epochs on {device} "
              f"({'DDP x' + str(world_size) + ' GPUs' if is_ddp else 'single GPU'}, "
              f"AMP={'on' if use_amp else 'off'}, "
              f"warmup={WARMUP_EPOCHS} epochs, "
              f"early_stopping={EARLY_STOPPING_PATIENCE} epochs)...\n")

    # Carbon tracking — only on rank 0, and only if CUDA is available
    # is_main check must come first to avoid all DDP ranks initializing their own tracker
    carbon_available = is_main and torch.cuda.is_available()
    if carbon_available:
        tracker = CarbonTracker(
            epochs=EPOCHS,
            log_dir=config["path"]["run_base_dir"],
            components="gpu",  # GPU only — skip CPU (no RAPL permissions on AI-LAB)
        )
        print("CarbonTracker: GPU tracking enabled")
    else:
        if is_main:
            print("CarbonTracker: no GPU detected, skipping carbon tracking")

    # simple training loop with train/val phases and MLflow logging for now
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", disable=not is_main)
    for epoch in epoch_bar:

        # start tracking carbon for this epoch
        if carbon_available:
            tracker.epoch_start()

        # In DDP mode, set epoch on sampler so shuffling is different each epoch
        if is_ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Training
        train_loss, train_acc = _run_one_epoch_train(
            model, train_loader, optimizer, scaler, device, use_amp, epoch, is_main)

        # Validation
        val_loss, val_acc = _run_one_epoch_val(model, val_loader, device, use_amp, epoch, is_main)

        # Warmup for first N epochs, then ReduceLROnPlateau takes over
        if warmup_scheduler is not None and plateau_scheduler is not None:
            if epoch <= WARMUP_EPOCHS:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # log metrics to MLflow — only on rank 0
        if is_main:
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

        if is_main:
            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}", train_acc=f"{train_acc:.2f}%",
                val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.2f}%",
            )
            tqdm.write(f"Epoch {epoch:>3}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%")

        # save best model based on val_loss, and also save last model each epoch
        # Only rank 0 saves models to avoid file conflicts
        if is_main:
        # Unwrap DDP model for saving
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch
                early_stopping_counter = 0
                torch.save(
                    {"epoch": epoch, "model_state": model_state,
                     "val_loss": val_loss, "val_acc": val_acc},
                    os.path.join(models_path, "best_model.pth"),
                )
                tqdm.write(f"          ↳ New best model saved (val_loss={val_loss:.4f})")
            else:
                early_stopping_counter += 1
                tqdm.write(f"          ↳ No improvement ({early_stopping_counter}/{EARLY_STOPPING_PATIENCE})")

            torch.save(
                {"epoch": epoch, "model_state": model_state,
                 "val_loss": val_loss, "val_acc": val_acc},
                os.path.join(models_path, "last_model.pth"),
            )

        # Early stopping check
        if is_ddp:
            counter_tensor = torch.tensor(early_stopping_counter, device=device)
            dist.broadcast(counter_tensor, src=0)
            early_stopping_counter = int(counter_tensor.item())

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            if is_main:
                tqdm.write(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
            break

        # end carbon tracking for this epoch
        if carbon_available:
            tracker.epoch_end()

    # Post-training logging — only on rank 0
    if is_main:
        print("\n" + "=" * 55)
        print(f"Training complete. Best epoch {best_epoch}, val_loss={best_val_loss:.4f}")

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

        # Build a clean CPU model from the saved checkpoint — avoids pickling
        # the DDP process group which MLflow cannot serialize.
        model_to_log = CustomCNN(num_classes=num_classes)
        _ckpt = torch.load(os.path.join(models_path, "best_model.pth"), map_location="cpu")
        model_to_log.load_state_dict(_ckpt["model_state"])
        model_to_log.eval()

        # Generate prediction examples on val set and log to MLflow
        pred_img_path = os.path.join(config["path"]["run_base_dir"], "predictions.png")

        show_predictions(model_to_log, val_loader, torch.device("cpu"), save_path=pred_img_path)
        mlflow.log_artifact(pred_img_path)
        print("Prediction examples logged to MLflow")
        mlflow.pytorch.log_model(
            model_to_log,
            name="model",
            registered_model_name="CustomCNN",
        )
        print(f"Model registered in MLflow registry (val_loss={best_val_loss:.4f})")
        mlflow.end_run()

    # DDP cleanup
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun sets LOCAL_RANK and WORLD_SIZE automatically when launching with multiple GPUs
    # Single GPU: python src/main.py
    # Multi GPU:  torchrun --nproc_per_node=2 src/main.py
    # rank/world_size are read from env inside train_model(), so calling with
    # no args works whether launched directly or imported by main.py
    train_model()