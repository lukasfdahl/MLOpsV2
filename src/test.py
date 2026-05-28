# future test script for custom model cnn
import torch
import os
from utility.testing import show_predictions
from utility.hardware import check_device
from train import CustomCNN
from config import config
from dataloader import get_dataloaders

models_path = os.path.join(config["path"]["run_base_dir"], "models")

CHECKPOINT = os.path.join(models_path, "best_model.pth")
NUM_CLASSES = 80

device = check_device()

# Load data — only need the val loader
_, val_loader, _ = get_dataloaders()

# Build model and load saved weights
model = CustomCNN(num_classes=NUM_CLASSES).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state"])
print(
    f"Loaded checkpoint — epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

# Run inference and save the prediction grid
show_predictions(
    model,
    val_loader,
    device,
    num_examples=8,
    save_path=os.path.join(models_path, "test_predictions.png"),
)
