"""
tests/test_training.py

Smoke test: can the full training loop run for 1 epoch on coco128_small?
This catches integration bugs (dataloader ↔ model ↔ loss) before pushing.
Runs on CPU; completes in ~10-20 s.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DATA_DIR = "data/coco128_small"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATA_DIR, "images")),
    reason="coco128_small dataset not found — skipping training smoke test",
)


def test_one_epoch_smoke():
    """
    Run exactly one epoch of training + validation on coco128_small.
    Pass criteria: no exception, loss is a finite number.
    """
    from dataloader import get_dataloaders
    from model import CustomCNN
    from utility.training import detection_loss_set

    device = torch.device("cpu")  # always CPU for CI

    train_loader, val_loader, num_classes = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=8,
        img_size=64,
        val_split=0.2,
    )

    model = CustomCNN(num_classes=num_classes, num_queries=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training pass
    model.train()
    train_loss_total = 0.0
    train_total = 0

    for images, targets in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        pred_logits, pred_boxes = model(images)
        loss, _ = detection_loss_set(
            pred_logits, pred_boxes, targets, num_classes=num_classes
        )
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item() * images.size(0)
        train_total += images.size(0)

    assert train_total > 0, "No training samples were processed"
    train_loss = train_loss_total / train_total
    assert train_loss == train_loss, "Training loss is NaN"
    assert train_loss < float("inf"), "Training loss is infinite"

    # ── Validation pass ───────────────────────────────────────────────────────
    model.eval()
    val_loss_total = 0.0
    val_total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            pred_logits, pred_boxes = model(images)
            loss, _ = detection_loss_set(
                pred_logits, pred_boxes, targets, num_classes=num_classes
            )
            val_loss_total += loss.item() * images.size(0)
            val_total += images.size(0)

    assert val_total > 0, "No validation samples were processed"
    val_loss = val_loss_total / val_total
    assert val_loss == val_loss, "Validation loss is NaN"
    assert val_loss < float("inf"), "Validation loss is infinite"

    print(f"\nSmoke test passed — train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")