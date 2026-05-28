"""
tests/test_model.py

Tests for CustomCNN — output shapes, forward pass, and a 1-step training check.
These tests run on CPU only and complete in a few seconds.
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import CustomCNN


NUM_CLASSES = 80
NUM_QUERIES = 10   # keep small for speed
BATCH_SIZE = 2
IMG_SIZE = 64


@pytest.fixture
def model():
    """Return a small CPU model for each test."""
    return CustomCNN(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES)


@pytest.fixture
def dummy_batch():
    """Random image batch — no real data needed."""
    return torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)


class TestCustomCNNOutputShape:

    def test_pred_logits_shape(self, model, dummy_batch):
        """cls head output: (B, num_queries, num_classes + 1)."""
        pred_logits, _ = model(dummy_batch)
        assert pred_logits.shape == (BATCH_SIZE, NUM_QUERIES, NUM_CLASSES + 1), (
            f"Expected {(BATCH_SIZE, NUM_QUERIES, NUM_CLASSES + 1)}, "
            f"got {pred_logits.shape}"
        )

    def test_pred_boxes_shape(self, model, dummy_batch):
        """bbox head output: (B, num_queries, 4)."""
        _, pred_boxes = model(dummy_batch)
        assert pred_boxes.shape == (BATCH_SIZE, NUM_QUERIES, 4), (
            f"Expected {(BATCH_SIZE, NUM_QUERIES, 4)}, got {pred_boxes.shape}"
        )

    def test_pred_boxes_in_unit_range(self, model, dummy_batch):
        """Sigmoid on bbox head must keep all values in [0, 1]."""
        _, pred_boxes = model(dummy_batch)
        assert pred_boxes.min().item() >= 0.0
        assert pred_boxes.max().item() <= 1.0

    def test_single_image_batch(self, model):
        """Model should handle batch size 1 without errors."""
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        logits, boxes = model(x)
        assert logits.shape[0] == 1
        assert boxes.shape[0] == 1


class TestCustomCNNTrainingStep:

    def test_loss_is_scalar_and_finite(self, model, dummy_batch):
        """A single forward + backward should produce a finite scalar loss."""
        import torch.nn as nn
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        pred_logits, pred_boxes = model(dummy_batch)

        # Dummy targets: one object per image, class 0, box at centre
        targets = [
            {
                "labels": torch.tensor([0], dtype=torch.long),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
            }
            for _ in range(BATCH_SIZE)
        ]

        # Simple loss: CE on first query slot + L1 on first box
        ce_loss = nn.CrossEntropyLoss()
        l1_loss = nn.SmoothL1Loss()

        # Use first query slot prediction for this sanity check
        cls_loss = ce_loss(
            pred_logits[:, 0, :],                        # (B, num_classes+1)
            torch.tensor([t["labels"][0] for t in targets]),  # (B,)
        )
        box_loss = l1_loss(
            pred_boxes[:, 0, :],                         # (B, 4)
            torch.stack([t["boxes"][0] for t in targets]),    # (B, 4)
        )
        loss = cls_loss + box_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() == loss.item(), "Loss is NaN"   # NaN check
        assert loss.item() < float("inf"), "Loss is infinite"

    def test_parameters_update_after_step(self, model, dummy_batch):
        """Model weights must change after one gradient step."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        import torch.nn as nn

        # Snapshot weights before
        before = [p.clone() for p in model.parameters()]

        pred_logits, pred_boxes = model(dummy_batch)
        targets_labels = torch.zeros(BATCH_SIZE, dtype=torch.long)
        loss = nn.CrossEntropyLoss()(pred_logits[:, 0, :], targets_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        after = list(model.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "No parameters were updated after the optimizer step"


class TestCustomCNNEdgeCases:

    def test_different_num_queries(self):
        """Model should work with any num_queries value."""
        for q in [1, 5, 100]:
            m = CustomCNN(num_classes=10, num_queries=q)
            x = torch.randn(2, 3, 64, 64)
            logits, boxes = m(x)
            assert logits.shape == (2, q, 11)
            assert boxes.shape == (2, q, 4)

    def test_different_num_classes(self):
        """Model should work with any num_classes value."""
        for nc in [2, 10, 80]:
            m = CustomCNN(num_classes=nc, num_queries=5)
            x = torch.randn(1, 3, 64, 64)
            logits, boxes = m(x)
            assert logits.shape[-1] == nc + 1