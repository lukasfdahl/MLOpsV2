"""
tests/test_dataset.py

Tests for the unified dataloader (both YOLO and COCO JSON formats).
The YOLO tests run against coco128_small and are fast (< 10 s).
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataloader import YoloDataset, get_dataloaders, _detect_format

DATA_DIR   = "data/coco128_small"
IMAGES_DIR = os.path.join(DATA_DIR, "images", "train2017")
LABELS_DIR = os.path.join(DATA_DIR, "labels",  "train2017")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(IMAGES_DIR),
    reason="coco128_small not found — skipping dataset tests",
)


class TestFormatDetection:

    def test_detects_yolo_format(self):
        assert _detect_format(DATA_DIR) == "yolo"

    def test_raises_on_unknown_format(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _detect_format(str(tmp_path))


class TestYoloDataset:

    def setup_method(self):
        self.dataset = YoloDataset(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR)

    def test_not_empty(self):
        assert len(self.dataset) > 0

    def test_returns_image_and_target(self):
        image, target = self.dataset[0]
        assert isinstance(target, dict)
        assert "labels" in target and "boxes" in target

    def test_image_shape_with_transform(self):
        from torchvision import transforms
        ds = YoloDataset(
            IMAGES_DIR, LABELS_DIR,
            transform=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
        )
        image, _ = ds[0]
        assert image.shape == (3, 64, 64)

    def test_boxes_in_unit_range(self):
        from torchvision import transforms
        ds = YoloDataset(
            IMAGES_DIR, LABELS_DIR,
            transform=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
        )
        for i in range(min(10, len(ds))):
            _, t = ds[i]
            assert t["boxes"].min() >= 0.0
            assert t["boxes"].max() <= 1.0

    def test_labels_valid_class_ids(self):
        from torchvision import transforms
        ds = YoloDataset(
            IMAGES_DIR, LABELS_DIR,
            transform=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
        )
        for i in range(min(10, len(ds))):
            _, t = ds[i]
            assert t["labels"].dtype == torch.long
            assert t["labels"].min() >= 0
            assert t["labels"].max() <= 79

    def test_all_objects_loaded(self):
        """All objects in a label file should be loaded, not just the first."""
        from torchvision import transforms
        ds = YoloDataset(
            IMAGES_DIR, LABELS_DIR,
            transform=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
        )
        # Find a sample with multiple objects and verify count matches file
        for i in range(len(ds)):
            label_path = os.path.join(LABELS_DIR, ds.image_files[i].replace(".jpg", ".txt"))
            with open(label_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) > 1:
                _, t = ds[i]
                assert len(t["labels"]) == len(lines), (
                    f"Expected {len(lines)} objects, got {len(t['labels'])}"
                )
                break


class TestGetDataloaders:

    def test_returns_three_values(self):
        train_loader, val_loader, num_classes = get_dataloaders(
            data_dir=DATA_DIR, batch_size=4, img_size=64
        )
        assert train_loader is not None
        assert val_loader is not None
        assert num_classes == 80

    def test_batch_shape(self):
        train_loader, _, _ = get_dataloaders(data_dir=DATA_DIR, batch_size=4, img_size=64)
        images, targets = next(iter(train_loader))
        assert images.shape == (4, 3, 64, 64) or images.shape[1:] == (3, 64, 64)

    def test_targets_match_batch_size(self):
        train_loader, _, _ = get_dataloaders(data_dir=DATA_DIR, batch_size=4, img_size=64)
        images, targets = next(iter(train_loader))
        assert len(targets) == images.shape[0]