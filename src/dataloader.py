import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

"""
dataset classes for the two supported formats: YOLO text files + images, and COCO JSON annotations + images in folders. 
Both return targets in the same format (labels and boxes tensors) to keep the training loop clean and format-agnostic.
"""

# dataset classes


class YoloDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # We only include images that have a corresponding non-empty label file to avoid issues during training
        all_images = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".png"))
        )
        self.image_files = []
        for f in all_images:
            label_path = os.path.join(
                labels_dir,
                f.replace(".jpg", ".txt").replace(".png", ".txt"),
            )
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)

    # Each label file contains lines like: "class_id cx cy w h" (all normalized to [0, 1])
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(
            self.labels_dir,
            img_name.replace(".jpg", ".txt").replace(".png", ".txt"),
        )

        # Load the image and labels from disk.
        image = Image.open(img_path).convert("RGB")

        # Parse the label file. We clamp boxes to [0, 1] just in case of any annotation errors.
        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                labels.append(int(parts[0]))
                boxes.append([float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4])])

        if self.transform:
            image = self.transform(image)

        # Return the image and a dict of targets. The "labels" tensor is (num_objects,)
        return image, {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).clamp(0.0, 1.0),
        }


# bigger dataset format: COCO JSON annotations + images in folders. This is what the full COCO dataset uses, and is more flexible for future extensions (e.g. adding test set, instance segmentation masks, etc.)
class CocoJsonDataset(Dataset):
    def __init__(self, images_dir: str, annotation_file: str, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(annotation_file) as f:
            data = json.load(f)

        # COCO category IDs are non-contiguous (1-90 with gaps) → remap to 0-79
        cat_ids_sorted = sorted(c["id"] for c in data["categories"])
        self.cat_id_to_idx = {cid: idx for idx,
                              cid in enumerate(cat_ids_sorted)}
        self.num_classes = len(cat_ids_sorted)

        id_to_image = {img["id"]: img for img in data["images"]}

        # group annotations by image_id for easy lookup in __getitem__
        anns_by_image: dict = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        for img_id, anns in anns_by_image.items():
            meta = id_to_image.get(img_id)
            if meta is None:
                continue
            img_path = os.path.join(images_dir, meta["file_name"])
            if not os.path.exists(img_path):
                continue
            self.samples.append({
                "img_path": img_path,
                "annotations": anns,
                "img_w": meta["width"],
                "img_h": meta["height"],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img_path"]).convert("RGB")
        W, H = sample["img_w"], sample["img_h"]

        boxes, labels = [], []
        for ann in sample["annotations"]:
            x, y, bw, bh = ann["bbox"]
            boxes.append([(x + bw / 2) / W, (y + bh / 2) / H, bw / W, bh / H])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        if self.transform:
            image = self.transform(image)

        return image, {
            "labels": torch.tensor(labels, dtype=torch.long),
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).clamp(0.0, 1.0),
        }


# collate function to handle batches of variable numbers of objects (targets are lists of dicts instead of stacked tensors)
def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


# helper function to detect dataset format based on directory structure
def _detect_format(data_dir: str) -> str:
    if os.path.isdir(os.path.join(data_dir, "labels")):
        return "yolo"
    if os.path.isdir(os.path.join(data_dir, "annotations")):
        return "coco_json"
    raise FileNotFoundError(
        f"Cannot determine dataset format in '{data_dir}'. "
        "Expected a 'labels/' folder (YOLO) or 'annotations/' folder (COCO JSON)."
    )

# main function to build dataloaders from config path


def get_dataloaders(
    data_dir: str = "data/coco128_small",
    batch_size: int = 16,
    img_size: int = 64,
    val_split: float = 0.2,
):
    fmt = _detect_format(data_dir)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # if yolo format // small dataset
    if fmt == "yolo":
        full_dataset = YoloDataset(
            images_dir=os.path.join(data_dir, "images", "train2017"),
            labels_dir=os.path.join(data_dir, "labels",  "train2017"),
            transform=transform,
        )

        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        num_classes = 80  # COCO128 is a subset of COCO 80 classes

        print(
            f"[dataloader] YOLO format | {len(full_dataset)} images "
            f"({train_size} train / {val_size} val) | {num_classes} classes"
        )

    # big dataset
    else:
        train_ds = CocoJsonDataset(
            images_dir=os.path.join(data_dir, "train2017"),
            annotation_file=os.path.join(
                data_dir, "annotations", "instances_train2017.json"
            ),
            transform=transform,
        )
        val_ds = CocoJsonDataset(
            images_dir=os.path.join(data_dir, "val2017"),
            annotation_file=os.path.join(
                data_dir, "annotations", "instances_val2017.json"
            ),
            transform=transform,
        )
        num_classes = train_ds.num_classes  # 80 for full COCO

        print(
            f"[dataloader] COCO JSON format | {len(train_ds)} train / "
            f"{len(val_ds)} val | {num_classes} classes"
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, num_classes
