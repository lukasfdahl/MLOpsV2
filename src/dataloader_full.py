import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json


# Intended for full dataset
# Custom dataset for COCO classification
# (using YOLO format labels) for image classification + bbox regression
class CocoClassificationDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Load annotations JSON
        with open(annotation_file, "r") as f:
            data = json.load(f)

        # Build category ID → contiguous 0-indexed label mapping
        # COCO uses 80 classes but IDs are non-contiguous (1-90 with gaps)
        cat_ids_sorted = sorted([c["id"] for c in data["categories"]])
        self.cat_id_to_idx = {cid: idx for idx,
                              cid in enumerate(cat_ids_sorted)}
        self.num_classes = len(cat_ids_sorted)

        # Build image_id → image metadata map
        id_to_image = {img["id"]: img for img in data["images"]}

        # Group all annotations by image_id
        anns_by_image = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        for img_id, anns in anns_by_image.items():
            img_meta = id_to_image.get(img_id)
            if img_meta is None:
                continue
            img_path = os.path.join(images_dir, img_meta["file_name"])
            if not os.path.exists(img_path):
                continue
            self.samples.append(
                {
                    "img_path": img_path,
                    "annotations": anns,
                    "img_w": img_meta["width"],
                    "img_h": img_meta["height"],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["img_path"]).convert("RGB")

        img_w, img_h = sample["img_w"], sample["img_h"]

        # normalize bbox
        # Convert COCO bbox [x, y, w, h] (pixels) to YOLO [cx, cy, w, h]
        boxes = []
        labels = []
        for ann in sample["annotations"]:
            x, y, bw, bh = ann["bbox"]  # [x, y, w, h] in pixels
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        boxes = torch.tensor(boxes, dtype=torch.float32).clamp(0.0, 1.0)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, {"labels": labels, "boxes": boxes}


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


#  DataLoader function
def get_dataloaders(data_dir="dataset/coco", batch_size=16, img_size=64):

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Uses official COCO train/val splits instead of random split
    train_dataset = CocoClassificationDataset(
        images_dir=os.path.join(data_dir, "train2017"),
        annotation_file=os.path.join(
            data_dir, "annotations", "instances_train2017.json"
        ),
        transform=transform,
    )

    val_dataset = CocoClassificationDataset(
        images_dir=os.path.join(data_dir, "val2017"),
        annotation_file=os.path.join(
            data_dir, "annotations", "instances_val2017.json"),
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    # add later the test loader too if neededd

    num_classes = train_dataset.num_classes  # 80 for full COCO
    print(
        f"Dataset size: {len(train_dataset)} train / {len(val_dataset)} val, "
        f"{num_classes} classes"
    )
    return train_loader, val_loader, num_classes
