import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# Custom dataset for COCO classification
# (using YOLO format labels) for image classification + bbox regression
class CocoClassificationDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Only keep images that have a corresponding label file
        all_images = sorted(
            [
                f
                for f in os.listdir(images_dir)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        )
        self.image_files = []
        for f in all_images:
            label_path = os.path.join(
                labels_dir, f.replace(".jpg", ".txt").replace(".png", ".txt")
            )
            if os.path.exists(label_path):
                self.image_files.append(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load label file
        label_name = img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        label_path = os.path.join(self.labels_dir, label_name)

        with open(label_path, "r") as f:
            lines = f.readlines()

        # YOLO format: class x_center y_center width height (all normalised 0-1)
        # Take the first object only for this simple single-object model
        parts = lines[0].split()
        class_id = int(parts[0])
        # bbox as [x_center, y_center, width, height], values in [0, 1]
        bbox_raw = [float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])]
        boxes = torch.tensor([bbox_raw], dtype=torch.float32)
        labels = torch.tensor([class_id], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, {"labels": labels, "boxes": boxes}


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

# DataLoader function


def get_dataloaders(
    data_dir="data/coco128_small", batch_size=16, img_size=64, val_split=0.2
):

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    full_dataset = CocoClassificationDataset(
        images_dir=f"{data_dir}/images/train2017",
        labels_dir=f"{data_dir}/labels/train2017",
        transform=transform,
    )

    # Split into train and val randomly
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # add later the test loader too.

    # coco128 is a subset of COCO which has 80 classes, change later
    num_classes = 80
    print(
        f"Dataset size: {len(full_dataset)} images "
        f"({train_size} train / {val_size} val), "
        f"{num_classes} classes"
    )
    return train_loader, val_loader, num_classes
