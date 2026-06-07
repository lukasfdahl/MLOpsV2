import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


# dataset classes
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# function to denormalise image tensor for inspection
def _denormalise_image(tensor):
    img = tensor.cpu().clone()
    img = img * 0.5 + 0.5
    img = img.permute(1, 2, 0)
    return img.numpy().clip(0, 1)


# convert yolo bbox to pixel
def _yolo_to_xyxy(bbox, img_w, img_h):
    single = bbox.dim() == 1
    if single:
        bbox = bbox.unsqueeze(0)

    cx, cy, bw, bh = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h

    # save result and return
    result = torch.stack([x1, y1, x2, y2], dim=1)
    return result.squeeze(0) if single else result


# inspect predictions
def show_predictions(
    model, val_loader, device, num_examples=8, save_path="custom_model/predictions.png"
):
    model.eval()
    collected = {
        "images": [],
        "gt_cls": [],
        "gt_bbox": [],
        "pred_cls": [],
        "pred_bbox": [],
    }

    num_classes = model.num_classes

    # disable gradient calculation
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            pred_logits, pred_boxes = model(images)  # [B,K,C+1], [B,K,4]

            for i in range(images.size(0)):
                t = targets[i]
                gt_label = t["labels"][0].item() if len(t["labels"]) > 0 else 0
                gt_box = t["boxes"][0] if len(
                    t["boxes"]) > 0 else torch.zeros(4)

                collected["images"].append(images[i].cpu())
                collected["gt_cls"].append(gt_label)
                collected["gt_bbox"].append(gt_box.cpu())

                logits_i = pred_logits[i]
                real_scores = logits_i[:, :num_classes]
                best_query = real_scores.max(dim=1).values.argmax().item()
                collected["pred_cls"].append(
                    real_scores[best_query].argmax().item())
                collected["pred_bbox"].append(pred_boxes[i, best_query].cpu())

                if len(collected["images"]) >= num_examples:
                    break
            if len(collected["images"]) >= num_examples:
                break

    # plot predictions
    n = len(collected["images"])
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    img_size = collected["images"][0].shape[-1]  # assumes square

    # Plot each image
    for i in range(n):
        ax = axes[i]
        img_np = _denormalise_image(collected["images"][i])
        ax.imshow(img_np)

        # Ground-truth box (pink)
        gt_box = _yolo_to_xyxy(collected["gt_bbox"][i], img_size, img_size)
        x1, y1, x2, y2 = gt_box.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="pink",
            facecolor="none",
            label="GT",
        )
        ax.add_patch(rect)

        # Predicted box (blue)
        pred_box = _yolo_to_xyxy(collected["pred_bbox"][i], img_size, img_size)
        px1, py1, px2, py2 = pred_box.tolist()
        rect_pred = patches.Rectangle(
            (px1, py1),
            px2 - px1,
            py2 - py1,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
            linestyle="--",
            label="Pred",
        )
        ax.add_patch(rect_pred)

        gt_name = (
            COCO_CLASSES[collected["gt_cls"][i]]
            if collected["gt_cls"][i] < len(COCO_CLASSES)
            else str(collected["gt_cls"][i])
        )
        pred_name = (
            COCO_CLASSES[collected["pred_cls"][i]]
            if collected["pred_cls"][i] < len(COCO_CLASSES)
            else str(collected["pred_cls"][i])
        )
        ax.set_title(f"GT: {gt_name}\nPred: {pred_name}", fontsize=9)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="pink", linewidth=2, label="Ground truth"),
        Line2D([0], [0], color="blue", linewidth=2,
               linestyle="--", label="Predicted"),
    ]
    fig.legend(handles=legend_elements,
               loc="lower center", ncol=2, fontsize=10)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction examples saved to {save_path}")
