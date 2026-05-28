import torch
import torch.nn as nn


class CustomCNN(nn.Module):

    def __init__(self, num_classes: int, num_queries: int = 50):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 64 → 32

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 32 → 16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 16 → 8
        )

        # Shared FC trunk
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Query embeddings (one per detection slot)
        self.query_embed = nn.Embedding(num_queries, 256)

        # Classification head
        # +1 for the "no-object" / background class
        self.cls_head = nn.Linear(256, num_classes + 1)

        # Bounding-box regression head
        # Sigmoid keeps output in (0, 1) matching normalised YOLO coords
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    # Forward pass takes (B, 3, 64, 64) images and outputs
    def forward(self, x: torch.Tensor):

        b = x.size(0)
        feat = self.trunk(self.features(x))          # (B, 256)

        # Broadcast feature vector across all query slots
        q = self.query_embed.weight.unsqueeze(
            0).expand(b, -1, -1)  # (B, Q, 256)
        # (B, Q, 256)
        h = feat.unsqueeze(1) + q

        pred_logits = self.cls_head(h)   # (B, Q, num_classes+1)
        pred_boxes = self.bbox_head(h)   # (B, Q, 4)

        return pred_logits, pred_boxes
