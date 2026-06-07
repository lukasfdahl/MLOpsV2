import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights


class CustomCNN(nn.Module):

    def __init__(self, num_classes: int, num_queries: int = 50):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        # ResNet-101 backbone for feature extraction (~42M params in backbone)
        # Pretrained on ImageNet for better feature representations
        backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

        # Remove the final avgpool and fc layers — we only want the feature maps
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # Adaptive pool to collapse spatial dims to 1x1 — gives a 2048-dim vector
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared FC trunk — 2048 channels from ResNet-101 layer4
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
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

        # Extract features with ResNet-101 backbone
        feat_map = self.features(x)              # (B, 2048, H', W')
        feat_map = self.adaptive_pool(feat_map)  # (B, 2048, 1, 1)
        feat = self.trunk(feat_map)              # (B, 256)

        # Broadcast feature vector across all query slots
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # (B, Q, 256)
        h = feat.unsqueeze(1) + q                                    # (B, Q, 256)

        pred_logits = self.cls_head(h)   # (B, Q, num_classes+1)
        pred_boxes = self.bbox_head(h)   # (B, Q, 4)

        return pred_logits, pred_boxes