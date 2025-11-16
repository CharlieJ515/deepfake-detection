from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from .base import BaseDiscriminator


# https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/blame/4476dc0c66daca2ef4a40d2625b4a7063f02b685/config.json
class DinoDiscriminator3(BaseDiscriminator):
    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        lr: float = 2e-4,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        embed_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            [
                {"params": self.backbone.parameters(), "lr": lr * 0.1},
                {"params": self.classifier.parameters(), "lr": lr},
            ]
        )

    def forward(self, x) -> torch.Tensor:
        emb = self.backbone(x)
        logit = self.classifier(emb)
        return logit
