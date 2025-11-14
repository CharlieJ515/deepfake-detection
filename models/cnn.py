import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import BaseDiscriminator


class CNNDiscriminator(BaseDiscriminator):
    def __init__(self, in_channels: int = 3, img_size: int = 224, lr: float = 2e-4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        ds = img_size // (2**5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * ds * ds, 1024),
            nn.Linear(1024, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.features(x)
        logit = self.classifier(x)
        return logit
