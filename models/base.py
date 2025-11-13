from abc import ABC, abstractmethod
from typing import Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDiscriminator(nn.Module, ABC):
    optimizer: torch.optim.Optimizer
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x) -> torch.Tensor: ...

    def step(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        self.optimizer.zero_grad()
        preds = self.forward(images)
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()
        return preds, loss.item()

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).long()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path: Path, map_location: torch.device = torch.device("cpu")):
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
