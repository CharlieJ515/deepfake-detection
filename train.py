from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from models.cnn_discriminator import Discriminator
from datasets.diffface import DiffusionFaceDataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Discriminator(img_size=256)
    model = model.to(device)

    diffface_shards = [
        "./data/diffusion_face/ADM-{000..005}.tar",
        "./data/diffusion_face/DDIM-{000..005}.tar",
        "./data/diffusion_face/DDPM-{000..005}.tar",
        "./data/diffusion_face/DiffSwap-{000..005}.tar",
        "./data/diffusion_face/Inpaint-{000..005}.tar",
        "./data/diffusion_face/LDM-{000..005}.tar",
        "./data/diffusion_face/PNDM-{000..005}.tar",
        "./data/diffusion_face/SDv15_DS0.3-{000..005}.tar",
        "./data/diffusion_face/SDv15_DS0.5-{000..005}.tar",
        "./data/diffusion_face/SDv15_DS0.7-{000..005}.tar",
        "./data/diffusion_face/SDv21_DS0.3-{000..005}.tar",
        "./data/diffusion_face/SDv21_DS0.5-{000..005}.tar",
        "./data/diffusion_face/SDv21_DS0.7-{000..005}.tar",
    ]
    celeba_shards = [
        "./data/mm_celeba_hq/mm_celeba_hq-{000..005}.tar"
    ]
    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])
    dataset = DiffusionFaceDataset(
        diffface_shards,
        celeba_shards,
        3000,
        256,
        transform,
    )
    dataloader = DataLoader(dataset, None, )

    model.train()
    step = 0
    num_epoch = 10
    log_interval = 100
    start_time = time.time()
    for epoch in range(1, num_epoch+1):
        running_loss = 0.0
        correct = 0
        total = 0

        images: torch.Tensor
        labels: torch.Tensor
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

            logits, loss = model.step(images, labels)

            preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()
            correct += (preds == labels).sum().item()

            running_loss += loss * images.size(0)
            total += images.size(0)
            step += 1

            if step % log_interval == 0:
                avg_loss = running_loss / total
                acc = correct / total
                elapsed = time.time() - start_time
                print(f"[epoch {epoch}/{num_epoch} | step {step}] "
                      f"loss={avg_loss:.4f} acc={acc:.4f} elapsed={elapsed:.1f}s")

        # end of epoch
        avg_loss = running_loss / total
        acc = correct / total
        print(f"Epoch {epoch} done: loss={avg_loss:.4f} acc={acc:.4f}")


if __name__ == "__main__":
    train()