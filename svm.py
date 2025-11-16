# svm_train.py

from pathlib import Path
from dataclasses import dataclass
from itertools import islice

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import numpy as np
from tqdm import tqdm
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from datasets import PairedDataset, get_sfhq_train, get_ffhq_train
from models import DinoDiscriminator3  # backbone = model.backbone


# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
@dataclass
class SVMConfig:
    batch_size: int = 128
    num_workers: int = 4

    # How many batches to load from the WDS pipeline
    num_steps: int = 200  # increase this to train on more data

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output paths
    svm_output: Path = Path("svm_classifier.pkl")
    scaler_output: Path = Path("svm_scaler.pkl")


# -------------------------------------------------------------
# Extract DINO embeddings
# -------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(backbone, dataloader, num_steps, device):
    backbone.eval()

    X_list = []
    y_list = []

    step_iter = islice(dataloader, num_steps)
    pbar = tqdm(step_iter, total=num_steps, desc="Extracting embeddings")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)

        feats = backbone(images)  # [B, 384] (or 318 depending on model)
        feats = feats.cpu().numpy()  # convert to numpy
        labels = labels.cpu().numpy()

        X_list.append(feats)
        y_list.append(labels)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"[Done] Extracted features: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


# -------------------------------------------------------------
# Train SVM
# -------------------------------------------------------------
def train_svm(X, y, svm_path, scaler_path):
    print("Standardizing features ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Linear SVM ...")
    svm = LinearSVC(C=1.0, max_iter=10000)
    svm.fit(X_scaled, y)

    print(f"Saving SVM → {svm_path}")
    joblib.dump(svm, svm_path)

    print(f"Saving scaler → {scaler_path}")
    joblib.dump(scaler, scaler_path)

    print("Training complete.")
    return svm, scaler


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    cfg = SVMConfig()

    # ---------------------------------------------------------
    # 1. Load model (only backbone is used)
    # ---------------------------------------------------------
    print("Loading DINO backbone...")
    model = DinoDiscriminator3(freeze_backbone=True)  # or DINO2 etc.
    model = model.to(cfg.device)
    backbone = model.backbone  # timm DINO model
    backbone.eval()

    # ---------------------------------------------------------
    # 2. Prepare training dataset
    # ---------------------------------------------------------
    print("Building dataset...")
    fake_shards = get_sfhq_train("./data")
    real_shards = get_ffhq_train("./data")

    transform = v2.Compose(
        [
            v2.RandomResizedCrop(518, scale=(0.6, 1.0), ratio=(1, 1)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PairedDataset(
        fake_shards=fake_shards,
        real_shards=real_shards,
        split="train",
        batch_size=cfg.batch_size,
        transform=transform,
        seed=42,
        shard_shuffle_size=20,
        data_shuffle_size=20000,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------
    # 3. Extract embeddings
    # ---------------------------------------------------------
    X, y = extract_embeddings(
        backbone=backbone,
        dataloader=dataloader,
        num_steps=cfg.num_steps,
        device=cfg.device,
    )

    # ---------------------------------------------------------
    # 4. Train & save SVM
    # ---------------------------------------------------------
    train_svm(X, y, cfg.svm_output, cfg.scaler_output)
