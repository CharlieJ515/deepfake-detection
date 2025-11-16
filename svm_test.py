# svm_eval.py

from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import joblib
from tqdm import tqdm

from datasets import PairedDataset
from models import DinoDiscriminator3
from analysis.score import BinaryClassificationMeter  # your eval meter


@dataclass
class EvalConfig:
    fake_shards: list[str]
    real_shards: list[str]

    batch_size: int = 128
    num_workers: int = 1
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    svm_path: Path = Path("svm_classifier.pkl")
    scaler_path: Path = Path("svm_scaler.pkl")


@torch.no_grad()
def extract_features(backbone, images, device):
    images = images.to(device, non_blocking=True)
    feats = backbone(images)  # torch → [B, D]
    feats = feats.cpu().numpy()  # numpy
    return feats


def run_eval(cfg: EvalConfig):
    print("Loading frozen DINO backbone...")
    model = DinoDiscriminator3(freeze_backbone=True).to(cfg.device)
    backbone = model.backbone
    backbone.eval()

    print(f"Loading SVM from {cfg.svm_path}")
    svm = joblib.load(cfg.svm_path)
    scaler = joblib.load(cfg.scaler_path)

    print("Preparing eval dataset...")

    transform = v2.Compose(
        [
            v2.Resize(518),
            v2.CenterCrop(518),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PairedDataset(
        fake_shards=cfg.fake_shards,
        real_shards=cfg.real_shards,
        split="eval",
        batch_size=cfg.batch_size,
        transform=transform,
        seed=42,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    # ------------------------------
    # Evaluation metrics
    # ------------------------------
    meter = BinaryClassificationMeter()
    all_results = []  # [(true, pred, prob)]

    print("Running inference...")
    pbar = tqdm(dataloader, desc="Evaluating", ncols=100)

    for images, labels in pbar:
        labels = labels.numpy()  # [B]

        # ---- extract DINO features ----
        feats = extract_features(backbone, images, cfg.device)  # numpy [B, D]

        # ---- scale features ----
        feats_scaled = scaler.transform(feats)

        # ---- SVM prediction ----
        decision = svm.decision_function(feats_scaled)  # [B], real scores
        probs = 1 / (1 + np.exp(-decision))
        preds = (probs > 0.5).astype(int)

        # ---- update meter ----
        logits_tensor = torch.tensor(decision).unsqueeze(1)
        labels_tensor = torch.tensor(labels).unsqueeze(1).float()
        meter.update_from_logits(logits_tensor, labels_tensor)

        # store results
        for y, p, pr in zip(labels, preds, probs):
            all_results.append((int(y), int(p), float(pr)))

    # ------------------------------
    # Print summary
    # ------------------------------
    f1, f1_pos, f1_neg = meter.f1_score
    print("")
    print("===== SVM EVAL RESULT =====")
    print(f"Accuracy: {meter.accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"F1-pos  : {f1_pos:.4f}")
    print(f"F1-neg  : {f1_neg:.4f}")
    print("============================")

    meter.plot_logit(save_path=Path("./plots/svm_logit.png"))
    meter.plot_prob(save_path=Path("./plots/svm_prob.png"))


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    cfg = EvalConfig(
        real_shards=["./data/eval/celeba_hq/shard-{000000..000001}.tar"],
        fake_shards=["./data/eval/sfhq/shard-{000000..000001}.tar"],
    )

    run_eval(cfg)
