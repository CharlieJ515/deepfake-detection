from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from tqdm import tqdm
from dotenv import load_dotenv

from models.cnn_discriminator import Discriminator
from datasets.wds import PairedDataset
from datasets.diff_face import get_diffface_shards
from datasets.gen_image import get_genimage_shards
from datasets.utils.brace_expand import expand_brace_patterns
from analysis.f1_score import compute_macro_f1

load_dotenv()

seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chkpt_path = Path("./checkpoints")

genimage_datasets = ["adm", "biggan", "glide", "vqdm", "wukong"]
eval_fraction = None
eval_count = None
_, _, ai_shards, real_shards = get_genimage_shards(genimage_datasets)
# _, real_shards1, ai_shards, real_shards2 = get_diffface_shards()
# real_shards = real_shards1 + real_shards2
ai_shards = expand_brace_patterns(
    ai_shards,
    fraction=eval_fraction,
    count=eval_count,
)
# ai_shards = ai_shards[:6]
real_shards = expand_brace_patterns(
    real_shards,
    fraction=eval_fraction,
    count=eval_count,
)

batch_size = 128
num_workers = 2

img_size = 256
transform = v2.Compose(
    [
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


@torch.no_grad()
def evaluate(checkpoint: Path):
    model = Discriminator(img_size=img_size).to(device)
    model.load(checkpoint, map_location=device)
    model.eval()

    dataset = PairedDataset(
        ai_shards=ai_shards,
        real_shards=real_shards,
        split="eval",
        batch_size=batch_size,
        transform=transform,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset, batch_size=None, pin_memory=True, num_workers=num_workers
    )

    # Tracking
    TP = TN = FP = FN = 0
    total_loss = 0.0
    total_n = 0

    for images, labels in tqdm(dataloader, desc="Evaluating", ncols=100):
        images = images.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

        logits = model(images)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits.view(-1))
        preds = (probs > 0.5).long()
        y_true = labels.view(-1).long()

        TP += int(((preds == 1) & (y_true == 1)).sum())
        TN += int(((preds == 0) & (y_true == 0)).sum())
        FP += int(((preds == 1) & (y_true == 0)).sum())
        FN += int(((preds == 0) & (y_true == 1)).sum())

        bs = y_true.numel()
        total_loss += float(loss.item()) * bs
        total_n += bs

    denom_pos = 2 * TP + FP + FN
    denom_neg = 2 * TN + FP + FN
    F1_pos = (2 * TP / denom_pos) if denom_pos > 0 else 0.0
    F1_neg = (2 * TN / denom_neg) if denom_neg > 0 else 0.0
    macro_f1 = (F1_pos + F1_neg) / 2.0

    acc = (TP + TN) / max(1, (TP + TN + FP + FN))
    avg_loss = total_loss / max(1, total_n)

    print(
        f"[Eval] loss={avg_loss:.4f}  acc={acc:.4f}  F1_pos={F1_pos:.4f}  F1_neg={F1_neg:.4f}  macro_f1={macro_f1:.4f}"
    )


if __name__ == "__main__":
    checkpoint = chkpt_path / "epoch-001.pt"
    print(checkpoint)
    evaluate(checkpoint)
