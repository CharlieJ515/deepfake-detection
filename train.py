from pathlib import Path
from itertools import islice

import torch
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
chkpt_path = Path("./checkpoints/attempt2")

### Train ###
lr = 2e-4

genimage_datasets = ["adm", "biggan", "glide", "vqdm", "wukong"]
data_shuffle_size = 10000
shard_shuffle_size = 20
train_fraction = None
train_count = None
ai_shards, real_shards, _, _ = get_genimage_shards(genimage_datasets)
ai_shards = expand_brace_patterns(
    ai_shards,
    fraction=train_fraction,
    count=train_count,
)
real_shards = expand_brace_patterns(
    real_shards,
    fraction=train_fraction,
    count=train_count,
)


batch_size = 128
num_workers = 2

img_size = 256
transform = v2.Compose(
    [
        v2.RandomCrop(img_size, pad_if_needed=True, padding_mode="reflect"),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

num_epoch = 10
num_step: int = 10_000
log_interval = 10

### Eval ###
eval_fraction = None
eval_count = 1


def train(writer: SummaryWriter):
    model = Discriminator(img_size=img_size, lr=lr).to(device)

    dataset = PairedDataset(
        ai_shards=ai_shards,
        real_shards=real_shards,
        split="train",
        batch_size=batch_size,
        data_shuffle_size=data_shuffle_size,
        shard_shuffle_size=shard_shuffle_size,
        transform=transform,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset, batch_size=None, pin_memory=True, num_workers=num_workers
    )

    model.train()

    epoch_bar = tqdm(total=num_epoch, desc="Training progress", position=0, ncols=100)
    for epoch in range(1, num_epoch + 1):
        train_epoch(epoch, dataloader, model, writer)
        model.save(chkpt_path / f"epoch-{epoch:03d}.pt")
        epoch_bar.update(1)

    epoch_bar.close()


def train_epoch(
    epoch: int, dataloader: DataLoader, model: Discriminator, writer: SummaryWriter
):
    step_bar = tqdm(
        total=num_step,
        desc=f"Epoch {epoch}/{num_epoch}",
        position=1,
        leave=False,
        ncols=100,
    )

    for step, (images, labels) in enumerate(islice(dataloader, num_step), 1):
        images = images.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

        logits, loss = model.step(images, labels)

        if step % log_interval != 0:
            step_bar.update(1)
            continue

        with torch.no_grad():
            probs = torch.sigmoid(logits.view(-1))
            preds = (probs > 0.5).long()
            acc = (preds == labels.view(-1).long()).float().mean().item()
            macro_f1 = compute_macro_f1(preds, labels)

        global_step = (epoch - 1) * num_step + step
        writer.add_scalar("train/loss", loss, global_step)
        writer.add_scalar("train/acc", acc, global_step)
        writer.add_scalar("train/macro_f1", macro_f1, global_step)

        step_bar.write(
            f"[epoch {epoch:03d} | step {step:06d}] loss={loss:.4f}  acc={acc:.4f}  macro_f1={macro_f1:.4f}"
        )
        step_bar.update(1)
    step_bar.close()


if __name__ == "__main__":
    writer = SummaryWriter()
    train(writer)
    writer.close()
