from pathlib import Path
from itertools import islice
from dataclasses import dataclass
from typing import Callable, Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from tqdm import tqdm
from dotenv import load_dotenv

from models.base import BaseDiscriminator
from datasets.wds import PairedDataset
from datasets.diff_face import get_diffface_shards
from datasets.gen_image import get_genimage_shards
from datasets.utils.brace_expand import expand_brace_patterns
from analysis.score import BinaryClassificationMeter

load_dotenv()


@dataclass
class DataConfig:
    fake_shards: list[str]
    real_shards: list[str]
    transform: v2.Compose

    seed: int = 42

    num_workers: int = 2
    batch_size: int = 128


@dataclass
class TrainConfig:
    num_epoch: int
    num_step: int
    log_interval: int

    threshold: float = 0.5
    device: torch.device = torch.device("cpu")
    checkpoint_path: Path = Path("./checkpoints")


def train(
    writer: SummaryWriter,
    model: BaseDiscriminator,
    data_config: DataConfig,
    train_config: TrainConfig,
):
    model = model.to(train_config.device)
    model.train()

    dataset = PairedDataset(
        fake_shards=data_config.fake_shards,
        real_shards=data_config.real_shards,
        split="train",
        batch_size=data_config.batch_size,
        data_shuffle_size=data_config.data_shuffle_size,
        shard_shuffle_size=data_config.shard_shuffle_size,
        transform=data_config.transform,
        seed=data_config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        num_workers=data_config.num_workers,
    )

    epoch_bar = tqdm(
        range(1, train_config.num_epoch + 1),
        total=train_config.num_epoch,
        desc="Training progress",
        position=0,
        ncols=100,
    )
    for epoch in epoch_bar:
        train_epoch(epoch, dataloader, model, writer, data_config, train_config)
        model.save(train_config.checkpoint_path / f"epoch-{epoch:03d}.pt")


def train_epoch(
    epoch: int,
    dataloader: DataLoader,
    model: BaseDiscriminator,
    writer: SummaryWriter,
    data_config: DataConfig,
    train_config: TrainConfig,
):
    model.train()
    meter = BinaryClassificationMeter()
    step_bar = tqdm(
        enumerate(islice(dataloader, train_config.num_step), 1),
        total=train_config.num_step,
        desc=f"Epoch {epoch}/{train_config.num_epoch}",
        position=1,
        leave=False,
        ncols=100,
    )
    for step, (images, labels) in step_bar:
        images = images.to(train_config.device, non_blocking=True)
        labels = labels.unsqueeze(1).to(
            train_config.device, dtype=torch.float32, non_blocking=True
        )

        logits, loss = model.step(images, labels)

        meter.update_from_logits(logits, labels, train_config.threshold)
        if step % train_config.log_interval != 0:
            continue

        acc = meter.accuracy
        macro_f1 = meter.f1_score

        global_step = (epoch - 1) * train_config.num_step + step
        writer.add_scalar("train/loss", loss, global_step)
        writer.add_scalar("train/acc", acc, global_step)
        writer.add_scalar("train/macro_f1", macro_f1, global_step)

        step_bar.write(
            f"[epoch {epoch:03d} | step {step:06d}] loss={loss:.4f} "
            "acc={acc:.4f}  macro_f1={macro_f1:.4f}"
        )
        meter.reset()


if __name__ == "__main__":

    genimage_datasets = ["adm", "biggan", "glide", "vqdm", "wukong"]
    data_shuffle_size = 10000
    shard_shuffle_size = 20
    train_fraction = None
    train_count = None
    fake_shards, real_shards, _, _ = get_genimage_shards(genimage_datasets)
    fake_shards = expand_brace_patterns(
        fake_shards,
        fraction=train_fraction,
        count=train_count,
    )
    real_shards = expand_brace_patterns(
        real_shards,
        fraction=train_fraction,
        count=train_count,
    )

    transform = v2.Compose(
        [
            v2.RandomCrop(256, pad_if_needed=True, padding_mode="reflect"),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    data_config = DataConfig(
        fake_shards=fake_shards,
        real_shards=real_shards,
        transform=transform,
        seed=42,
        num_workers=2,
        batch_size=128,
    )
    train_config = TrainConfig(
        num_epoch=100,
        num_step=1000,
        log_interval=10,
        threshold=0.5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_path=Path("./checkpoints/attempt2"),
    )

    writer = SummaryWriter()
    train(writer, model, data_config, train_config)
    writer.close()
