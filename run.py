from pathlib import Path
from itertools import islice
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from tqdm import tqdm
from dotenv import load_dotenv

from models import BaseDiscriminator, DinoDiscriminator
from datasets import PairedDataset, get_diffface_shards, get_genimage_shards
from datasets.utils import expand_brace_patterns
from analysis.score import BinaryClassificationMeter

load_dotenv()


@dataclass(slots=True, frozen=True)
class DataConfig:
    fake_shards: list[str]
    real_shards: list[str]
    transform: v2.Compose

    num_workers: int
    batch_size: int
    seed: int

    shard_shuffle_size: int = -1
    data_shuffle_size: int = -1


@dataclass
class TrainConfig:
    num_epoch: int
    num_step: int
    log_interval: int

    threshold: float
    device: torch.device
    checkpoint_path: Path


def train(
    writer: SummaryWriter,
    model: BaseDiscriminator,
    train_config: TrainConfig,
    train_data_config: DataConfig,
    eval_data_config: DataConfig,
):
    model = model.to(train_config.device)
    model.train()

    dataset = PairedDataset(
        fake_shards=train_data_config.fake_shards,
        real_shards=train_data_config.real_shards,
        split="train",
        batch_size=train_data_config.batch_size,
        data_shuffle_size=train_data_config.data_shuffle_size,
        shard_shuffle_size=train_data_config.shard_shuffle_size,
        transform=train_data_config.transform,
        seed=train_data_config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        num_workers=train_data_config.num_workers,
    )

    epoch_bar = tqdm(
        range(1, train_config.num_epoch + 1),
        total=train_config.num_epoch,
        desc="Training progress",
        position=0,
        ncols=100,
    )
    for epoch in epoch_bar:
        train_epoch(epoch, dataloader, model, writer, train_data_config, train_config)
        checkpoint = train_config.checkpoint_path / f"epoch-{epoch:03d}.pt"
        model.save(checkpoint)
        evaluate(
            writer, model, checkpoint, epoch, eval_data_config, train_config.device
        )


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
    images: torch.Tensor
    labels: torch.Tensor
    for step, (images, labels) in step_bar:
        images = images.to(train_config.device, non_blocking=True)
        labels = labels.unsqueeze(1).to(
            train_config.device, dtype=torch.float32, non_blocking=True
        )

        logits, loss = model.step(images, labels)

        meter.update_from_logits(logits, labels, train_config.threshold)
        if step % train_config.log_interval != 0:
            continue

        global_step = (epoch - 1) * train_config.num_step + step
        writer.add_scalar("train/loss", loss, global_step)
        meter.log(writer, "train", global_step)

        f1_score, f1_pos, f1_neg = meter.f1_score
        acc = meter.accuracy
        tqdm.write(
            f"Train [epoch {epoch:03d} | step {step:06d}] loss={loss:.4f} acc={acc:.4f} "
            f"f1_score={f1_score:.4f} f1_pos={f1_pos:.4f} f1_neg={f1_neg:.4f}"
        )

        meter.reset()


@torch.no_grad()
def evaluate(
    writer: SummaryWriter,
    model: BaseDiscriminator,
    checkpoint: Path,
    epoch: int,
    data_config: DataConfig,
    device: torch.device,
):
    model = model.to(device)
    model.load(checkpoint, map_location=device)
    model.eval()

    dataset = PairedDataset(
        fake_shards=data_config.fake_shards,
        real_shards=data_config.real_shards,
        split="eval",
        batch_size=data_config.batch_size,
        transform=data_config.transform,
        seed=data_config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        pin_memory=True,
        num_workers=data_config.num_workers,
    )

    meter = BinaryClassificationMeter()
    total_loss = 0.0
    total_n = 0
    step_bar = tqdm(
        dataloader,
        desc="Evaluating",
        ncols=100,
        position=1,
        leave=False,
    )
    images: torch.Tensor
    labels: torch.Tensor
    for images, labels in step_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).to(device, dtype=torch.float32, non_blocking=True)

        logits = model(images)
        loss = model.criterion(logits, labels)
        meter.update_from_logits(logits, labels)

        bs = labels.numel()
        total_loss += float(loss.item()) * bs
        total_n += bs

    meter.log(writer, "eval/metric", epoch)
    f1_score, f1_pos, f1_neg = meter.f1_score
    acc = meter.accuracy
    avg_loss = total_loss / total_n

    tqdm.write(
        f"Eval [step {epoch:06d}] loss={avg_loss:.4f} acc={acc:.4f} "
        f"f1_score={f1_score:.4f} f1_pos={f1_pos:.4f} f1_neg={f1_neg:.4f}"
    )


if __name__ == "__main__":
    # genimage shards (train)
    genimage_datasets = ["adm", "biggan", "glide", "vqdm", "wukong", "sdv5"]
    train_fraction = None
    train_count = None
    train_fake_shards, train_real_shards, _, _ = get_genimage_shards(genimage_datasets)
    train_fake_shards = expand_brace_patterns(
        train_fake_shards,
        fraction=train_fraction,
        count=train_count,
    )
    train_real_shards = expand_brace_patterns(
        train_real_shards,
        fraction=train_fraction,
        count=train_count,
    )

    # diffusion face shards (eval)
    _, eval_real_shards1, eval_fake_shards, eval_real_shards2 = get_diffface_shards()
    eval_fake_shards = expand_brace_patterns(eval_fake_shards)
    eval_fake_shards = eval_fake_shards[:6]
    eval_real_shards = eval_real_shards1 + eval_real_shards2
    eval_real_shards = expand_brace_patterns(eval_real_shards)

    # model and transform
    img_size = 518
    model = DinoDiscriminator(freeze_backbone=True)
    transform = v2.Compose(
        [
            v2.Resize(img_size),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # config
    train_data_config = DataConfig(
        fake_shards=train_fake_shards,
        real_shards=train_real_shards,
        transform=transform,
        num_workers=2,
        batch_size=128,
        shard_shuffle_size=20,
        data_shuffle_size=30_000,
        seed=42,
    )
    train_config = TrainConfig(
        num_epoch=100,
        num_step=500,
        log_interval=10,
        threshold=0.5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_path=Path("./checkpoints/cnn_genimage_pretrain"),
    )
    eval_data_config = DataConfig(
        fake_shards=eval_fake_shards,
        real_shards=eval_real_shards,
        transform=transform,
        num_workers=2,
        batch_size=128,
        seed=42,
    )

    writer = SummaryWriter()
    train(writer, model, train_config, train_data_config, eval_data_config)
    writer.close()
