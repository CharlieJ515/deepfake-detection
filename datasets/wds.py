from typing import Optional, cast, Literal, Callable, Any
from functools import partial

import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import v2
import webdataset as wds

from .utils.brace_expand import expand_brace_patterns


def _sample_map(x, label: bool, transform: v2.Compose):
    img = x.get("png") or x.get("jpg") or x.get("jpeg")
    return {
        "images": transform(img),
        "labels": torch.tensor(label, dtype=torch.bool),
    }


def _filter_None(x):
    if x["images"] is None:
        return False
    return True


def build_wds(
    shards: list[str],
    *,
    label: bool = True,
    split: Literal["train", "eval"] = "train",
    batch_size: int = 128,
    data_shuffle_size: int = 10_000,
    shard_shuffle_size: int = 1000,
    transform: Optional[v2.Compose] = None,
    seed: Optional[int] = 42,
) -> wds.compat.WebDataset:
    files = expand_brace_patterns(shards)
    transform = transform or v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    mapper = partial(_sample_map, label=label, transform=transform)

    if split == "eval":
        ds = (
            wds.compat.WebDataset(files, seed=seed)
            .decode("pil")
            .map(mapper)
            .select(_filter_None)
            .batched(batch_size, partial=True)
        )
        return cast(wds.compat.WebDataset, ds)

    ds = (
        wds.compat.WebDataset(
            files,
            shardshuffle=shard_shuffle_size,
            repeat=True,
            seed=seed,
        )
        .shuffle(data_shuffle_size)
        .decode("pil")
        .map(mapper)
        .select(_filter_None)
        .batched(batch_size, partial=True)
    )
    return cast(wds.compat.WebDataset, ds)


class PairedDataset(IterableDataset):
    def __init__(
        self,
        *,
        fake_shards: list[str],
        real_shards: list[str],
        split: Literal["train", "eval"] = "train",
        batch_size: int = 128,
        data_shuffle_size: int = 10_000,
        shard_shuffle_size: int = 1000,
        transform: Optional[v2.Compose] = None,
        concat_fn: Optional[Callable[[dict[str, Any], dict[str, Any]], Any]] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = split
        self.concat_fn = concat_fn or self.default_concat

        half = batch_size // 2
        self.fake_dataset = build_wds(
            fake_shards,
            label=True,
            split=split,
            batch_size=half,
            data_shuffle_size=data_shuffle_size,
            shard_shuffle_size=shard_shuffle_size,
            transform=transform,
            seed=seed,
        )
        self.real_dataset = build_wds(
            real_shards,
            label=False,
            split=split,
            batch_size=half,
            data_shuffle_size=data_shuffle_size,
            shard_shuffle_size=shard_shuffle_size,
            transform=transform,
            seed=seed,
        )

    @staticmethod
    def default_concat(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]):
        images = torch.cat([a["images"], b["images"]], dim=0)
        labels = torch.cat([a["labels"], b["labels"]], dim=0)
        return images, labels

    def __iter__(self):
        fake_iter = iter(self.fake_dataset)
        real_iter = iter(self.real_dataset)

        for fake_batch, real_batch in zip(fake_iter, real_iter):
            yield self.concat_fn(fake_batch, real_batch)
