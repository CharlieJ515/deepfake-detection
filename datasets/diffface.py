from typing import Optional, cast

import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import v2
import webdataset as wds

from .utils import expand_brace_patterns


def build_wds(
    shards: list[str],
    label: bool = True,
    batch_size: int = 128,
    shuffle_size: int = 10_000,
    transform: Optional[v2.Compose] = None,
) -> wds.compat.WebDataset:
    shards = expand_brace_patterns(shards)
    transform = transform or v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
        
    # pipeline = wds.pipeline.DataPipeline(
    #     wds.shardlists.ResampledShards(shards),
    #     wds.shardlists.split_by_worker,
    #     wds.tariterators.tarfile_to_samples(handler=wds.handlers.reraise_exception),
    #     wds.filters.shuffle(shuffle_size),
    #     wds.filters.decode("pil"),
    #     wds.filters.map(lambda x: {"images":transform(x["png"]), "labels":torch.tensor(label, dtype=torch.bool)}),
    #     wds.filters.batched(batch_size, partial=True)
    # # )
    # return pipeline

    dataset = wds.compat.WebDataset(shards, shardshuffle=False) \
          .shuffle(shuffle_size) \
          .decode("pil") \
          .map(lambda x: {"images": transform(x.get("png") or x.get("jpg")),
                          "labels": torch.tensor(label, dtype=torch.bool)}) \
          .batched(batch_size, partial=True)
    return cast(wds.compat.WebDataset, dataset)


class DiffusionFaceDataset(IterableDataset):
    def __init__(
        self,
        diffface_shards: list[str],
        celeba_shards: list[str],
        shuffle_size: int,
        batch_size: int,
        transform: Optional[v2.Compose] = None,
        concat_fn=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.diffface_dataset = build_wds(diffface_shards, True, batch_size//2, shuffle_size, transform)
        self.celeba_dataset = build_wds(celeba_shards, False, batch_size//2, shuffle_size, transform)

        self.concat_fn = concat_fn or self.default_concat

    def default_concat(self, a, b):
        images = torch.cat([a['images'], b['images']], 0)
        labels = torch.cat([a['labels'], b['labels']], 0)

        return images, labels

    def __iter__(self):
        diff_iter = iter(self.diffface_dataset)
        celeba_iter = iter(self.celeba_dataset)

        for a, b in zip(diff_iter, celeba_iter):
            yield self.concat_fn(a, b)


