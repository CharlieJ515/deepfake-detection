from typing import Optional, cast
import os

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
    seed: Optional[int] = 42,
) -> wds.compat.WebDataset:
    shards = expand_brace_patterns(shards)
    transform = transform or v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = wds.compat.WebDataset(shards, shardshuffle=True, repeat=True, seed=seed) \
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
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.diffface_dataset = build_wds(diffface_shards, True, batch_size//2, shuffle_size, transform, seed)
        self.celeba_dataset = build_wds(celeba_shards, False, batch_size//2, shuffle_size, transform, seed)

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



def get_diffface_shards() -> list[str]:
    NC_HOST = os.environ["NC_HOST"]
    NC_USER = os.environ["NC_USER"]
    NC_PASS = os.environ["NC_PASS"]

    BASE_URL = f"https://{NC_USER}:{NC_PASS}@{NC_HOST}/remote.php/dav/files/{NC_USER}"
    prefix = f"{BASE_URL}/data/diffusion_face"

    diffface_shards = [
        f"{prefix}/ADM-{{000..005}}.tar",
        f"{prefix}/DDIM-{{000..005}}.tar",
        f"{prefix}/DDPM-{{000..005}}.tar",
        f"{prefix}/DiffSwap-{{000..005}}.tar",
        f"{prefix}/Inpaint-{{000..005}}.tar",
        f"{prefix}/LDM-{{000..005}}.tar",
        f"{prefix}/PNDM-{{000..005}}.tar",
        f"{prefix}/SDv15_DS0.3-{{000..005}}.tar",
        f"{prefix}/SDv15_DS0.5-{{000..005}}.tar",
        f"{prefix}/SDv15_DS0.7-{{000..005}}.tar",
        f"{prefix}/SDv21_DS0.3-{{000..005}}.tar",
        f"{prefix}/SDv21_DS0.5-{{000..005}}.tar",
        f"{prefix}/SDv21_DS0.7-{{000..005}}.tar",
    ]

    return diffface_shards

def get_celeba_shards() -> list[str]:
    NC_HOST = os.environ["NC_HOST"]
    NC_USER = os.environ["NC_USER"]
    NC_PASS = os.environ["NC_PASS"]

    BASE_URL = f"https://{NC_USER}:{NC_PASS}@{NC_HOST}/remote.php/dav/files/{NC_USER}"
    celeba_prefix = f"{BASE_URL}/data/mm_celeba_hq"
    celeba_shards = [
        f"{celeba_prefix}/mm_celeba_hq-{{000..005}}.tar",
    ]

    return celeba_shards
