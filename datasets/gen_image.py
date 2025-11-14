from typing import Iterable
from .utils.nextcloud import get_base_url

_DATASET_RANGES: dict[str, tuple[int, int, int, int]] = {
    "adm": (80, 78, 2, 2),
    "biggan": (80, 80, 2, 2),
    "glide": (80, 80, 2, 2),
    "sdv5": (82, 74, 3, 3),
    "vqdm": (80, 80, 2, 2),
    "wukong": (80, 80, 2, 2),
}


def get_genimage_train(datasets: Iterable[str], base_url: str | None = None):
    base_url = base_url or get_base_url()

    fake_shards: list[str] = []
    real_shards: list[str] = []
    for ds in datasets:
        ds = ds.lower()
        if ds not in _DATASET_RANGES:
            raise ValueError(
                f"Unknown dataset '{ds}'. Valid: {sorted(_DATASET_RANGES)}"
            )

        fake_end, reak_end, _, _ = _DATASET_RANGES[ds]
        prefix = f"{base_url}/gen_image/imagenet_{ds}"

        fake_shard = f"{prefix}/train/ai_shards/shard-{{000000..{fake_end:06d}}}.tar.gz"
        real_shard = (
            f"{prefix}/train/nature_shards/shard-{{000000..{reak_end:06d}}}.tar.gz"
        )

        fake_shards.append(fake_shard)
        real_shards.append(real_shard)

    return fake_shards, real_shards


def get_genimage_eval(datasets: Iterable[str], base_url: str | None = None):
    base_url = base_url or get_base_url()

    fake_shards: list[str] = []
    real_shards: list[str] = []
    for ds in datasets:
        ds = ds.lower()
        if ds not in _DATASET_RANGES:
            raise ValueError(
                f"Unknown dataset '{ds}'. Valid: {sorted(_DATASET_RANGES)}"
            )

        _, _, fake_end, real_end = _DATASET_RANGES[ds]
        prefix = f"{base_url}/gen_image/imagenet_{ds}"

        fake_shard = f"{prefix}/val/ai_shards/shard-{{000000..{fake_end:06d}}}.tar.gz"
        real_shard = (
            f"{prefix}/val/nature_shards/shard-{{000000..{real_end:06d}}}.tar.gz"
        )

        fake_shards.append(fake_shard)
        real_shards.append(real_shard)

    return fake_shards, real_shards
