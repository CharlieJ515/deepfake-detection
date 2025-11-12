from typing import Iterable
from .utils.nextcloud import get_base_url

_DATASET_RANGES: dict[str, tuple[int, int, int, int]] = {
    "adm": (80, 78, 2, 2),
    "biggan": (80, 80, 2, 2),
    "glide": (80, 80, 2, 2),
    "vqdm": (80, 80, 2, 2),
    "wukong": (80, 80, 2, 2),
}


def get_genimage_shards(datasets: Iterable[str]):
    base_url = get_base_url()

    train_ai: list[str] = []
    train_real: list[str] = []
    val_ai: list[str] = []
    val_real: list[str] = []

    for ds in datasets:
        ds = ds.lower()
        if ds not in _DATASET_RANGES:
            raise ValueError(
                f"Unknown dataset '{ds}'. Valid: {sorted(_DATASET_RANGES)}"
            )

        tr_ai_end, tr_real_end, v_ai_end, v_real_end = _DATASET_RANGES[ds]
        prefix = f"{base_url}/data/gen_image/imagenet_{ds}"

        ai_train_shard = (
            f"{prefix}/train/ai_shards/shard-{{000000..{tr_ai_end:06d}}}.tar.gz"
        )
        real_train_shard = (
            f"{prefix}/train/nature_shards/shard-{{000000..{tr_real_end:06d}}}.tar.gz"
        )
        ai_val_shard = f"{prefix}/val/ai_shards/shard-{{000000..{v_ai_end:06d}}}.tar.gz"
        real_val_shard = (
            f"{prefix}/val/nature_shards/shard-{{000000..{v_real_end:06d}}}.tar.gz"
        )

        train_ai.append(ai_train_shard)
        train_real.append(real_train_shard)
        val_ai.append(ai_val_shard)
        val_real.append(real_val_shard)

    return train_ai, train_real, val_ai, val_real
