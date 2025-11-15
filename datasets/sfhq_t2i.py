from .utils.nextcloud import get_base_url


def get_sfhq_t2i_train(base_url: str | None = None):
    base_url = base_url or get_base_url()

    sfhq_prefix = f"{base_url}/SFHQ-T2I"
    shards = [
        f"{sfhq_prefix}/SDXL/shard-{{000000..000051}}.tar",
        f"{sfhq_prefix}/FLUX1_schnell/shard-{{000000..000056}}.tar",
    ]

    return shards


def get_sfhq_t2i_eval(base_url: str | None = None):
    base_url = base_url or get_base_url()

    sfhq_prefix = f"{base_url}/SFHQ-T2I"
    shards = [
        f"{sfhq_prefix}/SDXL/shard-{{000052..000052}}.tar",
        f"{sfhq_prefix}/FLUX1_schnell/shard-{{000057..000057}}.tar",
    ]

    return shards
