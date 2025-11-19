from .utils.nextcloud import get_base_url


def get_generated_train(base_url: str | None = None):
    base_url = base_url or get_base_url()

    generated_prefix = f"{base_url}/generated"
    shards = [
        f"{generated_prefix}/shard-{{000000..000032}}.tar",
    ]

    return shards


def get_generated_eval(base_url: str | None = None):
    base_url = base_url or get_base_url()

    generated_prefix = f"{base_url}/generated"
    shards = [
        f"{generated_prefix}/shard-{{000033..000033}}.tar",
    ]

    return shards
