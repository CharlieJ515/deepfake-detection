from .utils.nextcloud import get_base_url


def get_ffhq_train(base_url: str | None = None):
    base_url = base_url or get_base_url()

    ffhq_prefix = f"{base_url}/FFHQ"
    shards = [
        f"{ffhq_prefix}/FFHQ_1/FFHQ_1-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_2/FFHQ_2-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_3/FFHQ_3-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_4/FFHQ_4-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_5/FFHQ_5-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_6/FFHQ_6-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_7/FFHQ_7-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_8/FFHQ_8-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_9/FFHQ_9-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_10/FFHQ_10-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_11/FFHQ_11-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_12/FFHQ_12-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_13/FFHQ_13-{{000000..000048}}.tar",
        f"{ffhq_prefix}/FFHQ_14/FFHQ_14-{{000000..000048}}.tar",
    ]

    return shards


def get_ffhq_eval(base_url: str | None = None):
    base_url = base_url or get_base_url()

    ffhq_prefix = f"{base_url}/FFHQ"
    shards = [
        f"{ffhq_prefix}/FFHQ_1/FFHQ_1-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_2/FFHQ_2-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_3/FFHQ_3-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_4/FFHQ_4-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_5/FFHQ_5-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_6/FFHQ_6-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_7/FFHQ_7-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_8/FFHQ_8-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_9/FFHQ_9-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_10/FFHQ_10-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_11/FFHQ_11-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_12/FFHQ_12-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_13/FFHQ_13-{{000049..000049}}.tar",
        f"{ffhq_prefix}/FFHQ_14/FFHQ_14-{{000049..000049}}.tar",
    ]

    return shards
