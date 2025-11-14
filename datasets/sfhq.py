from .utils.nextcloud import get_base_url


def get_sfhq_train(base_url: str | None = None):
    base_url = base_url or get_base_url()

    sfhq_prefix = f"{base_url}/SFHQ"
    shards = [
        f"{sfhq_prefix}/SFHQ_1A/SFHQ_1A-{{000000..000222}}.tar",
        f"{sfhq_prefix}/SFHQ_1B/SFHQ_1B-{{000000..000222}}.tar",
        f"{sfhq_prefix}/SFHQ_1C/SFHQ_1C-{{000000..000222}}.tar",
        f"{sfhq_prefix}/SFHQ_1D/SFHQ_1D-{{000000..000222}}.tar",
        f"{sfhq_prefix}/SFHQ_2A/SFHQ_2A-{{000000..000226}}.tar",
        f"{sfhq_prefix}/SFHQ_2B/SFHQ_2B-{{000000..000226}}.tar",
        f"{sfhq_prefix}/SFHQ_2C/SFHQ_2C-{{000000..000226}}.tar",
        f"{sfhq_prefix}/SFHQ_2D/SFHQ_2D-{{000000..000226}}.tar",
        f"{sfhq_prefix}/SFHQ_3A/SFHQ_3A-{{000000..000293}}.tar",
        f"{sfhq_prefix}/SFHQ_3B/SFHQ_3B-{{000000..000293}}.tar",
        f"{sfhq_prefix}/SFHQ_3C/SFHQ_3C-{{000000..000293}}.tar",
        f"{sfhq_prefix}/SFHQ_3D/SFHQ_3D-{{000000..000293}}.tar",
        f"{sfhq_prefix}/SFHQ_4A/SFHQ_4A-{{000000..000312}}.tar",
        f"{sfhq_prefix}/SFHQ_4B/SFHQ_4B-{{000000..000312}}.tar",
        f"{sfhq_prefix}/SFHQ_4C/SFHQ_4C-{{000000..000312}}.tar",
        f"{sfhq_prefix}/SFHQ_4D/SFHQ_4D-{{000000..000312}}.tar",
    ]

    return shards


def get_sfhq_eval(base_url: str | None = None):
    base_url = base_url or get_base_url()

    sfhq_prefix = f"{base_url}/SFHQ"
    shards = [
        f"{sfhq_prefix}/SFHQ_1A/SFHQ_1A-{{000223..000223}}.tar",
        f"{sfhq_prefix}/SFHQ_1B/SFHQ_1B-{{000223..000223}}.tar",
        f"{sfhq_prefix}/SFHQ_1C/SFHQ_1C-{{000223..000223}}.tar",
        f"{sfhq_prefix}/SFHQ_1D/SFHQ_1D-{{000223..000223}}.tar",
        f"{sfhq_prefix}/SFHQ_2A/SFHQ_2A-{{000227..000227}}.tar",
        f"{sfhq_prefix}/SFHQ_2B/SFHQ_2B-{{000227..000227}}.tar",
        f"{sfhq_prefix}/SFHQ_2C/SFHQ_2C-{{000227..000227}}.tar",
        f"{sfhq_prefix}/SFHQ_2D/SFHQ_2D-{{000227..000227}}.tar",
        f"{sfhq_prefix}/SFHQ_3A/SFHQ_3A-{{000294..000294}}.tar",
        f"{sfhq_prefix}/SFHQ_3B/SFHQ_3B-{{000294..000294}}.tar",
        f"{sfhq_prefix}/SFHQ_3C/SFHQ_3C-{{000294..000294}}.tar",
        f"{sfhq_prefix}/SFHQ_3D/SFHQ_3D-{{000294..000294}}.tar",
        f"{sfhq_prefix}/SFHQ_4A/SFHQ_4A-{{000313..000313}}.tar",
        f"{sfhq_prefix}/SFHQ_4B/SFHQ_4B-{{000313..000313}}.tar",
        f"{sfhq_prefix}/SFHQ_4C/SFHQ_4C-{{000313..000313}}.tar",
        f"{sfhq_prefix}/SFHQ_4D/SFHQ_4D-{{000313..000313}}.tar",
    ]

    return shards
