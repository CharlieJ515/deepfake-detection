from .utils.nextcloud import get_base_url


def get_face3000_train(base_url: str | None = None):
    base_url = base_url or get_base_url()

    face3000_prefix = f"{base_url}/AIFaceDataset3000"
    shards = [
        f"{face3000_prefix}/AIFaceDataset3000-{{000000..000028}}.tar",
    ]

    return shards


def get_face3000_eval(base_url: str | None = None):
    base_url = base_url or get_base_url()

    face3000_prefix = f"{base_url}/AIFaceDataset3000"
    shards = [
        f"{face3000_prefix}/AIFaceDataset3000-{{000029..000029}}.tar",
    ]

    return shards
