import os


def get_base_url() -> str:
    nc_host = os.environ["NC_HOST"]
    nc_user = os.environ["NC_USER"]
    nc_pass = os.environ["NC_PASS"]

    base_url = (
        f"https://{nc_user}:{nc_pass}@{nc_host}/remote.php/dav/files/{nc_user}/data"
    )

    return base_url
