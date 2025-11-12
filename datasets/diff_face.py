from .utils.nextcloud import get_base_url


def get_diffface_shards():
    base_url = get_base_url()

    diff_prefix = f"{base_url}/data/diffusion_face"
    celeba_prefix = f"{base_url}/data/mm_celeba_hq"

    ai_shards = [
        f"{diff_prefix}/ADM-{{range}}.tar",
        f"{diff_prefix}/DDIM-{{range}}.tar",
        f"{diff_prefix}/DDPM-{{range}}.tar",
        f"{diff_prefix}/DiffSwap-{{range}}.tar",
        f"{diff_prefix}/Inpaint-{{range}}.tar",
        f"{diff_prefix}/LDM-{{range}}.tar",
        f"{diff_prefix}/PNDM-{{range}}.tar",
        f"{diff_prefix}/SDv15_DS0.3-{{range}}.tar",
        f"{diff_prefix}/SDv15_DS0.5-{{range}}.tar",
        f"{diff_prefix}/SDv15_DS0.7-{{range}}.tar",
        f"{diff_prefix}/SDv21_DS0.3-{{range}}.tar",
        f"{diff_prefix}/SDv21_DS0.5-{{range}}.tar",
        f"{diff_prefix}/SDv21_DS0.7-{{range}}.tar",
    ]
    real_shards = [f"{celeba_prefix}/mm_celeba_hq-{{range}}.tar"]

    train_ai = [p.format(range="{000..004}") for p in ai_shards]
    train_real = [p.format(range="{000..004}") for p in real_shards]

    val_ai = [p.format(range="{005..005}") for p in ai_shards]
    val_real = [p.format(range="{005..005}") for p in real_shards]

    return train_ai, train_real, val_ai, val_real
