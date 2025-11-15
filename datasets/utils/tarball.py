from pathlib import Path
import webdataset as wds


def split_tarball(input_tar: Path, output_pattern: str, count: int):
    tar_path = str(input_tar.as_posix())
    print(tar_path)
    with wds.writer.ShardWriter(output_pattern, maxcount=count) as sink:
        for sample in wds.compat.WebDataset(tar_path):
            keys = sample.keys()
            if any(k.endswith(("jpg", "jpeg", "png")) for k in keys):
                sink.write(sample)


def find_tarballs(root_dir: Path) -> list[str]:
    root = Path(root_dir).expanduser().absolute()
    tar_paths = sorted(p.as_posix() for p in root.glob("*.tar"))
    return tar_paths


def create_shards(images_dir: Path, output_pattern: str, maxcount: int = 1000):
    Path(output_pattern).parent.mkdir(parents=True, exist_ok=True)

    sink = wds.writer.ShardWriter(
        output_pattern,
        maxcount=maxcount,
        opener=lambda f: open(f, "wb"),
    )
    for img in images_dir.iterdir():
        if not img.is_file():
            continue
        suffix = img.suffix.lower().lstrip(".")
        if suffix not in ["jpg", "jpeg", "png", "webp"]:
            continue

        sample = {
            "__key__": str(img.stem),
            suffix: img.read_bytes(),
        }

        sink.write(sample)
    sink.close()
