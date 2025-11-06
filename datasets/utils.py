from pathlib import Path

import webdataset as wds
import braceexpand

def split_tarball(input_tar: Path, output_pattern:str, count: int):
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

def expand_brace_patterns(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        out.extend(list(braceexpand.braceexpand(p)))
    return out