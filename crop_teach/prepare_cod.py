#!/usr/bin/env python3
"""Download the camouflaged-object problem ("cod") and write it in the layout data.py reads.

    python prepare_cod.py --raw /path/to/raw_cod --out data/cod

A non-medical binary segmentation problem for testing Crop-Teach unchanged. Each domain is a whole
public benchmark, from the Hugging Face mirrors of the nobg organization (parquet, original bytes):
    A  CAMO          nobg/camo,   train + test   1250 images
    B  COD10K-test   nobg/COD10K, test           2026 images
    C  NC4K          nobg/NC4K,   test           4121 images
Output, per domain:
    <out>/domain_<letter>/Images/<NNNN>_<name>.<ext>   the original image bytes, unchanged
    <out>/domain_<letter>/Masks/<NNNN>_<name>.png      0 background, 1 object (mask > 127)
"""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

DOMAINS = {  # letter -> (repository, splits in stream order)
    "A": ("nobg/camo", ("train", "test")),
    "B": ("nobg/COD10K", ("test",)),
    "C": ("nobg/NC4K", ("test",)),
}


def parquet_files(repo: str, split: str, raw: Path) -> list[Path]:
    names = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    return [Path(hf_hub_download(repo, name, repo_type="dataset", local_dir=raw / repo.replace("/", "_")))
            for name in names]


def extension(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        return {"JPEG": ".jpg", "PNG": ".png", "BMP": ".bmp", "WEBP": ".webp"}.get(image.format, ".png")


def binary_mask(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as raw:
        array = np.asarray(raw.convert("L"), dtype=np.uint8)
    return Image.fromarray((array > 127).astype(np.uint8))


def write_domain(letter: str, repo: str, splits: tuple[str, ...], raw: Path, out: Path) -> None:
    folder = out / f"domain_{letter}"
    images, masks = folder / "Images", folder / "Masks"
    for path in (images, masks):
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
    index = 0
    for split in splits:
        for file in parquet_files(repo, split, raw):
            table = pq.read_table(file)
            names = table.column("image_name").to_pylist() if "image_name" in table.column_names else None
            for row, (image, mask) in enumerate(zip(table.column("image").to_pylist(),
                                                     table.column("mask").to_pylist())):
                name = Path(names[row]).stem if names else f"{file.stem}_{row:05d}"
                stem = f"{index:04d}_{split}_{name}"
                (images / f"{stem}{extension(image['bytes'])}").write_bytes(image["bytes"])
                binary_mask(mask["bytes"]).save(masks / f"{stem}.png")
                index += 1
    print(f"{folder}: {index} images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="Download cache for the parquet files")
    parser.add_argument("--out", type=Path, required=True, help="Usually data/cod")
    args = parser.parse_args()
    for letter, (repo, splits) in DOMAINS.items():
        write_domain(letter, repo, splits, args.raw, args.out)


if __name__ == "__main__":
    main()
