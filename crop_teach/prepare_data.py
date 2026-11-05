#!/usr/bin/env python3
"""Write the data in the layout data.py reads (run once, where the raw data is).

    python prepare_data.py --raw ../data --out data

--raw is the folder holding od_oc/, prostate-6/ and polyp-4/ as the ReDi-FM repository lays them out
(src/download/ there fetches them). Output, per problem and domain:
    <out>/<problem>/domain_<letter>/Images/<NNNN>_<name>.<ext>   the original file, unchanged
    <out>/<problem>/domain_<letter>/Masks/<NNNN>_<name>.png      class indices: 0 background, 1.., nested
The NNNN prefix keeps each domain's stream order (e.g. REFUGE val before test).
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def class_mask(raw: Image.Image) -> Image.Image:
    """Class indices from a mask that is either already 0/1/2 or the fundus convention
    255 = background, 128 = disc rim, 0 = cup."""
    array = np.asarray(raw.convert("L"), dtype=np.uint8)
    if set(np.unique(array).tolist()) <= {0, 1, 2}:
        return Image.fromarray(array)
    target = np.zeros(array.shape, dtype=np.uint8)
    target[array < 255] = 1
    target[array == 0] = 2
    return Image.fromarray(target)


def pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    return [(path, masks[path.stem]) for path in sorted(p for p in image_dir.iterdir() if p.is_file())]


def write_domain(out: Path, items: list[tuple[str, bytes, str, Image.Image]]) -> None:
    """items: (name, image bytes, image extension, mask)."""
    images, masks = out / "Images", out / "Masks"
    for folder in (images, masks):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    for index, (name, data, extension, mask) in enumerate(items):
        stem = f"{index:04d}_{name}"
        (images / f"{stem}{extension}").write_bytes(data)
        class_mask(mask).save(masks / f"{stem}.png")
    print(f"{out}: {len(items)} images")


def from_files(file_pairs: list[tuple[Path, Path]], prefix: str = "") -> list:
    return [(prefix + image.stem, image.read_bytes(), image.suffix.lower(), Image.open(mask))
            for image, mask in file_pairs]


def odoc(raw: Path, out: Path) -> None:
    root = raw / "od_oc"
    rim = root / "domain_A_RIM_ONE_r3/RIM-ONE-r3"
    write_domain(out / "domain_A", sum((from_files(pairs(rim / f"{s}/images", rim / f"{s}/masks"), f"{s}_")
                                        for s in ("training", "testing")), []))
    refuge = root / "domain_B_REFUGE_train/train"
    write_domain(out / "domain_B", from_files(pairs(refuge / "Images", refuge / "Masks")))
    origa = root / "domain_C_ORIGA"
    write_domain(out / "domain_C", from_files(pairs(origa / "Images", origa / "Masks")))
    target = root / "domain_D_REFUGE_val_test"
    write_domain(out / "domain_D", sum((from_files(pairs(target / f"{s}/Images", target / f"{s}/Masks"), f"{s}_")
                                        for s in ("val", "test")), []))
    items = []
    with zipfile.ZipFile(root / "FunduSegmenter_Datasets.zip") as archive:
        names = set(archive.namelist())
        for split in ("training", "testing"):
            prefix = f"Drishti-GS/{split}/images/"
            for member in sorted(n for n in names if n.startswith(prefix) and not n.endswith("/")):
                filename = Path(member).name
                mask = Image.open(io.BytesIO(archive.read(f"Drishti-GS/{split}/masks/{filename}")))
                items.append((f"{split}_{Path(filename).stem}", archive.read(member), Path(filename).suffix.lower(), mask))
    write_domain(out / "domain_E", items)


def site_folders(raw: Path, out: Path, folder: str, names: dict[str, str]) -> None:
    for letter, name in names.items():
        source = raw / folder / f"domain_{letter}_{name}"
        write_domain(out / f"domain_{letter}", from_files(pairs(source / "Images", source / "Masks")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    odoc(args.raw, args.out / "odoc")
    site_folders(args.raw, args.out / "prostate", "prostate-6",
                 {"A": "BIDMC", "B": "BMC", "C": "HK", "D": "I2CVB", "E": "RUNMC", "F": "UCL"})
    site_folders(args.raw, args.out / "polyp", "polyp-4",
                 {"A": "BKAI_IGH_NeoPolyp", "B": "CVC_ClinicDB", "C": "ETIS_LaribPolypDB", "D": "Kvasir_SEG"})


if __name__ == "__main__":
    main()
