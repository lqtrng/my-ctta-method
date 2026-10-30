#!/usr/bin/env python3
"""Download the four polyp datasets and lay them out like the OD/OC data.

0140.pdf, Section 4.1 evaluates polyp segmentation on four datasets:

    Domain A  BKAI-IGH NeoPolyp   1000 images   (source domain)
    Domain B  CVC-ClinicDB         612 images
    Domain C  ETIS-LaribPolypDB    196 images
    Domain D  Kvasir-SEG          1000 images

Output, under ``data/polyp-4`` (one folder per paper domain, as in
``data/od_oc`` and ``data/prostate-6``)::

    domain_A_BKAI_IGH_NeoPolyp/Images/<name>.jpeg   image, original bytes
    domain_A_BKAI_IGH_NeoPolyp/Masks/<name>.png     class index: 0 background, 1 polyp
    ...
    domain_D_Kvasir_SEG/...
    manifest.json                                   image list, counts, sources and settings

Images keep their original resolution and encoding (CVC-ClinicDB's TIFFs
become lossless PNGs).  The paper's 352 x 352 resize happens when the data is
loaded, as the OD/OC images are resized to 512 x 512 at load time.

Masks are binary.  CVC-ClinicDB and Kvasir-SEG masks carry grey edge values
(anti-aliasing, JPEG artefacts) and are thresholded at 128, the PraNet
convention.  BKAI-IGH colours neoplastic polyps red and non-neoplastic ones
green; both are polyp, so a pixel is polyp when its red or green channel is
above 128.

Sources:

* BKAI-IGH NeoPolyp: only distributed as the Kaggle competition
  ``bkai-igh-neopolyp`` (training set: ``train/`` and ``train_gt/``).  Accept
  the competition rules once on kaggle.com; the files are then downloaded with
  the Kaggle CLI, or read from ``/kaggle/input/bkai-igh-neopolyp`` when the
  competition is attached to a Kaggle notebook.  ``--bkai-dir`` points at an
  existing copy.
* CVC-ClinicDB (original TIFFs) and ETIS-LaribPolypDB: the Hugging Face mirror
  ``kinghoon/polyp-dataset``.
* Kvasir-SEG: the Hugging Face mirror ``kowndinya23/Kvasir-SEG`` (the official
  download at datasets.simula.no holds the same 1000 images).

    python src/download/prepare_polyp.py
    python src/download/prepare_polyp.py --bkai-dir /path/to/bkai-igh-neopolyp
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import POLYP_DIR, POLYP_RAW_DIR  # noqa: E402


BKAI_COMPETITION = "bkai-igh-neopolyp"
POLYP_MIRROR = ("kinghoon/polyp-dataset", "8b5ca7b325acd4d5529c377e32183f4723c9b8af")
KVASIR_MIRROR = ("kowndinya23/Kvasir-SEG", "01412ba699035de68fd55ae36045437c28073a06")
# Paper domain letter -> (output folder, the paper's image count).
DOMAINS = {
    "A": ("domain_A_BKAI_IGH_NeoPolyp", 1000),
    "B": ("domain_B_CVC_ClinicDB", 612),
    "C": ("domain_C_ETIS_LaribPolypDB", 196),
    "D": ("domain_D_Kvasir_SEG", 1000),
}
MASK_THRESHOLD = 128


# --- downloading ------------------------------------------------------------


def hf_snapshot(repo: str, revision: str, patterns: list[str] | None, output: Path) -> Path:
    """A pinned Hugging Face dataset snapshot; waits out the hub's rate limit."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(10):
        try:
            snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=patterns,
                              local_dir=output, max_workers=2)
            return output
        except HfHubHTTPError as error:
            if "429" not in str(error) or attempt == 9:
                raise
            print("Hugging Face rate limit; retrying in 5 minutes")
            time.sleep(300)
    return output


def bkai_dir(given: Path | None) -> Path:
    """The BKAI-IGH competition files: given, attached on Kaggle, or downloaded."""
    candidates = [given] if given else [Path("/kaggle/input") / BKAI_COMPETITION, POLYP_RAW_DIR / "bkai"]
    for candidate in candidates:
        if candidate and any(candidate.rglob("train_gt")):
            return candidate
    if given:
        raise FileNotFoundError(f"{given} has no train_gt/ folder from the {BKAI_COMPETITION} competition")
    output = POLYP_RAW_DIR / "bkai"
    output.mkdir(parents=True, exist_ok=True)
    cli = Path(sys.executable).with_name("kaggle")
    command = [str(cli) if cli.is_file() else "kaggle", "competitions", "download", "-c", BKAI_COMPETITION,
               "-p", str(output)]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise FileNotFoundError(
            f"Could not download the Kaggle competition {BKAI_COMPETITION} ({error}). It needs Kaggle API "
            f"credentials (KAGGLE_API_TOKEN or ~/.kaggle/kaggle.json) and the competition rules accepted on "
            f"https://www.kaggle.com/c/{BKAI_COMPETITION}/rules; or pass --bkai-dir."
        ) from error
    for archive in output.glob("*.zip"):
        with zipfile.ZipFile(archive) as opened:
            opened.extractall(output)
    return output


# --- formatting -------------------------------------------------------------


def save_pair(image: Path | bytes | np.ndarray, mask: np.ndarray, name: str, suffix: str,
              output: Path) -> dict[str, object]:
    """Write one image (original bytes when given a file or bytes) and its 0/1 mask."""
    images_dir, masks_dir = output / "Images", output / "Masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    target = images_dir / f"{name}{suffix}"
    if isinstance(image, Path):
        shutil.copyfile(image, target)
    elif isinstance(image, bytes):
        target.write_bytes(image)
    else:
        Image.fromarray(image).save(target)
    with Image.open(target) as opened:
        size = opened.size
    if size != (mask.shape[1], mask.shape[0]):
        raise RuntimeError(f"{output.name}/{name}: image {size} vs mask {mask.shape[::-1]}")
    Image.fromarray(mask.astype(np.uint8)).save(masks_dir / f"{name}.png")
    return {"image": target.name, "mask": f"{name}.png", "width": size[0], "height": size[1],
            "foreground_pixels": int(mask.sum())}


def grey_mask(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        array = array.max(axis=2)
    return array >= MASK_THRESHOLD


def prepare_bkai(source: Path, output: Path) -> list[dict[str, object]]:
    images = {path.stem: path for path in source.rglob("*.jpeg") if path.parent.name == "train"}
    masks = {path.stem: path for path in source.rglob("*.jpeg") if path.parent.name == "train_gt"}
    if images.keys() != masks.keys():
        raise RuntimeError(f"BKAI-IGH: {len(images)} images but {len(masks)} masks in {source}")
    rows = []
    for name in sorted(images):
        rgb = np.asarray(Image.open(masks[name]).convert("RGB"))
        mask = (rgb[..., 0] >= MASK_THRESHOLD) | (rgb[..., 1] >= MASK_THRESHOLD)
        rows.append(save_pair(images[name], mask, name, ".jpeg", output))
    return rows


def prepare_cvc(source: Path, output: Path) -> list[dict[str, object]]:
    import tifffile

    root = source / "CVC-ClinicDB/CVC-ClinicDB"
    rows = []
    for image in sorted((root / "Original").glob("*.tif"), key=lambda path: int(path.stem)):
        mask = grey_mask(tifffile.imread(root / "Ground Truth" / image.name))
        rows.append(save_pair(tifffile.imread(image), mask, image.stem, ".png", output))
    return rows


def prepare_etis(source: Path, output: Path) -> list[dict[str, object]]:
    root = source / "ETIS-LaribPolypDB/ETIS-LaribPolypDB"
    rows = []
    for image in sorted((root / "images").glob("*.png"), key=lambda path: int(path.stem)):
        mask = grey_mask(np.asarray(Image.open(root / "masks" / image.name).convert("L")))
        rows.append(save_pair(image, mask, image.stem, ".png", output))
    return rows


def prepare_kvasir(source: Path, output: Path) -> list[dict[str, object]]:
    import pyarrow.parquet as pq

    rows = []
    items = [item for path in sorted((source / "data").glob("*.parquet")) for item in pq.read_table(path).to_pylist()]
    for item in sorted(items, key=lambda item: item["name"]):
        mask = grey_mask(np.asarray(Image.open(io.BytesIO(item["annotation"]["bytes"])).convert("L")))
        rows.append(save_pair(item["image"]["bytes"], mask, item["name"], ".jpg", output))
    return rows


def main(args: argparse.Namespace) -> None:
    mirror = hf_snapshot(*POLYP_MIRROR, ["CVC-ClinicDB/*", "ETIS-LaribPolypDB/*"],
                         POLYP_RAW_DIR / "kinghoon-polyp-dataset")
    kvasir = hf_snapshot(*KVASIR_MIRROR, None, POLYP_RAW_DIR / "kvasir-seg-hf")
    builders = {
        "A": (prepare_bkai, lambda: bkai_dir(args.bkai_dir), f"https://www.kaggle.com/c/{BKAI_COMPETITION} (train, train_gt)"),
        "B": (prepare_cvc, lambda: mirror, f"https://huggingface.co/datasets/{POLYP_MIRROR[0]}@{POLYP_MIRROR[1]}"),
        "C": (prepare_etis, lambda: mirror, f"https://huggingface.co/datasets/{POLYP_MIRROR[0]}@{POLYP_MIRROR[1]}"),
        "D": (prepare_kvasir, lambda: kvasir, f"https://huggingface.co/datasets/{KVASIR_MIRROR[0]}@{KVASIR_MIRROR[1]}"),
    }

    manifest_path = args.output / "manifest.json"
    manifest: dict[str, object] = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.update({
        "reference": "0140.pdf Section 4.1; images are resized to 352 x 352 when loaded",
        "image": "original resolution and encoding (CVC-ClinicDB TIFF -> PNG)",
        "mask": f"class index PNG: 0 background, 1 polyp (grey values >= {MASK_THRESHOLD}; "
                "BKAI-IGH: red or green channel >= 128, i.e. neoplastic and non-neoplastic)",
    })
    manifest.setdefault("domains", {})
    for domain in args.domains:
        folder, expected = DOMAINS[domain]
        prepare, source, origin = builders[domain]
        output = args.output / folder
        try:
            source_dir = source()
        except FileNotFoundError as error:
            print(f"Domain {domain} skipped: {error}")
            continue
        if output.exists():
            shutil.rmtree(output)
        rows = prepare(source_dir, output)
        if len(rows) != expected:
            raise RuntimeError(f"Domain {domain} ({folder}): expected {expected} images, found {len(rows)}")
        manifest["domains"][domain] = {"folder": folder, "source": origin, "images": len(rows), "items": rows}
        print(f"Domain {domain}: {len(rows)} images -> {output}")
    manifest["domains"] = dict(sorted(manifest["domains"].items()))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    missing = [domain for domain in DOMAINS if domain not in manifest["domains"]]
    print(f"Saved: {manifest_path}" + (f" (missing domains: {', '.join(missing)})" if missing else ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bkai-dir", type=Path, help="existing copy of the bkai-igh-neopolyp competition files")
    parser.add_argument("--domains", nargs="+", choices=list(DOMAINS), default=list(DOMAINS))
    parser.add_argument("--output", type=Path, default=POLYP_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
