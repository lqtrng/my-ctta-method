#!/usr/bin/env python3
"""Lay out the six-site prostate MRI benchmark like the OD/OC data.

Input: ``Processed_data_nii.zip`` from the SAML / FedDG multi-site prostate
release (https://liuquande.github.io/SAML/), already resized to 384 x 384 in
the axial plane.  It contains one T2-weighted volume and one segmentation per
case, in one folder per site.

Output, under ``data/prostate-6`` (one folder per paper domain, as in
``data/od_oc``)::

    domain_A_BIDMC/Images/Case00_slice012.png    8-bit grayscale slice
    domain_A_BIDMC/Masks/Case00_slice012.png     class index: 0 background, 1 prostate
    ...
    domain_F_UCL/...
    manifest.json                                slice list, counts and settings

Following 0140.pdf (Section 4.1), only axial slices that contain prostate are
kept, which reproduces the paper's slice counts exactly (A 261, B 384, C 158,
D 468, E 421, F 175).  BMC and RUNMC annotate two zones (labels 1 and 2), the
other sites the whole gland, so every non-zero label becomes prostate.

The paper does not state how MRI intensities are stored.  Each volume is
clipped to its own 0.5-99.5 intensity percentiles and scaled to 0-255.  This is
a per-volume rescale, so slices of one case keep their relative brightness.
The shared protocol later min-max normalizes every image again.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import PROSTATE_ARCHIVE, PROSTATE_DIR  # noqa: E402


# Paper domain letter -> site folder in the release, with the paper's slice counts.
SITES = {
    "A": ("BIDMC", 261),
    "B": ("BMC", 384),
    "C": ("HK", 158),
    "D": ("I2CVB", 468),
    "E": ("RUNMC", 421),
    "F": ("UCL", 175),
}
PERCENTILES = (0.5, 99.5)


def to_uint8(volume: np.ndarray) -> np.ndarray:
    low, high = np.percentile(volume, PERCENTILES)
    scaled = (np.clip(volume, low, high) - low) / max(high - low, 1e-7)
    return np.rint(scaled * 255.0).astype(np.uint8)


def prepare_site(extracted: Path, site: str, output: Path) -> list[dict[str, object]]:
    import nibabel as nib

    images_dir = output / "Images"
    masks_dir = output / "Masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    volumes = sorted(
        path for path in (extracted / site).glob("*.nii.gz") if "egmentation" not in path.name
    )
    for volume_path in volumes:
        case = volume_path.name.removesuffix(".nii.gz")
        (mask_path,) = (extracted / site).glob(f"{case}_[Ss]egmentation.nii.gz")
        volume = np.asarray(nib.load(volume_path).dataobj, dtype=np.float32)
        labels = np.asarray(nib.load(mask_path).dataobj)
        if volume.shape != labels.shape:
            raise RuntimeError(f"{site}/{case}: image {volume.shape} vs mask {labels.shape}")
        volume = to_uint8(volume)
        for index in range(volume.shape[2]):
            mask = (labels[:, :, index] > 0).astype(np.uint8)
            if not mask.any():
                continue
            name = f"{case}_slice{index:03d}.png"
            # NIfTI stores (x, y); transpose to the usual (row, column) image view.
            Image.fromarray(volume[:, :, index].T).save(images_dir / name)
            Image.fromarray(mask.T).save(masks_dir / name)
            rows.append({"image": name, "case": case, "slice": index,
                         "foreground_pixels": int(mask.sum())})
    return rows


def main(args: argparse.Namespace) -> None:
    archive = args.archive
    extracted = args.extracted
    if not (extracted / "BIDMC").is_dir():
        with zipfile.ZipFile(archive) as source:
            source.extractall(extracted)

    manifest: dict[str, object] = {
        "source": "https://liuquande.github.io/SAML/ (Processed_data_nii.zip)",
        "reference": "0140.pdf Section 4.1; only slices containing prostate are kept",
        "intensity": f"per-volume clip to percentiles {PERCENTILES}, scaled to uint8",
        "mask": "class index PNG: 0 background, 1 prostate (all non-zero labels)",
        "domains": {},
    }
    for domain, (site, expected) in SITES.items():
        output = args.output / f"domain_{domain}_{site}"
        rows = prepare_site(extracted, site, output)
        if len(rows) != expected:
            raise RuntimeError(f"Domain {domain} ({site}): expected {expected} slices, found {len(rows)}")
        manifest["domains"][domain] = {
            "site": site,
            "folder": output.name,
            "cases": len({row["case"] for row in rows}),
            "slices": len(rows),
            "items": rows,
        }
        print(f"Domain {domain} ({site}): {len(rows)} slices from "
              f"{manifest['domains'][domain]['cases']} cases -> {output}")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved: {args.output / 'manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=PROSTATE_ARCHIVE)
    parser.add_argument("--extracted", type=Path, default=PROSTATE_ARCHIVE.parent / "Processed_data_nii")
    parser.add_argument("--output", type=Path, default=PROSTATE_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
