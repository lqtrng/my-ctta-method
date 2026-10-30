#!/usr/bin/env python3
"""Fetch every external file used by the methods in this repository.

This is the only script that downloads anything.  Every method loads its
files from the local paths in ``common/paths.py`` and fails with a pointer
back here when a file is missing.

    python src/download/download.py              # fetch everything the main pipeline needs
    python src/download/download.py dinov3 sam   # fetch selected assets
    python src/download/download.py --check      # report what is present

Public foundation-model weights come from their official sources.  Files with
no public download (the OD/OC data, the prepared prostate data, this
project's checkpoints) come from the private Kaggle asset bundle, see
``bundle.py``.  Assets marked optional are used only by ``extras/`` and are
fetched only when named.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import (  # noqa: E402
    BIOMEDCLIP_CHECKPOINT,
    BIOMEDCLIP_DIR,
    DATA_DIR,
    DINOV3_ARCH,
    DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    DINOV3_HEAD_CHECKPOINT,
    MEDDINOV3_CHECKPOINT,
    MEDDINOV3_REPO_DIR,
    MODEL_DIR,
    POLYP_DIR,
    PRETRAINED_RESUNET_CHECKPOINT,
    PROSTATE_ARCHIVE,
    PROSTATE_DINOV3_HEAD_CHECKPOINT,
    PROSTATE_DIR,
    SAM_CHECKPOINT,
    SOURCE_CHECKPOINTS,
    dinov3_weights,
)
from download.bundle import link_from_bundle  # noqa: E402


SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
BIOMEDCLIP_MODEL_ID = "chuhac/BiomedCLIP-vit-bert-hf"
BIOMEDCLIP_REVISION = "c44fed22e30460c0ea1b033a41b68fdab375d64c"
PROSTATE_DRIVE_ID = "1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-"  # Processed_data_nii.zip, liuquande.github.io/SAML
MEDDINOV3_MODEL_ID = "ricklisz123/MedDINOv3-ViTB-16-CT-3M"
MEDDINOV3_REVISION = "20d233b0ce1b1f93c46550787cf4b409a6f4a95e"
MEDDINOV3_REPO_URL = "https://github.com/ricklisz/MedDINOv3.git"
MEDDINOV3_REPO_COMMIT = "a2460c2547899283446e520b84523454f7e25608"


@dataclass(frozen=True)
class Asset:
    path: Path
    used_by: str
    fetch: Callable[[Path], None] | None = None
    manual: str = ""
    # Used only by extras/: fetched only when named on the command line.
    optional: bool = False
    # A checkpoint reproduce.sh can train when the bundle does not have it.
    trainable: bool = False


def from_bundle(path: Path) -> None:
    link_from_bundle(path)


def fetch_dinov3(path: Path) -> None:
    """timm port of the official DINOv3 ViT-L/16 LVD-1689M weights (ungated)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        f"timm/{DINOV3_ARCH}",
        allow_patterns=["model.safetensors", "config.json", "LICENSE.md"],
        local_dir=path.parent,
    )


def fetch_meddinov3(path: Path) -> None:
    """Official MedDINOv3 ViT-B/16 checkpoint, pretrained on CT-3M."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        MEDDINOV3_MODEL_ID, "model.pth", revision=MEDDINOV3_REVISION, local_dir=path.parent
    )


def fetch_meddinov3_code(path: Path) -> None:
    """Official MedDINOv3 implementation (model code and Primus decoder)."""
    import subprocess

    subprocess.run(["git", "clone", MEDDINOV3_REPO_URL, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", MEDDINOV3_REPO_COMMIT], check=True)


def fetch_prostate(path: Path) -> None:
    """Six-site prostate MRI release, then the OD/OC-style per-domain layout."""
    import subprocess

    import gdown

    if not PROSTATE_ARCHIVE.is_file():
        PROSTATE_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=PROSTATE_DRIVE_ID, output=str(PROSTATE_ARCHIVE))
    script = Path(__file__).resolve().parent / "prepare_prostate.py"
    subprocess.run([sys.executable, str(script)], check=True)


def fetch_polyp(path: Path) -> None:
    """The four public polyp datasets, then the OD/OC-style per-domain layout."""
    import subprocess

    script = Path(__file__).resolve().parent / "prepare_polyp.py"
    subprocess.run([sys.executable, str(script)], check=True)


def fetch_sam(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(SAM_URL) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 20)
    partial.replace(path)


def fetch_biomedclip_config(path: Path) -> None:
    """Config, remote code and tokenizer only; the weights are the fine-tuned checkpoint."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        BIOMEDCLIP_MODEL_ID,
        revision=BIOMEDCLIP_REVISION,
        ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.pth"],
        local_dir=path.parent,
    )


def checkpoint(path: Path, used_by: str, trainable: bool = True) -> Asset:
    return Asset(path, used_by, from_bundle, trainable=trainable)


ASSETS = {
    # Public foundation models.
    "dinov3": Asset(dinov3_weights(MODEL_DIR, DINOV3_ARCH), "redifm_dinov3, dinov3_*", fetch_dinov3),
    "sam": Asset(SAM_CHECKPOINT, "redifm_sam", fetch_sam),
    # Asset bundle (private Kaggle dataset).
    "data": Asset(DATA_DIR, "all OD/OC methods", lambda _: from_bundle(DATA_DIR)),
    "prostate": Asset(PROSTATE_DIR / "manifest.json", "prostate experiments", lambda _: from_bundle(PROSTATE_DIR)),
    "polyp": Asset(POLYP_DIR / "manifest.json", "polyp experiments", lambda _: from_bundle(POLYP_DIR)),
    "pretrained-resunet": checkpoint(PRETRAINED_RESUNET_CHECKPOINT, "training/train_source.py", trainable=False),
    **{
        f"source-{task}-{arch}": checkpoint(path, f"every {task} method with --arch {arch}")
        for (task, arch), path in SOURCE_CHECKPOINTS.items()
    },
    "dinov3-head": checkpoint(DINOV3_HEAD_CHECKPOINT, "redifm_dinov3 (odoc), dinov3_dual"),
    "dinov3-full-image-head": checkpoint(DINOV3_FULL_IMAGE_HEAD_CHECKPOINT, "dinov3_single, dinov3_dual"),
    "prostate-dinov3-head": checkpoint(PROSTATE_DINOV3_HEAD_CHECKPOINT, "redifm_dinov3 (prostate)"),
    # extras/ only.
    "prostate-public": Asset(
        PROSTATE_ARCHIVE, "regenerating data/prostate-6 with prepare_prostate.py", fetch_prostate, optional=True
    ),
    "polyp-public": Asset(
        POLYP_DIR / "raw", "regenerating data/polyp-4 with prepare_polyp.py", fetch_polyp, optional=True
    ),
    "meddinov3": Asset(MEDDINOV3_CHECKPOINT, "meddinov3", fetch_meddinov3, optional=True),
    "meddinov3-code": Asset(MEDDINOV3_REPO_DIR, "meddinov3", fetch_meddinov3_code, optional=True),
    "biomedclip-config": Asset(
        BIOMEDCLIP_DIR / "config.json", "extras/medclip_samv2", fetch_biomedclip_config, optional=True
    ),
    "biomedclip-weights": Asset(
        BIOMEDCLIP_CHECKPOINT,
        "extras/medclip_samv2",
        manual="fine-tuned BiomedCLIP pytorch_model.bin from the MedCLIP-SAMv2 release",
        optional=True,
    ),
}

def check() -> bool:
    ok = True
    for name, asset in ASSETS.items():
        present = asset.path.exists()
        ok &= present or asset.optional
        how = "downloadable" if asset.fetch else f"manual: {asset.manual}"
        if asset.optional:
            how += "; optional"
        print(f"[{'ok' if present else 'missing'}] {name:<26} {asset.path}")
        print(f"    used by {asset.used_by}; {how}")
    return ok

def main() -> None:
    downloadable = [name for name, asset in ASSETS.items() if asset.fetch]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assets", nargs="*", choices=downloadable,
                        help="default: every downloadable asset that is not optional")
    parser.add_argument("--check", action="store_true", help="only report which assets are present")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    for name in args.assets or [name for name in downloadable if not ASSETS[name].optional]:
        asset = ASSETS[name]
        if asset.path.exists() and not args.force:
            print(f"{name}: already present at {asset.path}")
            continue
        print(f"{name}: fetching to {asset.path}")
        try:
            asset.fetch(asset.path)
        except FileNotFoundError as error:
            if not asset.trainable:
                raise
            print(f"{name}: {error}\n    reproduce.sh trains it before it is used")

if __name__ == "__main__":
    main()
