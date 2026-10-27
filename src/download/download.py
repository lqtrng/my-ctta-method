#!/usr/bin/env python3
"""Fetch the external model weights used by the methods in this repository.

This is the only script that downloads anything.  Every method loads its
weights from the local paths in ``common/paths.py`` and fails with a pointer
back here when a file is missing.

    python src/download/download.py              # fetch everything downloadable
    python src/download/download.py dinov3 sam   # fetch selected assets
    python src/download/download.py --check      # report what is present

Assets that have no public download (datasets, the source ResUNet-34
checkpoints, the MedCLIP-SAMv2 fine-tuned BiomedCLIP weights) are listed by
``--check`` but must be placed by hand.
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
    DEFAULT_SOURCE_CHECKPOINT,
    DINOV3_ARCH,
    MODEL_DIR,
    PRETRAINED_RESUNET_CHECKPOINT,
    SAM_CHECKPOINT,
    dinov3_weights,
)


SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
BIOMEDCLIP_MODEL_ID = "chuhac/BiomedCLIP-vit-bert-hf"
BIOMEDCLIP_REVISION = "c44fed22e30460c0ea1b033a41b68fdab375d64c"


@dataclass(frozen=True)
class Asset:
    path: Path
    used_by: str
    fetch: Callable[[Path], None] | None = None
    manual: str = ""


def fetch_dinov3(path: Path) -> None:
    """timm port of the official DINOv3 ViT-L/16 LVD-1689M weights (ungated)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        f"timm/{DINOV3_ARCH}",
        allow_patterns=["model.safetensors", "config.json", "LICENSE.md"],
        local_dir=path.parent,
    )


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


ASSETS = {
    "dinov3": Asset(dinov3_weights(MODEL_DIR, DINOV3_ARCH), "redifm_dinov3", fetch_dinov3),
    "sam": Asset(SAM_CHECKPOINT, "redifm_sam, medclip_samv2", fetch_sam),
    "biomedclip-config": Asset(BIOMEDCLIP_DIR / "config.json", "medclip_samv2", fetch_biomedclip_config),
    "biomedclip-weights": Asset(
        BIOMEDCLIP_CHECKPOINT,
        "medclip_samv2",
        manual="fine-tuned BiomedCLIP pytorch_model.bin from the MedCLIP-SAMv2 release",
    ),
    "source-checkpoint": Asset(
        DEFAULT_SOURCE_CHECKPOINT, "all ReDi-FM methods, source_only", manual="ResUNet-34 trained on domain A"
    ),
    "pretrained-resunet": Asset(
        PRETRAINED_RESUNET_CHECKPOINT, "source_training", manual="ResUNet-34 initialization"
    ),
    "data": Asset(DATA_DIR, "all methods", manual="OD/OC benchmark, domains A-E"),
}


def check() -> bool:
    ok = True
    for name, asset in ASSETS.items():
        present = asset.path.exists()
        ok &= present
        how = "downloadable" if asset.fetch else f"manual: {asset.manual}"
        print(f"[{'ok' if present else 'missing'}] {name:<19} {asset.path}")
        print(f"    used by {asset.used_by}; {how}")
    return ok


def main() -> None:
    downloadable = [name for name, asset in ASSETS.items() if asset.fetch]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("assets", nargs="*", choices=downloadable, help="default: all downloadable assets")
    parser.add_argument("--check", action="store_true", help="only report which assets are present")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    for name in args.assets or downloadable:
        asset = ASSETS[name]
        if asset.path.exists() and not args.force:
            print(f"{name}: already present at {asset.path}")
            continue
        print(f"{name}: downloading to {asset.path}")
        asset.fetch(asset.path)


if __name__ == "__main__":
    main()
