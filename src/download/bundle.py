#!/usr/bin/env python3
"""The asset bundle: a private Kaggle dataset with everything that has no public download.

It holds exactly the files the methods read and nothing else:

* ``data/od_oc/``: the OD/OC benchmark, domains A-E (Drishti-GS as a zip with
  only its own members);
* ``data/prostate-6/``: the prepared six-site prostate slices (``prepare_prostate.py``);
* ``checkpoints/``: this project's trained files (source models, DINOv3 heads,
  the ResUNet-34 initialization).

Publishing (once, by whoever owns the files):

    python src/download/bundle.py build             # stage the files in .cache/kaggle-bundle
    kaggle datasets create -p .cache/kaggle-bundle --dir-mode zip      # first upload (private)
    kaggle datasets version -p .cache/kaggle-bundle --dir-mode zip -m "..."   # later updates

Fetching is done by ``download.py``: on Kaggle the attached dataset is linked
from ``/kaggle/input``; elsewhere it is downloaded once into ``.cache/`` with
the Kaggle CLI.  Either way the files are linked into ``data/`` and
``CHECKPOINT_DIR``, so every path in ``common/paths.py`` stays the same.
The dataset id is ``common.paths.KAGGLE_DATASET`` (``REDIFM_KAGGLE_DATASET``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import (  # noqa: E402
    CHECKPOINT_DIR,
    DATA_DIR,
    DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    DINOV3_HEAD_CHECKPOINT,
    DRISHTI_ARCHIVE,
    KAGGLE_DATASET,
    ON_KAGGLE,
    PRETRAINED_RESUNET_CHECKPOINT,
    PROJECT_DIR,
    PROSTATE_DINOV3_HEAD_CHECKPOINT,
    PROSTATE_DIR,
    SOURCE_CHECKPOINTS,
)


DEFAULT_STAGING_DIR = PROJECT_DIR / ".cache/kaggle-bundle"
DOWNLOAD_DIR = PROJECT_DIR / ".cache/redifm-assets"
# Folders under data/od_oc that common/data.py reads (Drishti-GS is the archive).
OD_OC_FOLDERS = (
    "domain_A_RIM_ONE_r3/RIM-ONE-r3/training",
    "domain_A_RIM_ONE_r3/RIM-ONE-r3/testing",
    "domain_B_REFUGE_train/train/Images",
    "domain_B_REFUGE_train/train/Masks",
    "domain_C_ORIGA/Images",
    "domain_C_ORIGA/Masks",
    "domain_D_REFUGE_val_test/val/Images",
    "domain_D_REFUGE_val_test/val/Masks",
    "domain_D_REFUGE_val_test/test/Images",
    "domain_D_REFUGE_val_test/test/Masks",
)
CHECKPOINTS = (
    *SOURCE_CHECKPOINTS.values(),
    DINOV3_HEAD_CHECKPOINT,
    DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    PROSTATE_DINOV3_HEAD_CHECKPOINT,
    PRETRAINED_RESUNET_CHECKPOINT,
)


def bundle_path(path: Path) -> str:
    """Location of a local asset inside the bundle."""
    if path.is_relative_to(CHECKPOINT_DIR):
        return f"checkpoints/{path.relative_to(CHECKPOINT_DIR)}"
    return str(path.relative_to(PROJECT_DIR))


# --- building ---------------------------------------------------------------


def _place(source: Path, destination: Path) -> None:
    """Hard-link when possible (no extra disk space), otherwise copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _place_tree(source: Path, destination: Path) -> None:
    for file in sorted(source.rglob("*")):
        if file.is_file():
            _place(file, destination / file.relative_to(source))


def build(staging: Path) -> None:
    missing = [path for path in CHECKPOINTS if not path.is_file()]
    if missing:
        print("Not in the bundle (train them first, or reproduce.sh will retrain them):")
        for path in missing:
            print(f"  {path}")

    for folder in OD_OC_FOLDERS:
        _place_tree(DATA_DIR / folder, staging / bundle_path(DATA_DIR / folder))
    drishti = staging / bundle_path(DRISHTI_ARCHIVE)
    if not drishti.exists():
        drishti.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(DRISHTI_ARCHIVE) as source, ZipFile(drishti, "w", ZIP_STORED) as target:
            for member in source.infolist():
                if member.filename.startswith("Drishti-GS/"):
                    target.writestr(member, source.read(member))
    for domain in sorted(PROSTATE_DIR.glob("domain_*")):
        for kind in ("Images", "Masks"):
            _place_tree(domain / kind, staging / bundle_path(domain / kind))
    _place(PROSTATE_DIR / "manifest.json", staging / bundle_path(PROSTATE_DIR / "manifest.json"))
    for path in CHECKPOINTS:
        if path.is_file():
            _place(path, staging / bundle_path(path))

    metadata = {"title": "ReDi-FM assets", "id": KAGGLE_DATASET, "licenses": [{"name": "other"}]}
    (staging / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    size = sum(file.stat().st_size for file in staging.rglob("*") if file.is_file())
    print(f"Staged {size / 1e9:.1f} GB in {staging} for Kaggle dataset {KAGGLE_DATASET}.")
    print("Keep the dataset private: the OD/OC datasets have their own licenses.")
    print(f"Upload: kaggle datasets create -p {staging} --dir-mode zip")


# --- fetching ---------------------------------------------------------------


def bundle_root() -> Path:
    """The bundle's files; a failed lookup is not retried for every asset."""
    global _BUNDLE_ROOT
    if isinstance(_BUNDLE_ROOT, Exception):
        raise FileNotFoundError(str(_BUNDLE_ROOT))
    if _BUNDLE_ROOT is None:
        try:
            _BUNDLE_ROOT = _find_bundle_root()
        except FileNotFoundError as error:
            _BUNDLE_ROOT = error
            raise
    return _BUNDLE_ROOT


_BUNDLE_ROOT: Path | Exception | None = None


def _find_bundle_root() -> Path:
    """The bundle's files: the attached Kaggle dataset, or a one-off download."""
    slug = KAGGLE_DATASET.split("/")[-1]
    if ON_KAGGLE:
        inputs = Path("/kaggle/input")
        for candidate in (inputs / slug, inputs / "datasets" / KAGGLE_DATASET, *inputs.glob(f"*/*/{slug}")):
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Attach the Kaggle dataset {KAGGLE_DATASET} to this notebook (Add Input), "
            "or set REDIFM_KAGGLE_DATASET to its id."
        )
    if not (DOWNLOAD_DIR / "checkpoints").is_dir():
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cli = Path(sys.executable).with_name("kaggle")
        command = [str(cli) if cli.is_file() else "kaggle", "datasets", "download", KAGGLE_DATASET]
        try:
            subprocess.run([*command, "-p", str(DOWNLOAD_DIR), "--unzip"], check=True)
        except (OSError, subprocess.CalledProcessError) as error:
            raise FileNotFoundError(
                f"Could not download the asset bundle {KAGGLE_DATASET} ({error}). It needs Kaggle "
                "API credentials (~/.kaggle/kaggle.json) with access to the dataset."
            ) from error
    return DOWNLOAD_DIR


def _locate(relative: str) -> Path:
    """A bundle file or folder; folders uploaded with --dir-mode zip may arrive as archives."""
    for root in (bundle_root(), DOWNLOAD_DIR):
        if (root / relative).exists():
            return root / relative
    parts = Path(relative).parts
    for depth in range(1, len(parts) + 1):
        prefix = Path(*parts[:depth])
        archive = bundle_root() / f"{prefix}.zip"
        if archive.is_file():
            with ZipFile(archive) as opened:
                opened.extractall(DOWNLOAD_DIR / prefix)
            rest = Path(*parts[depth:]) if depth < len(parts) else Path()
            # The archive holds either the folder's contents or the folder itself.
            for candidate in (DOWNLOAD_DIR / prefix / rest, DOWNLOAD_DIR / prefix / prefix.name / rest):
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"{relative} is not in the asset bundle {KAGGLE_DATASET}")


def link_from_bundle(path: Path) -> None:
    """Make ``path`` available by linking it to its copy in the bundle."""
    source = _locate(bundle_path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    path.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--output", type=Path, default=DEFAULT_STAGING_DIR)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
