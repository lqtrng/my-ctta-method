#!/usr/bin/env python3
"""Put every file the method reads in place: data/, models/ (SAM ViT-H) and checkpoints/.

    python fetch_assets.py                     # from the asset folder found automatically (below)
    python fetch_assets.py --from <folder>     # from a given asset folder
    python fetch_assets.py --raw <folder>      # build data/ from the raw datasets instead
    python fetch_assets.py --check             # only report what is present
    python fetch_assets.py --bundle <folder>   # stage data/ + checkpoints/ for a Kaggle dataset upload

An asset folder holds data/<problem>/domain_<letter>/{Images,Masks} and checkpoints/<problem>/*.pth
(crop_teach/'s runs/checkpoints/ layout is accepted too). It is looked for in this order: --from,
$CT_ASSETS, any dataset attached under /kaggle/input with both folders, then ../crop_teach.
Files are hard-linked when possible (no extra disk space), otherwise copied. SAM ViT-H is
downloaded from its official URL when no asset folder has it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

import config


def place(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def place_tree(source: Path, destination: Path) -> int:
    files = [file for file in sorted(source.rglob("*")) if file.is_file() and "__pycache__" not in file.parts]
    for file in files:
        place(file, destination / file.relative_to(source))
    return len(files)


def checkpoints_in(folder: Path) -> Path | None:
    for candidate in (folder / "checkpoints", folder / "runs" / "checkpoints"):
        if candidate.is_dir():
            return candidate
    return None


def find_asset_folder(given: Path | None) -> Path | None:
    candidates = [given] if given else []
    if os.environ.get("CT_ASSETS"):
        candidates.append(Path(os.environ["CT_ASSETS"]))
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        for pattern in ("*/data", "*/*/data", "*/*/*/data"):
            candidates += sorted(path.parent for path in kaggle.glob(pattern) if path.is_dir())
    candidates.append(config.HERE.parent / "crop_teach")
    for folder in candidates:
        if folder and (folder / "data").is_dir() and checkpoints_in(folder):
            return folder
    if given:
        raise FileNotFoundError(f"{given} has no data/ and checkpoints/ folders")
    return None


def fetch_sam(folder: Path | None) -> None:
    if config.SAM_CHECKPOINT.is_file():
        return
    if folder and (folder / "models" / config.SAM_CHECKPOINT.name).is_file():
        place(folder / "models" / config.SAM_CHECKPOINT.name, config.SAM_CHECKPOINT)
        return
    print(f"Downloading SAM ViT-H (2.6 GB) from {config.SAM_URL}")
    partial = config.SAM_CHECKPOINT.with_suffix(".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.SAM_URL) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 22)
    partial.rename(config.SAM_CHECKPOINT)


def check() -> None:
    print(f"SAM ViT-H: {'ok' if config.SAM_CHECKPOINT.is_file() else 'MISSING'}")
    for name, problem in config.PROBLEMS.items():
        data = [d for d in problem.domains if (config.DATA_DIR / name / f"domain_{d}" / "Images").is_dir()]
        complete = [s for s in problem.domains
                    if all(config.checkpoint_path(name, s, kind).is_file() for kind in ("target", "crop", "head"))]
        print(f"{name}: data for domains {''.join(data) or '-'}; "
              f"checkpoints (target, crop, head) for sources {''.join(complete) or '-'}")


def bundle(staging: Path) -> None:
    files = place_tree(config.DATA_DIR, staging / "data") + place_tree(config.CHECKPOINT_DIR, staging / "checkpoints")
    print(f"Staged {files} files in {staging}. Upload: kaggle datasets create -p {staging} --dir-mode zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="folder", type=Path)
    parser.add_argument("--raw", type=Path, help="Raw datasets: od_oc/, prostate-6/, polyp-4/")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.bundle:
        bundle(args.bundle)
        return
    if not args.check:
        folder = find_asset_folder(args.folder)
        if args.raw:
            from prepare_data import prepare

            prepare(args.raw, config.DATA_DIR)
        if folder:
            print(f"Assets from {folder}")
            place_tree(folder / "data", config.DATA_DIR)
            place_tree(checkpoints_in(folder), config.CHECKPOINT_DIR)
        elif not args.raw:
            print("No asset folder found: pass --from <folder> or --raw <folder>, or set CT_ASSETS")
        fetch_sam(folder)
    check()


if __name__ == "__main__":
    main()
