#!/usr/bin/env python3
"""Run one code base's command line (crop_teach/adapt.py or refactor_crop_teach/run.py) with an
optional replacement for SAM's image encoder. Used by compare_with_crop_teach.py, one process per
code base (both have modules named config, data, ...).

    python drive.py <code folder> <script> --sam real|cached|stub --cache <folder> -- <script args>

--sam real     the genuine encoder (one SAM ViT-H forward per image: minutes on a CPU)
--sam cached   the genuine encoder, each output saved under --cache by a hash of its input; an input
               already seen is answered from the cache. The second code base must give SAM exactly
               the inputs the first one did, otherwise this raises.
--sam stub     a cheap fixed random projection with SAM's output shape (exercises every other step)
"""

from __future__ import annotations

import argparse
import hashlib
import runpy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def input_key(images: torch.Tensor) -> str:
    return hashlib.sha1(images.detach().float().cpu().numpy().tobytes()).hexdigest()


def cached_forward(forward, cache: Path, must_hit: bool):
    def wrapped(images: torch.Tensor) -> torch.Tensor:
        path = cache / f"{input_key(images)}.pt"
        if path.is_file():
            return torch.load(path).to(device=images.device, dtype=images.dtype)
        if must_hit:
            raise RuntimeError("SAM got an input the other code base never gave it")
        output = forward(images)
        cache.mkdir(parents=True, exist_ok=True)
        torch.save(output.cpu(), path)
        return output
    return wrapped


class StubEncoder(nn.Module):
    """1024 x 1024 image -> 256 x 64 x 64, by a fixed random patch projection."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.weight = nn.Parameter(torch.randn(256, 3, 16, 16, generator=generator) / 30, requires_grad=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tanh(F.conv2d(images, self.weight.to(images.dtype), stride=16))


def patch_sam(mode: str, cache: Path, must_hit: bool) -> None:
    import segment_anything

    build = segment_anything.sam_model_registry["vit_h"]

    def patched(checkpoint=None):
        if mode == "stub":
            sam = nn.Module()
            sam.image_encoder = StubEncoder()
            return sam
        sam = build(checkpoint=checkpoint)
        if mode == "cached":
            sam.image_encoder.forward = cached_forward(sam.image_encoder.forward, cache, must_hit)
        return sam

    segment_anything.sam_model_registry["vit_h"] = patched


def main() -> None:
    arguments = sys.argv[1:]
    split = arguments.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("script")
    parser.add_argument("--sam", choices=["real", "cached", "stub"], default="real")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--must-hit", action="store_true")
    args = parser.parse_args(arguments[:split])
    if args.sam != "real":
        patch_sam(args.sam, args.cache, args.must_hit)
    sys.path.insert(0, str(args.folder.resolve()))
    sys.argv = [args.script, *arguments[split + 1:]]
    runpy.run_path(str(args.folder / args.script), run_name="__main__")


if __name__ == "__main__":
    main()
