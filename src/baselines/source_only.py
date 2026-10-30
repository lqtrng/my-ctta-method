"""Source Only: the domain-A model, never updated (the paper's "Source Only" row).

The checkpoint is loaded once, placed in evaluation mode, and predicts every
target image without any parameter or BN-statistic update.  ``--arch nnunet``
uses nnU-Net's 2D network instead of ResUNet-34 (method ``source_only_nnunet``,
the README's "nnU-Net only" row).

    python src/evaluate.py --method source_only [--task prostate] [--arch nnunet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from common.models import ARCHITECTURES, build_network
from common.paths import SOURCE_CHECKPOINTS, require_file
from common.protocol import image_to_tensor
from common.tasks import Task


# 0140.pdf Table 1, Source Only, source domain A.
PAPER_DO_A = {"odoc": 53.59, "prostate": 51.02}


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default="resunet34")
    parser.add_argument("--checkpoint", type=Path, help="Default: the domain-A checkpoint of --task and --arch")


def build(args: argparse.Namespace, task: Task):
    args.checkpoint = require_file(
        (args.checkpoint or SOURCE_CHECKPOINTS[(task.name, args.arch)]).resolve(), "Source checkpoint")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_network(args.arch, task.num_classes, task.image_size)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()
    print(f"Checkpoint: {args.checkpoint} ({args.arch}) on {device}")

    @torch.inference_mode()
    def segment(image: Image.Image) -> tuple[torch.Tensor, dict]:
        return model(image_to_tensor(image, device)).argmax(dim=1)[0], {}

    return segment
