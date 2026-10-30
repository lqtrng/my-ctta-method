#!/usr/bin/env python3
"""Source-only baseline with nnU-Net's 2D network instead of ResUNet-34.

Identical to ``source_only/evaluate.py`` (no parameter updates, same target
domains, preprocessing and per-image DSC) except for the network:
``common.models.NnUNet2D`` trained on domain A with the repository's source
recipe (``source_training/train.py --arch nnunet``).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.paths import RESULTS_DIR  # noqa: E402
from source_only.evaluate import build_parser, main  # noqa: E402


if __name__ == "__main__":
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(arch="nnunet")
    args = parser.parse_args()
    if args.output is None:
        prefix = "" if args.task == "odoc" else f"{args.task}_"
        args.output = RESULTS_DIR / f"nnunet_only/{prefix}nnunet_only_domain_A.json"
    main(args, method="nnunet_only")
