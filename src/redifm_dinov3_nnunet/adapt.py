#!/usr/bin/env python3
"""ReDi-FM continual test-time adaptation with DINOv3, using nnU-Net instead of ResUNet-34.

Identical to ``redifm_dinov3/adapt.py`` (same DINOv3 crop teacher, decoder
head, objective and online protocol) except for the segmentation network: the
frozen source teacher and the adapted student are nnU-Net's 2D network
(``common.models.NnUNet2D``), trained on domain A with the repository's source
recipe (``source_training/train.py --arch nnunet``).  The DINOv3 crop is placed
from the nnU-Net source pseudo-label.

nnU-Net normalizes with InstanceNorm, so unlike ResUNet-34's BatchNorm there
are no running statistics: the online update acts through the weights only.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from redifm_dinov3.adapt import build_parser, main  # noqa: E402


if __name__ == "__main__":
    # The source checkpoint and output default to the nnU-Net ones of --task
    # (results/redifm_dinov3_nnunet/[prostate_]redifm_dinov3_nnunet_domain_A.json).
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(arch="nnunet")
    main(parser.parse_args())
