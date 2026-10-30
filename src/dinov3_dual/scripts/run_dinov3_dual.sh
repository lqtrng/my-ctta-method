#!/usr/bin/env bash
# Two frozen DINOv3 models (full image in place of the source ResUNet, plus the
# crop teacher) fused with ReDi-FM's consensus weighting, on the full Domain-A
# target stream B -> C -> D -> E, with no ResUNet-34 and no adaptation.
# Prerequisite (one-off): .venv/bin/python src/download/download.py dinov3
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
FULL_HEAD=$($PY -c "import sys; sys.path.insert(0, 'src'); from redifm_dinov3.train_head import DEFAULT_FULL_IMAGE_OUTPUT as h; print(h)")
CROP_HEAD=$($PY -c "import sys; sys.path.insert(0, 'src'); from redifm_dinov3.adapt import DEFAULT_HEAD_CHECKPOINT as h; print(h)")

if [ ! -f "$FULL_HEAD" ]; then
  $PY src/redifm_dinov3/train_head.py --full-image --crop-resolution 1024 --output "$FULL_HEAD"
fi
if [ ! -f "$CROP_HEAD" ]; then
  $PY src/redifm_dinov3/train_head.py --output "$CROP_HEAD"
fi
mkdir -p results/dinov3_dual
$PY src/dinov3_dual/evaluate.py --output results/dinov3_dual/dinov3_dual_domain_A.json
