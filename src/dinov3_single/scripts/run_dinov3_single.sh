#!/usr/bin/env bash
# One frozen full-image DINOv3 on the full Domain-A target stream B -> C -> D -> E,
# with no ResUNet-34 and no adaptation.
# Prerequisite (one-off): $PY src/download/download.py dinov3
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PYTHON:-.venv/bin/python}
FULL_HEAD=$($PY src/common/paths.py DINOV3_FULL_IMAGE_HEAD_CHECKPOINT)

if [ ! -f "$FULL_HEAD" ]; then
  $PY src/redifm_dinov3/train_head.py --full-image --crop-resolution 1024 --output "$FULL_HEAD"
fi
mkdir -p results/dinov3_single
$PY src/dinov3_single/evaluate.py --output results/dinov3_single/dinov3_single_domain_A.json
