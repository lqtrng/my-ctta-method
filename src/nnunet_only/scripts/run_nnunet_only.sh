#!/usr/bin/env bash
# nnU-Net 2D source-only baseline on the Domain-A target stream B, C, D, E.
# Trains the nnU-Net source model on domain A first if it is missing.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
CKPT=$($PY -c "import sys; sys.path.insert(0, 'src'); from common.paths import NNUNET_SOURCE_CHECKPOINT as c; print(c)")

if [ ! -f "$CKPT" ]; then
  $PY src/source_training/train.py --arch nnunet --output "$CKPT"
fi
mkdir -p results/nnunet_only
$PY src/nnunet_only/evaluate.py
