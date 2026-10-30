#!/usr/bin/env bash
# nnU-Net 2D source-only baseline on the prostate benchmark: source domain A
# (BIDMC), target domains B-F, no adaptation.  ResUNet-34 source-only is run too,
# as the reference baseline of 0140.pdf Table 1.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
path() { $PY -c "import sys; sys.path.insert(0, 'src'); from common.paths import $1 as p; print(p)"; }
NNUNET=$(path PROSTATE_NNUNET_SOURCE_CHECKPOINT)
RESUNET=$(path PROSTATE_RESUNET_SOURCE_CHECKPOINT)

if [ ! -f "$NNUNET" ]; then
  $PY src/source_training/train.py --task prostate --arch nnunet --output "$NNUNET"
fi
if [ ! -f "$RESUNET" ]; then
  $PY src/source_training/train.py --task prostate --arch resunet34 --output "$RESUNET"
fi
mkdir -p results/nnunet_only results/source_only
$PY src/nnunet_only/evaluate.py --task prostate
$PY src/source_only/evaluate.py --task prostate --arch resunet34
