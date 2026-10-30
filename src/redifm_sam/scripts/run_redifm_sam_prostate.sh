#!/usr/bin/env bash
# ReDi-FM/SAM (ResUNet-34) on the prostate benchmark: source domain A (BIDMC),
# continual stream B -> C -> D -> E -> F, no reset, Do. A = mean over B-F.
# Trains the prostate ResUNet-34 source model first if missing.
# Prerequisite (one-off): .venv/bin/python src/download/download.py sam prostate
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
SOURCE=$($PY -c "import sys; sys.path.insert(0, 'src'); from common.paths import PROSTATE_RESUNET_SOURCE_CHECKPOINT as p; print(p)")

if [ ! -f "$SOURCE" ]; then
  $PY src/source_training/train.py --task prostate --arch resunet34 --output "$SOURCE"
fi
$PY src/redifm_sam/adapt.py --task prostate "$@"
