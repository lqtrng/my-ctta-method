#!/usr/bin/env bash
# ReDi-FM/DINOv3 with nnU-Net on the prostate benchmark: source domain A (BIDMC),
# continual stream B -> C -> D -> E -> F, no reset, Do. A = mean over B-F.
# Trains the prostate nnU-Net source model and DINOv3 head first if missing.
# Prerequisite (one-off): .venv/bin/python src/download/download.py dinov3 prostate
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
path() { $PY -c "import sys; sys.path.insert(0, 'src'); from common.paths import $1 as p; print(p)"; }
SOURCE=$(path PROSTATE_NNUNET_SOURCE_CHECKPOINT)
HEAD=$(path PROSTATE_DINOV3_HEAD_CHECKPOINT)
SEEDS=${SEEDS:-"0"}

if [ ! -f "$SOURCE" ]; then
  $PY src/source_training/train.py --task prostate --arch nnunet --output "$SOURCE"
fi
if [ ! -f "$HEAD" ]; then
  $PY src/redifm_dinov3/train_head.py --task prostate --output "$HEAD"
fi
mkdir -p results/redifm_dinov3_nnunet
for seed in $SEEDS; do
  $PY src/redifm_dinov3_nnunet/adapt.py --task prostate --seed "$seed" \
    --output "results/redifm_dinov3_nnunet/prostate_all_domains_seed$seed.json"
done
