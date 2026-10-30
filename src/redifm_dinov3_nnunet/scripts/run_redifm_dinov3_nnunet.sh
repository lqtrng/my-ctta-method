#!/usr/bin/env bash
# ReDi-FM/DINOv3 with nnU-Net as source/target network on the full Domain-A
# protocol: continual stream B -> C -> D -> E, no reset, seed 0 (set SEEDS for more).
# Prerequisite (one-off): $PY src/download/download.py dinov3
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PYTHON:-.venv/bin/python}
CKPT=$($PY src/common/paths.py NNUNET_SOURCE_CHECKPOINT)
HEAD=$($PY src/common/paths.py DINOV3_HEAD_CHECKPOINT)
SEEDS=${SEEDS:-"0"}

if [ ! -f "$CKPT" ]; then
  $PY src/source_training/train.py --arch nnunet --output "$CKPT"
fi
if [ ! -f "$HEAD" ]; then
  $PY src/redifm_dinov3/train_head.py --output "$HEAD"
fi
mkdir -p results/redifm_dinov3_nnunet
for seed in $SEEDS; do
  $PY src/redifm_dinov3_nnunet/adapt.py --seed "$seed" \
    --output "results/redifm_dinov3_nnunet/all_domains_seed$seed.json"
done
