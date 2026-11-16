#!/usr/bin/env bash
# Crop-Teach with a 20M-parameter DINOv3 full-image head (--hidden 1024).
# The same 20M head places the crops (frozen) and starts the target (a trainable copy).
# Per problem: train the head on domain A (if missing), then evaluate.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
OUT=results/crop_teach_d3_target
CK="$($P src/common/paths.py CHECKPOINT_DIR)"
while pgrep -f "src/training/train_dinov3_head.py" > /dev/null; do sleep 30; done
E="src/evaluate.py --method redifm_dinov3_nnunet_crop --target dinov3_head --crop-region all --seed 0"
for t in odoc polyp; do
  HEAD="$CK/${t}_dinov3_vitl16_head_full_image_h1024_domain_A.pth"
  [ -f "$HEAD" ] || $P src/training/train_dinov3_head.py --task $t --full-image --hidden 1024 \
    --output "$HEAD" > $OUT/train_${t}_h1024.log 2>&1
  $P $E --task $t --full-head-checkpoint "$HEAD" \
    --output $OUT/${t}_all_lowres_h1024_seed0.json > $OUT/${t}_all_lowres_h1024_seed0.log 2>&1
done
