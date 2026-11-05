#!/usr/bin/env bash
# OD/OC with the 20M DINOv3 full-image head (--hidden 1024) retrained at 1024 px, the resolution of the
# original OD/OC head (reproduce.sh). The first 20M head was trained at 512 px by mistake.
# Waits for the given PID (the polyp queue) first.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
OUT=results/crop_teach_d3_target
CK="$($P src/common/paths.py CHECKPOINT_DIR)"
HEAD="$CK/odoc_dinov3_vitl16_head_full_image_h1024_r1024_domain_A.pth"
SHARP="$($P src/common/paths.py RESUNET_SHARP_CROP_CHECKPOINT)"
while kill -0 "$1" 2> /dev/null; do sleep 30; done
# Batch 8 as the original; batch 4 only if 8 does not fit in GPU memory.
$P src/training/train_dinov3_head.py --task odoc --full-image --hidden 1024 --crop-resolution 1024 \
    --output "$HEAD" > $OUT/train_odoc_h1024_r1024.log 2>&1 \
  || $P src/training/train_dinov3_head.py --task odoc --full-image --hidden 1024 --crop-resolution 1024 \
    --batch-size 4 --output "$HEAD" > $OUT/train_odoc_h1024_r1024_batch4.log 2>&1
E="src/evaluate.py --method redifm_dinov3_nnunet_crop --crop-region all --seed 0 --task odoc --full-head-checkpoint $HEAD"
# Crop-Teach: 20M head as target (low-res crop model), and R34 target (native-res crop model).
$P $E --target dinov3_head --output $OUT/odoc_all_lowres_h1024_r1024_seed0.json \
  > $OUT/odoc_all_lowres_h1024_r1024_seed0.log 2>&1
$P $E --crop-from original --foundation-checkpoint "$SHARP" \
  --output $OUT/odoc_all_nativeres_h1024_r1024_r34target_seed0.json > $OUT/odoc_all_nativeres_h1024_r1024_r34target_seed0.log 2>&1
# No adaptation: head only, and head-placed crops -> R34 native-res crop model.
$P src/evaluate.py --method dinov3_single --task odoc --seed 0 --full-head-checkpoint "$HEAD" \
  --output $OUT/odoc_no_adapt_head_only_h1024_r1024.json > $OUT/odoc_no_adapt_head_only_h1024_r1024.log 2>&1
$P src/evaluate.py --method dinov3_crop --task odoc --seed 0 --crop-region all --crop-from original \
  --full-head-checkpoint "$HEAD" --crop-checkpoint "$SHARP" \
  --output $OUT/odoc_no_adapt_all_nativeres_h1024_r1024.json > $OUT/odoc_no_adapt_all_nativeres_h1024_r1024.log 2>&1
