#!/usr/bin/env bash
# The remaining 20M-head (--hidden 1024) runs.
#   R34 target: the 20M head places crops (frozen), the R34 full-image CNN adapts.
#   d3target:  the 20M head places crops and a trainable copy of it adapts.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
OUT=results/crop_teach_d3_target
CK="$($P src/common/paths.py CHECKPOINT_DIR)"
head_of() { echo "$CK/${1}_dinov3_vitl16_head_full_image_h1024_domain_A.pth"; }
E="src/evaluate.py --method redifm_dinov3_nnunet_crop --crop-region all --seed 0"
# OD/OC, R34 target, R34 native-res crop model (as CT1).
$P $E --task odoc --full-head-checkpoint "$(head_of odoc)" --crop-from original \
  --foundation-checkpoint "$($P src/common/paths.py RESUNET_SHARP_CROP_CHECKPOINT)" \
  --output $OUT/odoc_all_nativeres_h1024_r34target_seed0.json > $OUT/odoc_all_nativeres_h1024_r34target_seed0.log 2>&1
# Polyp: train the head, then D3 target, head only, and R34 target.
$P src/training/train_dinov3_head.py --task polyp --full-image --hidden 1024 --output "$(head_of polyp)" \
  > $OUT/train_polyp_h1024.log 2>&1
$P $E --task polyp --target dinov3_head --full-head-checkpoint "$(head_of polyp)" \
  --output $OUT/polyp_all_lowres_h1024_seed0.json > $OUT/polyp_all_lowres_h1024_seed0.log 2>&1
$P src/evaluate.py --method dinov3_single --task polyp --seed 0 --full-head-checkpoint "$(head_of polyp)" \
  --output $OUT/polyp_no_adapt_head_only_h1024.json > $OUT/polyp_no_adapt_head_only_h1024.log 2>&1
# Polyp, R34 target: no R34 native-res crop model exists for polyp, so low-res (as CT3).
$P $E --task polyp --full-head-checkpoint "$(head_of polyp)" \
  --output $OUT/polyp_all_lowres_h1024_r34target_seed0.json > $OUT/polyp_all_lowres_h1024_r34target_seed0.log 2>&1
