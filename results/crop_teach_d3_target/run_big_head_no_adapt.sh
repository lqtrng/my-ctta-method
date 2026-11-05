#!/usr/bin/env bash
# No adaptation with the 20M DINOv3 full-image head (--hidden 1024):
#   head only (dinov3_single), and head places crops -> R34 native-res crop model (dinov3_crop, OD/OC only).
set -euo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
OUT=results/crop_teach_d3_target
CK="$($P src/common/paths.py CHECKPOINT_DIR)"
while pgrep -f "run_big_head.sh" > /dev/null; do sleep 60; done
for t in odoc polyp; do
  $P src/evaluate.py --method dinov3_single --task $t --seed 0 \
    --full-head-checkpoint "$CK/${t}_dinov3_vitl16_head_full_image_h1024_domain_A.pth" \
    --output $OUT/${t}_no_adapt_head_only_h1024.json > $OUT/${t}_no_adapt_head_only_h1024.log 2>&1
done
$P src/evaluate.py --method dinov3_crop --task odoc --seed 0 --crop-region all --crop-from original \
  --full-head-checkpoint "$CK/odoc_dinov3_vitl16_head_full_image_h1024_domain_A.pth" \
  --crop-checkpoint "$($P src/common/paths.py RESUNET_SHARP_CROP_CHECKPOINT)" \
  --output $OUT/odoc_no_adapt_all_nativeres_h1024.json > $OUT/odoc_no_adapt_all_nativeres_h1024.log 2>&1
