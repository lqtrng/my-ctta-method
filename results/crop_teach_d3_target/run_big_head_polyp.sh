#!/usr/bin/env bash
# Polyp runs with the 20M DINOv3 full-image head (--hidden 1024), at polyp's head resolution (704,
# as in reproduce.sh). Waits for the given PID (the running OD/OC evaluation) first.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
OUT=results/crop_teach_d3_target
CK="$($P src/common/paths.py CHECKPOINT_DIR)"
HEAD="$CK/polyp_dinov3_vitl16_head_full_image_h1024_domain_A.pth"
E="src/evaluate.py --method redifm_dinov3_nnunet_crop --crop-region all --seed 0 --task polyp --full-head-checkpoint $HEAD"
while kill -0 "$1" 2> /dev/null; do sleep 30; done
$P src/training/train_dinov3_head.py --task polyp --full-image --hidden 1024 --crop-resolution 704 \
  --output "$HEAD" > $OUT/train_polyp_h1024.log 2>&1
# D3 target, head only, then R34 target (low-res crop model: no R34 native-res one exists for polyp).
$P $E --target dinov3_head --output $OUT/polyp_all_lowres_h1024_seed0.json > $OUT/polyp_all_lowres_h1024_seed0.log 2>&1
$P src/evaluate.py --method dinov3_single --task polyp --seed 0 --full-head-checkpoint "$HEAD" \
  --output $OUT/polyp_no_adapt_head_only_h1024.json > $OUT/polyp_no_adapt_head_only_h1024.log 2>&1
$P $E --output $OUT/polyp_all_lowres_h1024_r34target_seed0.json > $OUT/polyp_all_lowres_h1024_r34target_seed0.log 2>&1
