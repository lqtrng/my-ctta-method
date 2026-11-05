#!/usr/bin/env bash
# Long-term CTTA (3 passes over the target stream, never reset) for the current best
# method of each task. Options copied from the configs of the best 1-round runs:
#   OD/OC    O5  results/sharp_crop/adapt_all_seed0.json                         (82.66)
#   prostate P2  results/redifm_dinov3_nnunet_crop/prostate_final_seed0.json     (78.85 +- 0.33)
#   polyp    Y2  results/polyp_nnunet_sharp/adapt_all_seed0.json                 (80.42)
# Run from the ReDi-FM folder:  bash results/long_term/run_long_term.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
M=/run/media/lqtrung/1tb-ssd/Model/ReDi-FM
OUT=results/long_term
mkdir -p "$OUT"
PY=.venv/bin/python

echo "== OD/OC (O5), 3 rounds"
$PY src/evaluate.py --method redifm_dinov3_nnunet_crop --task odoc --rounds 3 --seed 0 \
  --arch resunet34 --learning-rate 2e-5 --crop-region all --crop-from original \
  --source-checkpoint      $M/dinov3_vitl16_odoc_head_full_image_domain_A.pth \
  --full-head-checkpoint   $M/dinov3_vitl16_odoc_head_full_image_domain_A.pth \
  --foundation-checkpoint  $M/domain_A_ResUNet34_crop_original.pth \
  --target-checkpoint      $M/domain_A_ResUNet34_last_20261017_215955.pth \
  --output $OUT/odoc_O5_long_term_seed0.json 2>&1 | tee $OUT/odoc_O5_long_term_seed0.log

echo "== prostate (P2), 3 rounds"
$PY src/evaluate.py --method redifm_dinov3_nnunet_crop --task prostate --rounds 3 --seed 0 \
  --arch nnunet_official --learning-rate 2e-5 --crop-region largest \
  --source-checkpoint      $M/prostate_dinov3_vitl16_head_full_image_domain_A.pth \
  --full-head-checkpoint   $M/prostate_dinov3_vitl16_head_full_image_domain_A.pth \
  --foundation-checkpoint  $M/prostate_domain_A_nnUNet2D_crop.pth \
  --target-checkpoint      $M/prostate_domain_A_nnUNet2D_official.pth \
  --output $OUT/prostate_P2_long_term_seed0.json 2>&1 | tee $OUT/prostate_P2_long_term_seed0.log

echo "== polyp (Y2), 3 rounds"
$PY src/evaluate.py --method redifm_dinov3_nnunet_crop --task polyp --rounds 3 --seed 0 \
  --arch nnunet_official --cnn nnunet_official --learning-rate 2e-5 --crop-region all --crop-from original \
  --source-checkpoint      $M/polyp_dinov3_vitl16_head_full_image_domain_A.pth \
  --full-head-checkpoint   $M/polyp_dinov3_vitl16_head_full_image_domain_A.pth \
  --foundation-checkpoint  $M/polyp_domain_A_nnUNet2D_crop_original.pth \
  --target-checkpoint      $M/polyp_domain_A_nnUNet2D_official.pth \
  --output $OUT/polyp_Y2_long_term_seed0.json 2>&1 | tee $OUT/polyp_Y2_long_term_seed0.log

echo "== done"; grep -h -E 'Round|Do\. A|mean' $OUT/*_long_term_seed0.log | grep -v 'it/s' || true
