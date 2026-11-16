#!/usr/bin/env bash
# Crop-Teach with the new defaults: ReDi-FM's paper loss, native-res crop models, frozen
# batch-norm, lr 2e-5, all regions, seed 0. Compares crop placers / targets:
#   small = the 1.4M DINOv3 full-image head (the original), big = the 20M head (--hidden 1024).
# Polyp uses nnU (its only native-res crop model); OD/OC uses R34.
# Waits for the given PID (the running polyp 20M-head training) first.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
OUT=results/crop_teach_paper_loss
CK="$($P src/common/paths.py CHECKPOINT_DIR)"
BIG_POLYP="$CK/polyp_dinov3_vitl16_head_full_image_h1024_domain_A.pth"
BIG_ODOC="$CK/odoc_dinov3_vitl16_head_full_image_h1024_r1024_domain_A.pth"
E="src/evaluate.py --method redifm_dinov3_nnunet_crop --loss paper --crop-region all --seed 0"
run() { local name=$1; shift; $P "$@" --output $OUT/$name.json > $OUT/$name.log 2>&1; }
while kill -0 "$1" 2> /dev/null; do sleep 30; done

# Polyp (nnU native-res crop model).
run polyp_small_head_cnn_target         $E --task polyp
run polyp_big_head_cnn_target           $E --task polyp --full-head-checkpoint "$BIG_POLYP"
run polyp_big_head_head_target          $E --task polyp --full-head-checkpoint "$BIG_POLYP" --target dinov3_head
run polyp_no_adapt_big_head_only        src/evaluate.py --method dinov3_single --task polyp --seed 0 \
                                        --full-head-checkpoint "$BIG_POLYP"

# OD/OC (R34 native-res crop model). The 1.4M-head run needs no new head, so it goes first.
run odoc_small_head_cnn_target          $E --task odoc
$P src/training/train_dinov3_head.py --task odoc --full-image --hidden 1024 --crop-resolution 1024 \
    --output "$BIG_ODOC" > $OUT/train_odoc_h1024_r1024.log 2>&1 \
  || $P src/training/train_dinov3_head.py --task odoc --full-image --hidden 1024 --crop-resolution 1024 \
    --batch-size 4 --output "$BIG_ODOC" > $OUT/train_odoc_h1024_r1024_batch4.log 2>&1
run odoc_big_head_cnn_target            $E --task odoc --full-head-checkpoint "$BIG_ODOC"
run odoc_big_head_head_target           $E --task odoc --full-head-checkpoint "$BIG_ODOC" --target dinov3_head
run odoc_no_adapt_big_head_only         src/evaluate.py --method dinov3_single --task odoc --seed 0 \
                                        --full-head-checkpoint "$BIG_ODOC"
run odoc_no_adapt_big_head_crop_cnn     src/evaluate.py --method dinov3_crop --task odoc --seed 0 \
                                        --full-head-checkpoint "$BIG_ODOC"
