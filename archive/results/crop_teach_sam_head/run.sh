#!/usr/bin/env bash
# OD/OC: the 1.4M full-image head (hidden 256) on SAM ViT-H's image encoder instead of DINOv3.
# Same head recipe as the original DINOv3 head (1024 px, 40 epochs, batch 8, lr 1e-3).
# Then: head only; the SAM head places crops for Crop-Teach (paper loss, native-res R34 crop
# model, R34 target); and the same crops with no adaptation. Waits for the given PID first.
set -euo pipefail
cd "$(dirname "$0")/../.."
P=.venv/bin/python
OUT=results/crop_teach_sam_head
HEAD="$($P src/common/paths.py CHECKPOINT_DIR)/odoc_sam_vith_head_full_image_h256_domain_A.pth"
SAM="--dinov3-arch sam_vit_h --full-head-checkpoint $HEAD"
run() { local name=$1; shift; $P "$@" --output $OUT/$name.json > $OUT/$name.log 2>&1; }
if [ -n "${1:-}" ]; then while kill -0 "$1" 2> /dev/null; do sleep 30; done; fi
[ -f "$HEAD" ] || $P src/training/train_dinov3_head.py --task odoc --full-image --dinov3-arch sam_vit_h \
  --hidden 256 --crop-resolution 1024 --output "$HEAD" > $OUT/train_odoc_sam_head.log 2>&1
run odoc_no_adapt_sam_head_only   src/evaluate.py --method dinov3_single --task odoc --seed 0 $SAM
run odoc_sam_head_cnn_target      src/evaluate.py --method redifm_dinov3_nnunet_crop --loss paper --crop-region all \
                                  --seed 0 --task odoc $SAM
run odoc_no_adapt_sam_head_crop_cnn src/evaluate.py --method dinov3_crop --task odoc --seed 0 $SAM
