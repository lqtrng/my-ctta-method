#!/usr/bin/env bash
# Reproduce every result in README.md from a fresh clone, locally or in a Kaggle notebook.
#
#   ./reproduce.sh                    # every stage, in order
#   ./reproduce.sh setup assets       # only the named stages
#
# Stages:
#   setup      install the Python dependencies (requirements.txt)
#   assets     fetch data, foundation models and checkpoints (src/download/download.py)
#   odoc       OD/OC table: source only, nnU-Net only, ReDi-FM + SAM, ReDi-FM + DINOv3
#              (ResUNet-34 and nnU-Net), DINOv3 single and dual
#   prostate   prostate table: the same methods on the six-site prostate benchmark
#   long_term  long-term CTTA (3 rounds of B -> E); needs the odoc stage's source-only result
#
# A checkpoint missing from the asset bundle is trained on domain A before it is used.
# Every run is logged to MLflow (mlruns/) and saved as JSON under results/.
# Browse the runs with: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
#
# Environment:
#   PYTHON                 interpreter (default: .venv/bin/python if present, else python)
#   SEEDS                  seeds of ReDi-FM + DINOv3 on OD/OC (default "0 1 2";
#                          long_term default "0")
#   REDIFM_KAGGLE_DATASET  asset bundle id (default in src/common/paths.py)
#   REDIFM_MLFLOW=0        turn MLflow tracking off
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${PYTHON:-}" ]; then
  if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python; else PYTHON=python; fi
fi
export PYTHON MLFLOW_DISABLE_AGENT_HINT=1
PY=$PYTHON
path() { $PY src/common/paths.py "$1"; }
STAGES=${*:-setup assets odoc prostate long_term}

stage_setup() {
  # PyTorch with CUDA comes preinstalled on Kaggle; locally install it first (pytorch.org).
  if command -v uv >/dev/null; then
    uv pip install --python "$PY" -r requirements.txt
  else
    $PY -m pip install -r requirements.txt
  fi
}

stage_assets() {
  $PY src/download/download.py
  $PY src/download/download.py --check || true
}

# Train a checkpoint named in common/paths.py if the asset bundle did not provide it.
train_if_missing() {
  local file
  file=$(path "$1"); shift
  [ -f "$file" ] || "$@" --output "$file"
}

stage_odoc() {
  train_if_missing DEFAULT_SOURCE_CHECKPOINT $PY src/training/train_source.py
  train_if_missing NNUNET_SOURCE_CHECKPOINT $PY src/training/train_source.py --arch nnunet
  train_if_missing DINOV3_HEAD_CHECKPOINT $PY src/training/train_dinov3_head.py
  train_if_missing DINOV3_FULL_IMAGE_HEAD_CHECKPOINT \
    $PY src/training/train_dinov3_head.py --full-image --crop-resolution 1024

  $PY src/evaluate.py --method source_only
  $PY src/evaluate.py --method source_only --arch nnunet
  $PY src/evaluate.py --method dinov3_single
  $PY src/evaluate.py --method dinov3_dual
  $PY src/evaluate.py --method redifm_sam
  for seed in ${SEEDS:-0 1 2}; do
    $PY src/evaluate.py --method redifm_dinov3 --seed "$seed" \
      --output "results/redifm_dinov3/final_all_domains_seed$seed.json"
  done
  $PY src/common/report.py results/redifm_dinov3/final_all_domains_seed*.json
  $PY src/evaluate.py --method redifm_dinov3 --arch nnunet --seed 0 \
    --output results/redifm_dinov3_nnunet/all_domains_seed0.json
}

stage_prostate() {
  # Seed 0, as in the README table.
  train_if_missing PROSTATE_RESUNET_SOURCE_CHECKPOINT $PY src/training/train_source.py --task prostate
  train_if_missing PROSTATE_NNUNET_SOURCE_CHECKPOINT \
    $PY src/training/train_source.py --task prostate --arch nnunet
  train_if_missing PROSTATE_DINOV3_HEAD_CHECKPOINT $PY src/training/train_dinov3_head.py --task prostate

  $PY src/evaluate.py --method source_only --task prostate
  $PY src/evaluate.py --method source_only --task prostate --arch nnunet
  $PY src/evaluate.py --method redifm_sam --task prostate
  $PY src/evaluate.py --method redifm_dinov3 --task prostate --seed 0 \
    --output results/redifm_dinov3/prostate_all_domains_seed0.json
  $PY src/evaluate.py --method redifm_dinov3 --task prostate --arch nnunet --seed 0 \
    --output results/redifm_dinov3_nnunet/prostate_all_domains_seed0.json
}

stage_long_term() {
  for seed in ${SEEDS:-0}; do
    $PY src/evaluate.py --method redifm_dinov3 --rounds 3 --seed "$seed" \
      --output "results/redifm_dinov3/long_term_seed$seed.json"
  done
  $PY src/common/report.py results/redifm_dinov3/long_term_seed*.json
}

for stage in $STAGES; do
  case $stage in
    setup | assets | odoc | prostate | long_term) ;;
    *) echo "Unknown stage: $stage (setup, assets, odoc, prostate, long_term)" >&2; exit 2 ;;
  esac
done
for stage in $STAGES; do
  echo "=== $stage ==="
  "stage_$stage"
done
