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
#   SEEDS                  seeds of ReDi-FM + DINOv3 on OD/OC (default "0 1 2")
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

stage_odoc() {
  local source
  source=$(path DEFAULT_SOURCE_CHECKPOINT)
  if [ ! -f "$source" ]; then
    $PY src/source_training/train.py --output "$source"
  fi
  $PY src/source_only/evaluate.py
  bash src/nnunet_only/scripts/run_nnunet_only.sh
  $PY src/redifm_sam/adapt.py
  SEEDS=${SEEDS:-"0 1 2"} bash src/redifm_dinov3/scripts/run_dinov3_all_domains.sh
  bash src/redifm_dinov3_nnunet/scripts/run_redifm_dinov3_nnunet.sh
  bash src/dinov3_single/scripts/run_dinov3_single.sh
  bash src/dinov3_dual/scripts/run_dinov3_dual.sh
}

stage_prostate() {
  # Seed 0, as in the README table.
  SEEDS=0 bash src/nnunet_only/scripts/run_nnunet_only_prostate.sh
  SEEDS=0 bash src/redifm_dinov3/scripts/run_dinov3_prostate.sh
  SEEDS=0 bash src/redifm_dinov3_nnunet/scripts/run_redifm_dinov3_nnunet_prostate.sh
  bash src/redifm_sam/scripts/run_redifm_sam_prostate.sh
}

stage_long_term() {
  SEEDS=0 bash src/redifm_dinov3/scripts/run_dinov3_long_term.sh
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
