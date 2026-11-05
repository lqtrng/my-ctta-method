#!/usr/bin/env bash
# One problem's full experiment on this machine (RTX 5060, 8 GB), with the SAM method:
#   ./run_local.sh prostate      (or odoc, polyp)
# = run_queue.sh <problem> (every source domain: train target / crop / SAM head, source-only,
# Crop-Teach seeds 0-2; then, OD/OC and prostate, 3 rounds x seeds 0-2), then the tables.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHON=../.venv/bin/python CT_SAM=/run/media/lqtrung/1tb-ssd/Model/ReDi-FM/sam_vit_h_4b8939.pth
LONG_TERM=${LONG_TERM-C} ./run_queue.sh "$1"   # Long-term runs: source domain C only, for now.
./run_queue.sh tables
