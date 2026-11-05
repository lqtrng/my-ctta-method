#!/usr/bin/env bash
# The ReDi-FM-paper experiments for Crop-Teach, one problem per call (two calls can share a 16 GB GPU):
#
#   ./run_queue.sh smoke              phase 0: every code path on a few images, into runs_smoke/
#   ./run_queue.sh odoc|prostate|polyp       in layers (see problem_queue):
#       every source domain: train target / crop / head, source-only, Crop-Teach seed 0
#       then long-term (3 rounds) seed 0; then seeds 1 and 2 everywhere; then long-term seeds 1, 2
#       LONG_TERM="C" limits the long-term runs to those source domains (default: all)
#   ./run_queue.sh tables             write runs/results/tables.md
#
# Every step skips its output if it already exists, so the queue can be stopped and restarted.
# Logs: runs/logs/<step>.log
set -euo pipefail
cd "$(dirname "$0")"
P=${PYTHON:-python}
export MLFLOW_DISABLE_AGENT_HINT=1

step() {  # step <log name> <python args...>
  local name=$1; shift
  mkdir -p "${CT_RUNS:-runs}/logs"
  echo "$(date '+%F %T') start $name"
  if ! $P "$@" > "${CT_RUNS:-runs}/logs/$name.log" 2>&1; then
    echo "$(date '+%F %T') FAILED $name (see ${CT_RUNS:-runs}/logs/$name.log)"; return 1
  fi
  echo "$(date '+%F %T') done  $name"
}

sources() { $P -c "from config import PROBLEMS; print(' '.join(PROBLEMS['$1'].domains))"; }

train_models() {  # train_models <problem> <source>
  for kind in target crop; do
    step "${1}_train_${kind}_source$2" train.py --problem "$1" --source "$2" --kind "$kind"
  done
  # A 20M head at 1024 px needs ~13-14 GB at batch 8: one head training at a time on the GPU.
  mkdir -p "${CT_RUNS:-runs}"
  flock "${CT_RUNS:-runs}/head_training.lock" \
    "$0" _step "${1}_train_head_source$2" train.py --problem "$1" --source "$2" --kind head
}

crop_teach() {  # crop_teach <problem> <source> <seed> [rounds]
  local rounds=${4:-1} tag=""
  [ "$rounds" -gt 1 ] && tag="_rounds$rounds"
  step "${1}_crop_teach_source${2}_seed$3$tag" adapt.py --problem "$1" --source "$2" --method crop_teach \
    --seed "$3" --rounds "$rounds"
}

problem_queue() {
  # In layers, so that a complete table exists as early as possible:
  #   layer 1: every source domain: train its models, source-only, Crop-Teach seed 0
  #   long-term (3 rounds) seed 0, for the sources in LONG_TERM (default: all; OD/OC, prostate only)
  #   layer 2: every source domain, seed 1;  layer 3: seed 2
  #   long-term seeds 1, 2
  local problem=$1
  local long=${LONG_TERM-$(sources "$problem")}
  [ "$problem" = polyp ] && long=""  # The paper's long-term table covers OD/OC and prostate.
  for s in $(sources "$problem"); do
    train_models "$problem" "$s"
    step "${problem}_source_only_source$s" adapt.py --problem "$problem" --source "$s" --method source_only
    crop_teach "$problem" "$s" 0
  done
  for s in $long; do crop_teach "$problem" "$s" 0 3; done
  for seed in 1 2; do
    for s in $(sources "$problem"); do crop_teach "$problem" "$s" "$seed"; done
  done
  for seed in 1 2; do
    for s in $long; do crop_teach "$problem" "$s" "$seed" 3; done
  done
}

smoke() {  # Tiny versions of every step, in a separate runs folder.
  export CT_RUNS=runs_smoke CT_MLFLOW=0
  for problem in odoc prostate polyp; do
    for kind in target crop head; do
      step "${problem}_train_${kind}" train.py --problem "$problem" --source A --kind "$kind" \
        --epochs 1 --max-images 16 --workers 2 --force
    done
    step "${problem}_source_only" adapt.py --problem "$problem" --source A --method source_only --max-images 2 --force
    step "${problem}_crop_teach" adapt.py --problem "$problem" --source A --method crop_teach --max-images 2 \
      --rounds 2 --force
  done
  step tables summarize.py
  echo "Smoke test passed: every step ran. Delete runs_smoke/ when done."
}

case "${1:-}" in
  _step) shift; step "$@" ;;  # Internal: one step, run under flock.
  smoke) smoke ;;
  odoc|prostate|polyp) problem_queue "$1" ;;
  tables) step tables summarize.py && cat "${CT_RUNS:-runs}/results/tables.md" ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
