#!/usr/bin/env bash
# Long-term CTTA / catastrophic-forgetting check (ReDi-FM Table 3 protocol):
# 3 rounds of B -> C -> D -> E without reset, 3 seeds.
# Requires the DINOv3 weights and decoder head (see src/redifm_dinov3/scripts/run_dinov3_all_domains.sh).
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
SEEDS=${SEEDS:-"0 1 2"}
ROUNDS=${ROUNDS:-3}
mkdir -p results/redifm_dinov3
for seed in $SEEDS; do
  $PY src/redifm_dinov3/long_term.py --seed "$seed" --rounds "$ROUNDS" \
    --output "results/redifm_dinov3/long_term_seed$seed.json"
done
SEEDS="$SEEDS" $PY - <<'PY'
import json, os
import numpy as np
seeds = os.environ["SEEDS"].split()
runs = [json.load(open(f"results/redifm_dinov3/long_term_seed{s}.json")) for s in seeds]
rounds = np.array([[r["do_a_dsc"] for r in run["rounds"]] for run in runs])
mean = rounds.mean(axis=0)
average = mean.mean()
source_only = runs[0]["source_only_dsc"]
print(f"Mean over seeds {seeds}:")
print("  " + " | ".join(f"Round {i + 1} {v:.2f}" for i, v in enumerate(mean)))
print(f"  Average {average:.2f} | Gain {average - source_only:+.2f} "
      f"(Source Only {source_only:.2f}) | Perform. Degra. {mean[0] - average:.2f}")
PY
