#!/usr/bin/env bash
# ReDi-FM/DINOv3 on the full Domain-A protocol: continual stream B -> C -> D -> E,
# no reset between domains, Do. A = unweighted mean of the four domain DSCs.
# Prerequisite (one-off): .venv/bin/python src/download/download.py dinov3
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
HEAD=$($PY -c "import sys; sys.path.insert(0, 'src'); from redifm_dinov3.adapt import DEFAULT_HEAD_CHECKPOINT as h; print(h)")
SEEDS=${SEEDS:-"0 1 2"}

if [ ! -f "$HEAD" ]; then
  $PY src/redifm_dinov3/train_head.py --output "$HEAD"
fi
mkdir -p results/redifm_dinov3
for seed in $SEEDS; do
  $PY src/redifm_dinov3/adapt.py --seed "$seed" \
    --output "results/redifm_dinov3/final_all_domains_seed$seed.json"
done
SEEDS="$SEEDS" $PY - <<'PY'
import json, os
import numpy as np
seeds = os.environ["SEEDS"].split()
runs = [json.load(open(f"results/redifm_dinov3/final_all_domains_seed{s}.json")) for s in seeds]
print(f"Mean over seeds {seeds}:")
for index, domain in enumerate(runs[0]["target_domains"]):
    rows = [run["target_domains"][index] for run in runs]
    print(f"  Domain {domain['domain']} ({domain['dataset']}): " + ", ".join(
        f"{key.split('_')[0].upper()}={np.mean([r[key] for r in rows]):.2f}"
        for key in ("disc_dsc", "cup_dsc", "mean_dsc")))
do_a = [run["do_a_dsc"] for run in runs]
print(f"  Do. A DSC: {np.mean(do_a):.2f} +- {np.std(do_a):.2f}  ({', '.join(f'{v:.2f}' for v in do_a)})")
PY
