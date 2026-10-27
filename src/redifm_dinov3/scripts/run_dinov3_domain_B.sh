#!/usr/bin/env bash
# Reproduce the ReDi-FM/DINOv3 domain-B OD/OC evaluation (source: domain A).
# Prerequisite (one-off): .venv/bin/python src/download/download.py dinov3
# 1) one-off: train the frozen-backbone decoder on labeled source domain A only
# 2) online continual test-time adaptation on target domain B, 3 seeds
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
HEAD=checkpoints/dinov3_vitl16_odoc_head_domain_A.pth

if [ ! -f "$HEAD" ]; then
  $PY src/redifm_dinov3/train_head.py --output "$HEAD"
fi
mkdir -p results/redifm_dinov3
for seed in 0 1 2; do
  $PY src/redifm_dinov3/adapt.py --domains B --seed "$seed" \
    --output "results/redifm_dinov3/final_domain_B_seed$seed.json"
done
$PY - <<'PY'
import json
rows = [json.load(open(f"results/redifm_dinov3/final_domain_B_seed{s}.json"))["target_domains"][0] for s in range(3)]
for key in ("disc_dsc", "cup_dsc", "mean_dsc"):
    print(f"{key}: {sum(r[key] for r in rows) / 3:.2f}  ({', '.join(f'{r[key]:.2f}' for r in rows)})")
PY
