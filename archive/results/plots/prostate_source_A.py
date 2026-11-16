"""Prostate, source A, seed 0, same R34 source checkpoint: source only vs ReDi-FM + SAM vs Crop-Teach."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CT = ROOT / "crop_teach/runs/results/prostate"
DOMAINS = list("BCDEF")


def crop_teach(name):
    d = json.load(open(CT / name))
    r = d["rounds"][0]["domains"]
    return [r[k]["prostate_dsc"] for k in DOMAINS] + [d["do_dsc"]]


redi = json.load(open(ROOT / "results/redifm_sam/prostate_domain_A_seed0_crop_teach_source.json"))
redi = [t["prostate_dsc"] for t in redi["target_domains"]] + [redi["do_a_dsc"]]
series = {
    "Source only": crop_teach("source_only_sourceA.json"),
    "ReDi-FM + SAM": redi,
    "Crop-Teach": crop_teach("crop_teach_sourceA_seed0.json"),
}
colors = ["#9b9a94", "#eb6834", "#2a78d6"]

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(DOMAINS) + 1)
w = 0.26
for i, ((name, vals), c) in enumerate(zip(series.items(), colors)):
    bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=c, label=name)
    ax.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2, color="#3d3d3a")
ax.set_xticks(x, DOMAINS + ["Do. A (mean)"])
ax.set_ylabel("DSC (%)")
ax.set_ylim(0, 100)
ax.set_title("Prostate, source A, seed 0, same ResUNet-34 source", loc="left", fontsize=11)
ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
fig.tight_layout()
out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=150)
print(out)
for name, vals in series.items():
    print(f"{name:14s}", " ".join(f"{v:6.2f}" for v in vals))
