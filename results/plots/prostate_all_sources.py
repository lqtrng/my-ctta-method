"""Prostate, every source domain, mean ± std over seeds: ReDi-FM + SAM vs Crop-Teach.

Both use Crop-Teach's own ResUNet-34 source checkpoint for each source. Crop-Teach source D
leaves out seed 1, which collapsed (Do. 9.13).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = list("ABCDEF")


def redifm(source):
    return [json.load(open(ROOT / f"results/redifm_sam/prostate_crop_teach_source_{source}_seed{k}.json"))["do_a_dsc"]
            for k in (0, 1, 2)]


def crop_teach(source):
    seeds = (0, 2) if source == "D" else (0, 1, 2)
    return [json.load(open(ROOT / f"crop_teach/runs/results/prostate/crop_teach_source{source}_seed{k}.json"))["do_dsc"]
            for k in seeds]


series = {"ReDi-FM + SAM": [redifm(s) for s in SOURCES], "Crop-Teach": [crop_teach(s) for s in SOURCES]}
colors = ["#eb6834", "#2a78d6"]

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(SOURCES) + 1)
w = 0.38
for i, ((name, runs), c) in enumerate(zip(series.items(), colors)):
    means = [np.mean(r) for r in runs] + [np.mean([np.mean(r) for r in runs])]
    stds = [np.std(r) for r in runs] + [0]
    bars = ax.bar(x + (i - 0.5) * w, means, w - 0.03, yerr=stds, capsize=2, color=c, label=name,
                  error_kw={"elinewidth": 1, "ecolor": "#3d3d3a"})
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3, color="#3d3d3a")
ax.set_xticks(x, [f"Source {s}" for s in SOURCES] + ["Mean"])
ax.set_ylabel("Do. DSC (%)")
ax.set_ylim(0, 100)
ax.set_title("Prostate, mean of 3 seeds, same ResUNet-34 source per domain", loc="left", fontsize=11)
ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.1))
fig.tight_layout()
out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=150)
print(out)
