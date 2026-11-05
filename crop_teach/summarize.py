#!/usr/bin/env python3
"""Collect runs/results/ into the ReDi-FM paper's tables, next to the paper's own numbers.

    python summarize.py        # writes runs/results/tables.md

Table 1 (OD/OC, prostate) and Table 2 (polyp): Do. X = mean over the target domains of source X;
Crop-Teach is mean +- std over seeds. Table 3: long-term, 3 rounds, averaged over source domains
and seeds; Gain = Average - source-only Average; Degradation = round 1 - Average.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from config import PROBLEMS, RESULTS_DIR

# The ReDi-FM paper (IJCAI-26), Tables 1-3: {problem: {row: [per source domain..., average]}}.
PAPER_T1 = {
    "odoc": {"Source only (paper)": [53.59, 74.01, 67.24, 57.39, 71.17, 64.68],
             "ReDi-FM (paper)": [73.11, 75.69, 73.31, 73.48, 72.37, 73.59]},
    "prostate": {"Source only (paper)": [51.02, 58.79, 26.88, 24.15, 60.42, 53.33, 45.77],
                 "ReDi-FM (paper)": [66.10, 66.44, 49.89, 41.24, 73.50, 73.69, 61.81]},
}
PAPER_T2 = {  # per source: (DSC, E-phi max, S-alpha); last = average
    "Source only (paper)": [(74.22, 83.82, 81.37), (70.41, 82.67, 80.02), (68.46, 83.05, 77.47),
                            (76.55, 88.34, 84.56), (72.41, 84.47, 80.86)],
    "ReDi-FM (paper)": [(77.47, 87.02, 82.78), (73.97, 85.65, 81.75), (69.56, 82.48, 78.49),
                        (78.35, 90.02, 85.22), (74.84, 86.29, 82.06)],
}
PAPER_T3 = {"odoc": [73.59, 73.17, 72.74, 73.17, 8.48, 0.43], "prostate": [61.81, 64.85, 65.33, 64.00, 18.23, -2.19]}


def load() -> dict:
    """{(problem, method, source, rounds): [result per seed]}"""
    results = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*/*.json")):
        r = json.loads(path.read_text())
        if r["config"].get("max_images"):
            continue  # Smoke tests.
        results[(r["problem"], r["method"], r["source"], len(r["rounds"]))].append(r)
    return results


def cell(values: list[float]) -> str:
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def table_1(results: dict, problem: str) -> list[str]:
    sources = list(PROBLEMS[problem].domains)
    lines = [f"### Table 1, {problem}: Do. X DSC", "",
             "| Method | " + " | ".join(f"Do. {s}" for s in sources) + " | Average |",
             "|---" * (len(sources) + 2) + "|"]
    for name, values in PAPER_T1[problem].items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    for method, label in (("source_only", "Source only (ours)"), ("crop_teach", "Crop-Teach")):
        per_source = [[r["do_dsc"] for r in results.get((problem, method, s, 1), [])] for s in sources]
        # The average over sources, per seed (so its std is over seeds, as in the paper).
        complete = all(per_source) and len({len(v) for v in per_source}) == 1
        average = [float(np.mean(seed)) for seed in zip(*per_source)] if complete else []
        lines.append(f"| {label} | " + " | ".join(cell(v) for v in per_source) + f" | {cell(average)} |")
    return lines + [""]


def table_2(results: dict) -> list[str]:
    sources = list(PROBLEMS["polyp"].domains)
    keys = ("mean_dsc", "e_phi_max", "s_alpha")
    lines = ["### Table 2, polyp: Do. X (DSC / E-phi max / S-alpha)", "",
             "| Method | " + " | ".join(f"Do. {s}" for s in sources) + " | Average |",
             "|---" * (len(sources) + 2) + "|"]
    for name, values in PAPER_T2.items():
        lines.append(f"| {name} | " + " | ".join(" / ".join(f"{x:.2f}" for x in v) for v in values) + " |")
    for method, label in (("source_only", "Source only (ours)"), ("crop_teach", "Crop-Teach")):
        cells, averages = [], {k: [] for k in keys}
        for s in sources:
            runs = results.get(("polyp", method, s, 1), [])
            cells.append(" / ".join(f"{np.mean([r['rounds'][0]['mean'][k] for r in runs]):.2f}" for k in keys)
                         if runs else "–")
            for k in keys:
                averages[k].append(np.mean([r["rounds"][0]["mean"][k] for r in runs]) if runs else np.nan)
        average = " / ".join(f"{np.mean(averages[k]):.2f}" for k in keys) if all(cells[i] != "–" for i in range(len(sources))) else "–"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {average} |")
    return lines + ["", "Crop-Teach cells are seed means; Table 1 gives the seed std of the DSC.", ""]


def table_3(results: dict) -> list[str]:
    lines = ["### Table 3, long-term (3 rounds), averaged over source domains and seeds", "",
             "| Problem | Method | Round 1 | Round 2 | Round 3 | Average | Gain | Degradation |",
             "|---|---|---|---|---|---|---|---|"]
    for problem in ("odoc", "prostate"):
        p = PAPER_T3[problem]
        lines.append(f"| {problem} | ReDi-FM (paper) | " + " | ".join(f"{v:.2f}" for v in p) + " |")
        sources = list(PROBLEMS[problem].domains)
        runs = [results.get((problem, "crop_teach", s, 3), []) for s in sources]
        source_only = [results.get((problem, "source_only", s, 1), []) for s in sources]
        if not all(runs) or not all(source_only):
            lines.append(f"| {problem} | Crop-Teach | – | – | – | – | – | – |")
            continue
        rounds = [float(np.mean([np.mean([r["rounds"][k]["mean"]["mean_dsc"] for r in per]) for per in runs]))
                  for k in range(3)]
        average = float(np.mean(rounds))
        gain = average - float(np.mean([per[0]["do_dsc"] for per in source_only]))
        lines.append(f"| {problem} | Crop-Teach | " + " | ".join(f"{v:.2f}" for v in rounds)
                     + f" | {average:.2f} | {gain:+.2f} | {rounds[0] - average:.2f} |")
    return lines + [""]


def main() -> None:
    results = load()
    lines = ["# Crop-Teach on the ReDi-FM paper's benchmarks", ""]
    lines += table_1(results, "odoc") + table_1(results, "prostate") + table_2(results) + table_3(results)
    output = RESULTS_DIR / "tables.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
