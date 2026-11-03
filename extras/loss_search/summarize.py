#!/usr/bin/env python3
"""Rank the losses run so far by the mean of prostate and OD/OC on dev images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results/loss_search"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main() -> None:
    table: dict[str, dict] = {}
    for path in RESULTS_DIR.glob(f"*_seed{SEED}_r1.json"):
        result = json.loads(path.read_text())
        table.setdefault(result["loss"], {})[result["task"]] = result["rounds"][-1]

    def key(item: tuple[str, dict]) -> tuple[int, float]:
        tasks = item[1]
        return (len(tasks), sum(tasks[t]["dev"] for t in tasks) / len(tasks))

    print(f"{'loss':<22}{'pro dev':>9}{'odoc dev':>9}{'mean dev':>10}{'pro held':>9}{'odoc held':>10}{'pro all':>9}{'odoc all':>9}")
    for loss, tasks in sorted(table.items(), key=key, reverse=True):
        p, o = tasks.get("prostate"), tasks.get("odoc")
        mean = (p["dev"] + o["dev"]) / 2 if p and o else float("nan")

        def cell(result: dict | None, name: str) -> str:
            return f"{result[name]:.2f}" if result else "-"

        print(f"{loss:<22}{cell(p, 'dev'):>9}{cell(o, 'dev'):>9}{mean:>10.2f}{cell(p, 'heldout'):>9}"
              f"{cell(o, 'heldout'):>10}{cell(p, 'all'):>9}{cell(o, 'all'):>9}")


if __name__ == "__main__":
    main()
