#!/usr/bin/env python3
"""Average result JSONs over seeds and print them like the README tables.

    python src/common/report.py results/redifm_dinov3/final_all_domains_seed*.json

Works for every script's output: per target domain, the mean of each DSC over
the given files, then Do. A as mean +- std over the files.  Long-term results
(``evaluate.py --rounds``) are printed per round with Average, Gain and
Perform. Degra. as in Table 3 of the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def domain_table(runs: list[dict]) -> None:
    for index, domain in enumerate(runs[0]["target_domains"]):
        rows = [run["target_domains"][index] for run in runs]
        keys = [key for key in domain if key.endswith("_dsc")]
        print(f"  Domain {domain['domain']} ({domain.get('dataset', '')}): " + ", ".join(
            f"{key.removesuffix('_dsc')}={np.mean([row[key] for row in rows]):.2f}" for key in keys))
    do_a = [run["do_a_dsc"] for run in runs]
    print(f"  Do. A DSC: {np.mean(do_a):.2f} +- {np.std(do_a):.2f}  ({', '.join(f'{v:.2f}' for v in do_a)})")


def long_term_table(runs: list[dict]) -> None:
    rounds = np.array([[row["do_a_dsc"] for row in run["rounds"]] for run in runs]).mean(axis=0)
    average = rounds.mean()
    source_only = runs[0].get("source_only_dsc")
    gain = "" if source_only is None else f" | Gain {average - source_only:+.2f} (Source Only {source_only:.2f})"
    print("  " + " | ".join(f"Round {index + 1} {value:.2f}" for index, value in enumerate(rounds)))
    print(f"  Average {average:.2f}{gain} | Perform. Degra. {rounds[0] - average:.2f}")


def main(paths: list[Path]) -> None:
    runs = [json.loads(path.read_text()) for path in paths]
    print(f"Mean over {len(runs)} file(s): {', '.join(path.name for path in paths)}")
    if "rounds" in runs[0]:
        long_term_table(runs)
    else:
        domain_table(runs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main([Path(path) for path in sys.argv[1:]])
