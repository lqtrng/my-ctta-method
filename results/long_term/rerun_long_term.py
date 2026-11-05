#!/usr/bin/env python3
"""Long-term CTTA (3 passes, never reset) for the prominent adaptation methods of the README.

Each run is rebuilt from the ``config`` of its 1-round result JSON and re-run
through ``src/evaluate.py`` with ``--rounds 3``.  Options a config predates take
the value they had then: ``--crop-region largest`` (``all`` came later).

    .venv/bin/python results/long_term/rerun_long_term.py            # run all missing
    .venv/bin/python results/long_term/rerun_long_term.py --dry-run  # print the commands
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import build_parser  # noqa: E402

OUT = ROOT / "results" / "long_term"
# The prominent adaptation methods, not every row.  README row -> (method, its 1-round result).
# Crop-teach's best (O5, P2, Y2) runs in run_long_term.sh; ReDi-FM + D3 with R34 (O7) already
# has results/redifm_dinov3/long_term_seed0.json.  Polyp has no ReDi-FM run.
RUNS = {
    "O6": ("redifm_dinov3", "results/redifm_dinov3_nnunet/all_domains_seed0.json"),        # ReDi-FM + D3, best backbone (nnU*)
    "O8": ("redifm_sam", "results/redifm_sam/redifm_sam_domain_A.json"),                   # ReDi-FM + SAM, the paper's method
    "P6": ("redifm_dinov3", "results/redifm_dinov3_nnunet_official/prostate_all_domains_seed0.json"),  # ReDi-FM + D3, best (nnU)
}
SKIP = {"method", "rounds", "output", "device", "max_images"}


def command(row: str, method: str, result: str) -> tuple[list[str], Path]:
    payload = json.loads((ROOT / result).read_text())
    config = payload.get("config", {})
    task = config.get("task") or payload.get("task") or "odoc"
    parser = build_parser(method)
    options = {action.dest: action for action in parser._actions if action.option_strings}
    if "crop_region" in options and "crop_region" not in config:
        config["crop_region"] = "largest"
    output = OUT / f"{task}_{row}_long_term_seed{config.get('seed', 42)}.json"
    argv = ["--method", method, "--task", task, "--rounds", "3", "--output", str(output)]
    for key, value in config.items():
        if key in SKIP or key == "task" or key not in options or value is None:
            continue
        action, flag = options[key], options[key].option_strings[-1]
        if action.nargs == 0:  # store_true / store_false
            if value != action.default:
                argv.append(flag)
        elif isinstance(value, list):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    parser.parse_args(argv)  # Fails here, not mid-queue, on an option that no longer exists.
    return argv, output


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("rows", nargs="*", default=list(RUNS))
    cli.add_argument("--dry-run", action="store_true")
    args = cli.parse_args()
    commands = {row: command(row, *RUNS[row]) for row in args.rows}
    for row, (argv, output) in commands.items():
        print(f"== {row}: python src/evaluate.py {' '.join(argv)}", flush=True)
        if args.dry_run:
            continue
        if output.is_file():
            print(f"   exists: {output.name}", flush=True)
            continue
        with open(output.with_suffix(".log"), "w") as log:
            status = subprocess.run([sys.executable, "src/evaluate.py", *argv], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"   {'done' if status == 0 else f'FAILED ({status})'}: {output.with_suffix('.log').name}", flush=True)


if __name__ == "__main__":
    main()
