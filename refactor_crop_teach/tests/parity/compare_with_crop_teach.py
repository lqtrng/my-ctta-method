#!/usr/bin/env python3
"""Check the refactor against the original crop_teach/ code: same per-image outputs.

Runs the same experiment with both code bases (each in its own process) and compares their
per-image CSV rows (every score, the loss, the consensus, the number of crops) and the summaries.

    # Crop-Teach, 2 images per domain, SAM replaced by a cheap stand-in in both code bases (minutes)
    python tests/parity/compare_with_crop_teach.py --method crop_teach --max-images 2 --sam stub
    # Crop-Teach with the genuine SAM ViT-H: computed once, replayed for the refactor (slow on a CPU)
    python tests/parity/compare_with_crop_teach.py --method crop_teach --max-images 1 --sam cached
    # Compare with a result crop_teach/ already saved (runs/results/), e.g. a full source-only run
    python tests/parity/compare_with_crop_teach.py --method source_only --stored

Exit status 0 when every value matches within --tolerance (default: exactly).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFACTOR = HERE.parents[1]


def run(folder: Path, script: str, args: argparse.Namespace, output: Path, must_hit: bool) -> None:
    command = [sys.executable, str(HERE / "drive.py"), str(folder), script, "--sam", args.sam,
               "--cache", str(args.work / "sam_cache"), *(["--must-hit"] if must_hit else []), "--",
               "--problem", args.problem, "--source", args.source, "--method", args.method, "--seed", str(args.seed),
               "--rounds", str(args.rounds), "--output", str(output), "--device", "cpu", "--force"]
    if args.max_images:
        command += ["--max-images", str(args.max_images)]
    print(f"$ {' '.join(command)}", flush=True)
    environment = {**os.environ, "CT_MLFLOW": "0", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    subprocess.run(command, check=True, cwd=folder, env=environment)


def read_rows(path: Path) -> list[dict]:
    with open(path.with_suffix(".csv")) as handle:
        return list(csv.DictReader(handle))


def same(first: str, second: str, tolerance: float) -> bool:
    try:
        return abs(float(first) - float(second)) <= tolerance
    except ValueError:
        return first == second


def compare(original: Path, refactor: Path, tolerance: float) -> int:
    rows_original, rows_refactor = read_rows(original), read_rows(refactor)
    problems = []
    if len(rows_original) != len(rows_refactor):
        problems.append(f"{len(rows_original)} rows vs {len(rows_refactor)}")
    keys = sorted(set(rows_original[0]) & set(rows_refactor[0]))
    if set(rows_original[0]) != set(rows_refactor[0]):
        problems.append(f"columns differ: {sorted(set(rows_original[0]) ^ set(rows_refactor[0]))}")
    largest = 0.0
    for index, (a, b) in enumerate(zip(rows_original, rows_refactor)):
        for key in keys:
            if not same(a[key], b[key], tolerance):
                problems.append(f"row {index} ({a['domain']} {a['image']}) {key}: {a[key]} vs {b[key]}")
            elif a[key] != b[key]:
                largest = max(largest, abs(float(a[key]) - float(b[key])))
    summary_original = json.loads(original.read_text())["rounds"]
    summary_refactor = json.loads(refactor.read_text())["rounds"]
    for round_original, round_refactor in zip(summary_original, summary_refactor):
        for key, value in round_original["mean"].items():
            if not same(str(value), str(round_refactor["mean"][key]), tolerance * 100):
                problems.append(f"round {round_original['round']} mean {key}: {value} vs {round_refactor['mean'][key]}")
    print(f"Compared {len(rows_refactor)} images x {len(keys)} columns; largest difference {largest:g}")
    for line in problems[:30]:
        print(f"  MISMATCH {line}")
    print("PARITY OK" if not problems else f"PARITY FAILED: {len(problems)} mismatches")
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", type=Path, default=REFACTOR.parent / "crop_teach")
    parser.add_argument("--problem", default="prostate")
    parser.add_argument("--source", default="A")
    parser.add_argument("--method", choices=["source_only", "crop_teach"], default="crop_teach")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--sam", choices=["real", "cached", "stub"], default="stub")
    parser.add_argument("--stored", action="store_true", help="Compare with crop_teach/runs/results/ instead")
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--work", type=Path, default=REFACTOR / "runs" / "parity")
    args = parser.parse_args()
    args.work = args.work.resolve()
    tag = f"{args.problem}_{args.method}_source{args.source}_seed{args.seed}_max{args.max_images}_{args.sam}"
    refactor = args.work / f"refactor_{tag}.json"
    if args.stored:
        seed = f"_seed{args.seed}" if args.method == "crop_teach" else ""
        rounds = f"_rounds{args.rounds}" if args.rounds > 1 else ""
        original = args.original / "runs" / "results" / args.problem / f"{args.method}_source{args.source}{seed}{rounds}.json"
    else:
        original = args.work / f"original_{tag}.json"
        run(args.original, "adapt.py", args, original, must_hit=False)
    run(REFACTOR, "run.py", args, refactor, must_hit=args.sam == "cached")
    sys.exit(compare(original, refactor, args.tolerance))


if __name__ == "__main__":
    main()
