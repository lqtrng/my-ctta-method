#!/usr/bin/env python3
"""Long-term CTTA (catastrophic-forgetting check) for ReDi-FM with DINOv3.

Follows "Comparison in a Long-Term CTTA" / Table 3 of the ReDi-FM paper: the
target model is adapted sequentially over the target-domain stream
(B -> C -> D -> E) for several rounds, simulating repeated real-world domain
shifts.  Nothing is reset between domains or rounds: the target model and its
Adam state carry over.  Each round uses exactly the same per-image protocol as
``redifm_dinov3/adapt.py`` (one online update per image, scoring after
the update, per-image DSC).

Reported per round: Do. A DSC (unweighted mean over the target domains).
Summary, as in Table 3:
    Average        = mean of the round DSCs
    Gain           = Average - Source Only
    Perform. Degra. = Round 1 - Average   (negative = improves over rounds)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data import atomic_json_write, build_domains  # noqa: E402
from common.paths import RESULTS_DIR  # noqa: E402
from common.protocol import seed_everything  # noqa: E402
from redifm_dinov3.adapt import build_parser, evaluate_stream, load_models  # noqa: E402


DEFAULT_OUTPUT = RESULTS_DIR / "redifm_dinov3/long_term_domain_A.json"
SOURCE_ONLY_RESULT = RESULTS_DIR / "source_only/baseline_no_adaptation_domain_A.json"


def source_only_dsc(domains: list[str]) -> float:
    """Do. A of the unadapted source model over the same target domains."""
    payload = json.loads(SOURCE_ONLY_RESULT.read_text())
    scores = [row["mean_dsc"] for row in payload["target_domains"] if row["domain"] in domains]
    if len(scores) != len(domains):
        raise RuntimeError(f"{SOURCE_ONLY_RESULT} does not cover domains {domains}")
    return float(np.mean(scores))


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--rounds", type=int, default=3)
    parser.set_defaults(output=DEFAULT_OUTPUT)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.rounds < 1:
        raise ValueError("--rounds must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    domains = build_domains(args.domains, args.max_images)
    source_model, target_model, segmenter, optimizer = load_models(args, device)
    print(f"Long-term CTTA: {args.rounds} rounds of {' -> '.join(domains)}, no reset")

    rounds = []
    for index in range(1, args.rounds + 1):
        print(f"=== Round {index} ===")
        results, _ = evaluate_stream(
            source_model, target_model, segmenter, optimizer, domains, device, args
        )
        do_a = float(np.mean([result.mean_dsc for result in results]))
        print(f"Round {index} Do. A DSC: {do_a:.2f}")
        rounds.append(
            {"round": index, "do_a_dsc": do_a, "target_domains": [asdict(r) for r in results]}
        )

    round_scores = [row["do_a_dsc"] for row in rounds]
    average = float(np.mean(round_scores))
    source_only = source_only_dsc(args.domains)
    gain = average - source_only
    degradation = round_scores[0] - average
    print(
        "Rounds: " + " | ".join(f"{score:.2f}" for score in round_scores)
        + f" | Average {average:.2f} | Gain {gain:+.2f} | Perform. Degra. {degradation:.2f}"
    )

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    atomic_json_write(
        args.output,
        {
            "protocol": "redifm_dinov3_long_term_ctta",
            "source_domain": "A",
            "config": config,
            "rounds": rounds,
            "average_dsc": average,
            "source_only_dsc": source_only,
            "gain_dsc": gain,
            "performance_degradation": degradation,
        },
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main(parse_args())
