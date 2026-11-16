#!/usr/bin/env python3
"""Run one loss on the frozen supervision saved by ``cache_supervision.py``.

    python extras/loss_search/run_loss.py --task prostate --loss baseline --seed 0

Every image adapts, in stream order, once per round.  The images are split in two
for scoring only: losses are chosen on ``dev`` and reported on ``heldout``.  Prostate
splits by case so that slices of one patient stay together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_supervision import CACHE_DIR  # noqa: E402
from losses import LOSSES, Context  # noqa: E402

from common.models import build_network  # noqa: E402
from common.paths import SOURCE_CHECKPOINTS  # noqa: E402
from common.protocol import make_augmentation, predict_after_update, seed_everything  # noqa: E402
from common.redifm import (  # noqa: E402
    consensus_score,
    nested_foundation_classes,
    structure_probabilities,
)
from common.tasks import TASKS  # noqa: E402
from methods.redifm_dinov3 import add_arguments  # noqa: E402
from methods.redifm_dinov3_nnunet_crop import FOUNDATION_ARCH  # noqa: E402


RESULTS_DIR = PROJECT_DIR / "results/loss_search"


def split_of(task: str, row: dict) -> str:
    key = row["image"]
    if task == "prostate":
        key = row["domain"] + re.match(r"Case\d+", row["image"]).group(0)
    return "dev" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 == 0 else "heldout"


def do_a(rows: list[dict], split: str | None) -> float:
    """Unweighted mean over domains of the mean image DSC, in percent."""
    means = {}
    for domain in sorted({row["domain"] for row in rows}):
        values = [row["dsc"] for row in rows if row["domain"] == domain and split in (None, row["split"])]
        means[domain] = 100.0 * float(np.mean(values))
    return float(np.mean(list(means.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["odoc", "prostate", "polyp"], required=True)
    parser.add_argument("--loss", choices=list(LOSSES), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-images", type=int, help="Stop early, for a smoke test")
    parser.add_argument("--output", type=Path)
    add_arguments(parser)
    args = parser.parse_args()
    task = TASKS[args.task]
    device = torch.device("cuda")
    seed_everything(args.seed)

    folder = CACHE_DIR / task.name
    index = json.loads((folder / "index.json").read_text())
    images = np.load(folder / "images.npy", mmap_mode="r")
    targets = np.load(folder / "targets.npy", mmap_mode="r")
    source_all = np.load(folder / "source_classes.npy", mmap_mode="r")
    foundation_all = np.load(folder / "foundation_structures.npy", mmap_mode="r")
    boxes = np.load(folder / "boxes.npy", mmap_mode="r")

    arch = FOUNDATION_ARCH[task.name]
    target_model = build_network(arch, task.num_classes, task.image_size)
    target_model.load_state_dict(torch.load(SOURCE_CHECKPOINTS[(task.name, arch)], map_location="cpu", weights_only=True))
    target_model.to(device).train()
    initial = {key: value.clone() for key, value in target_model.state_dict().items()}
    learning_rate = args.learning_rate or LOSSES[args.loss].learning_rate or task.learning_rate
    optimizer = torch.optim.Adam(target_model.parameters(), lr=learning_rate)
    augmentation = make_augmentation(args)

    count = min(args.max_images or len(index), len(index))
    start = time.perf_counter()
    rounds = []
    for round_index in range(args.rounds):
        rows = []
        for position in range(count):
            array = np.asarray(images[position], dtype=np.float32).transpose(2, 0, 1)
            array = (array - array.min()) / max(float(array.max() - array.min()), 1e-7)
            image_tensor = torch.from_numpy(array.copy())[None].to(device)
            source_classes = torch.from_numpy(np.asarray(source_all[position], dtype=np.float32).copy())[None].to(device)
            foundation = torch.from_numpy(np.asarray(foundation_all[position], dtype=np.float32).copy())[None].to(device)
            source = structure_probabilities(source_classes)
            source_masks = source > args.pseudo_threshold
            foundation_masks = foundation > 0.5
            consensus = consensus_score(source_masks, foundation_masks).detach()

            target_model.train()
            if LOSSES[args.loss].freeze_batch_norm:
                for module in target_model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            optimizer.zero_grad(set_to_none=True)
            augmented = augmentation(image_tensor.squeeze(0)).unsqueeze(0)
            target_classes = F.softmax(target_model(augmented), dim=1)
            context = Context(
                target_classes=target_classes,
                target=structure_probabilities(target_classes),
                source_classes=source_classes,
                source=source,
                source_masks=source_masks,
                foundation_classes=nested_foundation_classes(foundation),
                foundation=foundation,
                foundation_masks=foundation_masks,
                consensus=consensus,
                box=tuple(int(value) for value in boxes[position]),
                entropy_threshold=args.entropy_threshold,
                model=target_model,
                initial=initial,
            )
            loss = LOSSES[args.loss].loss(context)
            if loss.requires_grad:
                loss.backward()
                if LOSSES[args.loss].scale_lr_by_agreement:
                    optimizer.param_groups[0]["lr"] = learning_rate * float(consensus)
                optimizer.step()

            prediction = predict_after_update(target_model, image_tensor)[0].cpu()
            target = torch.from_numpy(np.asarray(targets[position]).copy()).long()
            scores = task.score(prediction, target)
            row = index[position]
            rows.append({"domain": row["domain"], "image": row["image"], "split": split_of(task.name, row),
                         "dsc": float(np.mean(list(scores.values()))), "consensus": float(consensus)})
        rounds.append(rows)

    def summary(rows: list[dict]) -> dict:
        return {
            "all": do_a(rows, None),
            "dev": do_a(rows, "dev"),
            "heldout": do_a(rows, "heldout"),
            "domains": {domain: 100.0 * float(np.mean([r["dsc"] for r in rows if r["domain"] == domain]))
                        for domain in sorted({r["domain"] for r in rows})},
        }

    result = {"task": task.name, "loss": args.loss, "seed": args.seed, "rounds": [summary(rows) for rows in rounds],
              "learning_rate": learning_rate, "seconds": time.perf_counter() - start}
    output = args.output or RESULTS_DIR / f"{task.name}_{args.loss}_seed{args.seed}_r{args.rounds}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))
    last = result["rounds"][-1]
    print(f"{task.name} {args.loss} seed{args.seed}: dev={last['dev']:.2f} heldout={last['heldout']:.2f} "
          f"all={last['all']:.2f} {({k: round(v, 1) for k, v in last['domains'].items()})} "
          f"{result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
