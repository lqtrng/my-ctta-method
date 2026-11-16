#!/usr/bin/env python3
"""Save the frozen supervision of ``redifm_dinov3_nnunet_crop`` once per image.

The teachers never change, so a loss experiment does not have to run them again:

    python extras/loss_search/cache_supervision.py --task prostate
    python extras/loss_search/run_loss.py --task prostate --loss baseline

Arrays go to .cache/loss_search/<task>/ as one file per array, in stream order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from common.dinov3 import load_dinov3  # noqa: E402
from common.protocol import image_to_tensor, prepare_sample  # noqa: E402
from common.redifm import structure_probabilities  # noqa: E402
from common.tasks import TASKS  # noqa: E402
from methods.redifm_dinov3 import add_arguments  # noqa: E402
from methods.redifm_dinov3_dual import FULL_IMAGE_HEAD_CHECKPOINTS, load_full_image_network  # noqa: E402
from methods.redifm_dinov3_nnunet_crop import NnUNetCropSegmenter  # noqa: E402


CACHE_DIR = PROJECT_DIR / ".cache/loss_search"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["odoc", "prostate", "polyp"], required=True)
    add_arguments(parser)
    args = parser.parse_args()
    task = TASKS[args.task]
    args.image_size = task.image_size
    args.foundation_checkpoint = None
    device = torch.device("cuda")

    backbone = load_dinov3(args.dinov3_dir, args.dinov3_arch, device)
    source_model = load_full_image_network(backbone, FULL_IMAGE_HEAD_CHECKPOINTS[task.name], args.dinov3_arch)
    source_model.requires_grad_(False).to(device).eval()
    segmenter = NnUNetCropSegmenter(args, task, device)

    domains = task.build_domains(list(task.target_domains), None)
    count = sum(len(samples) for samples in domains.values())
    size = task.image_size
    structures = len(task.structures)
    folder = CACHE_DIR / task.name
    folder.mkdir(parents=True, exist_ok=True)

    def array(name: str, dtype: str, shape: tuple[int, ...]) -> np.memmap:
        return np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+", dtype=dtype, shape=(count, *shape))

    images = array("images", "uint8", (size, size, 3))
    targets = array("targets", "uint8", (size, size))
    source_classes = array("source_classes", "float16", (task.num_classes, size, size))
    foundation_structures = array("foundation_structures", "float16", (structures, size, size))
    boxes = array("boxes", "int32", (3,))

    index_rows = []
    index = 0
    with torch.inference_mode():
        for domain, samples in domains.items():
            for sample in tqdm(samples, desc=f"{task.name} {domain}"):
                image, target = prepare_sample(sample, size)
                image_tensor = image_to_tensor(image, device)
                classes = F.softmax(source_model(image_tensor), dim=1)
                source = structure_probabilities(classes)
                _, foundation, _ = segmenter.supervision(image, image_tensor, source, args.pseudo_threshold)
                soft = source[0].cpu().numpy()
                boxes[index] = segmenter.crop_box(soft[0] > args.pseudo_threshold, soft[0])
                images[index] = np.asarray(image, dtype=np.uint8)
                targets[index] = target.numpy()
                source_classes[index] = classes[0].cpu().numpy().astype(np.float16)
                foundation_structures[index] = foundation[0].cpu().numpy().astype(np.float16)
                index_rows.append({"domain": domain, "dataset": sample.dataset, "image": sample.identifier})
                index += 1
    for memmap in (images, targets, source_classes, foundation_structures, boxes):
        memmap.flush()
    (folder / "index.json").write_text(json.dumps(index_rows))
    print(f"Saved {index} images to {folder}")


if __name__ == "__main__":
    main()
