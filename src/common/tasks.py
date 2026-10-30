"""Segmentation benchmarks: classes, structures, domains and per-image scoring.

``odoc``: optic disc / optic cup, classes {background, disc rim, cup}, nested
structures OD (rim + cup) and OC; source domain A, target stream B-E, 512 x 512.
``prostate``: six-site prostate MRI (0140.pdf, Section 4.1), classes
{background, prostate}, one structure; source domain A, target stream B-F,
384 x 384.

Every task scores a prediction per structure with DSC; the image score is the
mean over structures, and Do. A the unweighted mean over target domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from common.data import (
    DomainADataset,
    Sample,
    SampleDataset,
    build_domains,
    build_prostate_domains,
)
from common.paths import DOMAIN_A_DIR


@dataclass(frozen=True)
class Task:
    name: str
    num_classes: int
    structures: tuple[str, ...]
    image_size: int
    target_domains: tuple[str, ...]
    build_domains: Callable[[list[str], int | None], dict[str, list[Sample]]]

    def source_dataset(self) -> Dataset:
        """Labeled source domain A in the format of ``DomainADataset``."""
        if self.name == "odoc":
            return DomainADataset(DOMAIN_A_DIR, self.image_size)
        return SampleDataset(self.build_domains(["A"], None)["A"], self.image_size)

    def structure_masks(self, classes: torch.Tensor) -> list[torch.Tensor]:
        """Per-structure binary masks of a class map (structures are nested)."""
        if self.name == "odoc":
            return [classes >= 1, classes == 2]
        return [classes >= 1]

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Per-structure DSC of one image, as ``{"<structure>_dsc": value}``."""
        from common.protocol import image_dice

        return {
            f"{name}_dsc": image_dice(predicted, expected)
            for name, predicted, expected in zip(
                self.structures, self.structure_masks(prediction), self.structure_masks(target)
            )
        }

    def domain_summary(
        self, domain: str, dataset: str, rows: list[dict[str, float]]
    ) -> dict[str, object]:
        """Domain means in percent, in the field order of ``DomainResult``."""
        keys = [f"{name}_dsc" for name in self.structures]
        means = {key: 100.0 * float(np.mean([row[key] for row in rows])) for key in keys}
        return {
            "domain": domain,
            "dataset": dataset,
            "images": len(rows),
            **means,
            "mean_dsc": float(np.mean(list(means.values()))),
        }

    def describe(self, summary: dict[str, object]) -> str:
        labels = {"disc": "OD", "cup": "OC", "prostate": "Prostate"}
        scores = ", ".join(
            f"{labels[name]}={summary[f'{name}_dsc']:.2f}" for name in self.structures
        )
        if len(self.structures) > 1:
            scores += f", Mean={summary['mean_dsc']:.2f}"
        return (
            f"Domain {summary['domain']} ({summary['dataset']}, n={summary['images']}): {scores}"
        )


TASKS = {
    "odoc": Task("odoc", 3, ("disc", "cup"), 512, tuple("BCDE"), build_domains),
    "prostate": Task("prostate", 2, ("prostate",), 384, tuple("BCDEF"), build_prostate_domains),
}
