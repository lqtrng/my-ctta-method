"""Class maps <-> nested structures.

Classes: 0 = background, 1.. = foreground classes. Structures are nested: structure k is
"class >= k + 1". OD/OC: (disc = rim + cup, cup); prostate and polyp: one foreground.
"""

from __future__ import annotations

import torch

EPS = 1e-6


def structure_probabilities(classes: torch.Tensor) -> torch.Tensor:
    """Class probabilities (B, C, H, W) -> nested structure probabilities (B, C - 1, H, W)."""
    if classes.shape[1] == 2:
        return classes[:, 1:2].clamp(EPS, 1.0 - EPS)
    disc = classes[:, 1] + classes[:, 2]
    return torch.stack((disc, classes[:, 2]), dim=1).clamp(EPS, 1.0 - EPS)


def nested_classes(structures: torch.Tensor) -> torch.Tensor:
    """Nested structure probabilities -> class probabilities (the inverse of the above)."""
    if structures.shape[1] == 1:
        foreground = structures[:, 0].clamp(EPS, 1.0 - EPS)
        return torch.stack((1.0 - foreground, foreground), dim=1)
    disc = structures[:, 0].clamp(EPS, 1.0 - EPS)
    cup = (structures[:, 1] * disc).clamp(EPS, 1.0 - EPS)
    classes = torch.stack((1.0 - disc, (disc - cup).clamp_min(EPS), cup), dim=1)
    return classes / classes.sum(dim=1, keepdim=True).clamp_min(EPS)


def structure_masks(class_map: torch.Tensor, num_structures: int) -> list[torch.Tensor]:
    """Class-index map (H, W) -> one boolean mask per nested structure."""
    return [class_map >= index + 1 for index in range(num_structures)]
