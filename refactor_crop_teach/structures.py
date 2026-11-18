from __future__ import annotations

import torch


def structure_probabilities(classes: torch.Tensor) -> torch.Tensor:
    if classes.shape[1] == 2:
        return classes[:, 1:2].clamp(1e-6, 1.0 - 1e-6)
    if classes.shape[1] == 3:
        disc = classes[:, 1] + classes[:, 2]
        return torch.stack((disc, classes[:, 2]), dim=1).clamp(1e-6, 1.0 - 1e-6)
    return classes.flip(1).cumsum(1).flip(1)[:, 1:].clamp(1e-6, 1.0 - 1e-6)


def nested_classes(structures: torch.Tensor) -> torch.Tensor:
    if structures.shape[1] == 1:
        foreground = structures[:, 0].clamp(1e-6, 1.0 - 1e-6)
        return torch.stack((1.0 - foreground, foreground), dim=1)
    if structures.shape[1] == 2:
        disc = structures[:, 0].clamp(1e-6, 1.0 - 1e-6)
        cup = (structures[:, 1] * disc).clamp(1e-6, 1.0 - 1e-6)
        classes = torch.stack((1.0 - disc, (disc - cup).clamp_min(1e-6), cup), dim=1)
        return classes / classes.sum(dim=1, keepdim=True).clamp_min(1e-6)
    cumulative = structures[:, 0].clamp(1e-6, 1.0 - 1e-6)
    inside = [cumulative]
    for index in range(1, structures.shape[1]):
        cumulative = (structures[:, index] * cumulative).clamp(1e-6, 1.0 - 1e-6)
        inside.append(cumulative)
    channels = [1.0 - inside[0]]
    for outer, inner in zip(inside, inside[1:]):
        channels.append((outer - inner).clamp_min(1e-6))
    channels.append(inside[-1])
    stacked = torch.stack(channels, dim=1)
    return stacked / stacked.sum(dim=1, keepdim=True).clamp_min(1e-6)


def structure_masks(classes: torch.Tensor, num_structures: int) -> list[torch.Tensor]:
    return [classes >= index + 1 for index in range(num_structures)]
