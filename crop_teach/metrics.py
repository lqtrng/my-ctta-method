"""Per-image scores: Dice per structure (all problems); S-alpha and max E-phi (polyp, as in the
ReDi-FM paper's Table 2), computed on the foreground probability map as in py_sod_metrics."""

from __future__ import annotations

import numpy as np
import torch

EPS = np.spacing(1)


def structure_masks(classes: torch.Tensor, num_structures: int) -> list[torch.Tensor]:
    """Nested structures of a class map: structure k is class >= k + 1."""
    return [classes >= index + 1 for index in range(num_structures)]


def dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction, target = prediction.flatten().double(), target.flatten().double()
    return float((2.0 * (prediction * target).sum() + 1e-6) / (prediction.sum() + target.sum() + 1e-6))


def _normalize(probability: np.ndarray) -> np.ndarray:
    low, high = probability.min(), probability.max()
    return (probability - low) / (high - low) if high > low else probability


# --- S-alpha (Fan et al., ICCV 2017), alpha = 0.5 ------------------------------------------


def _s_object(prediction: np.ndarray, gt: np.ndarray) -> float:
    values = prediction[gt]
    if values.size == 0:
        return 0.0
    mean = values.mean()
    std = values.std(ddof=1) if values.size > 1 else 0.0
    return 2 * mean / (mean ** 2 + 1 + std + EPS)


def _ssim(prediction: np.ndarray, gt: np.ndarray) -> float:
    count = prediction.size
    if count < 2:
        return 1.0 if count == 0 or prediction.mean() == gt.mean() else 0.0
    x, y = prediction.mean(), gt.mean()
    sigma_x = ((prediction - x) ** 2).sum() / (count - 1)
    sigma_y = ((gt - y) ** 2).sum() / (count - 1)
    sigma_xy = ((prediction - x) * (gt - y)).sum() / (count - 1)
    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)
    if alpha != 0:
        return alpha / (beta + EPS)
    return 1.0 if beta == 0 else 0.0


def s_measure(probability: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    prediction, gt = _normalize(probability.astype(np.float64)), gt.astype(bool)
    y = gt.mean()
    if y == 0:
        return float(1 - prediction.mean())
    if y == 1:
        return float(prediction.mean())
    s_object = y * _s_object(prediction, gt) + (1 - y) * _s_object(1 - prediction, ~gt)
    height, width = gt.shape
    rows, columns = np.nonzero(gt)
    cy, cx = int(round(rows.mean())) + 1, int(round(columns.mean())) + 1
    area = height * width
    weights = [cx * cy / area, (width - cx) * cy / area, cx * (height - cy) / area]
    weights.append(1 - sum(weights))
    blocks = [(slice(0, cy), slice(0, cx)), (slice(0, cy), slice(cx, width)),
              (slice(cy, height), slice(0, cx)), (slice(cy, height), slice(cx, width))]
    s_region = sum(w * _ssim(prediction[b], gt[b].astype(np.float64)) for w, b in zip(weights, blocks))
    return float(max(alpha * s_object + (1 - alpha) * s_region, 0.0))


# --- max E-phi (Fan et al., IJCAI 2018), over 256 thresholds ---------------------------------


def max_e_measure(probability: np.ndarray, gt: np.ndarray) -> float:
    prediction = (_normalize(probability.astype(np.float64)) * 255).astype(np.uint8)
    gt = gt.astype(bool)
    total, gt_fg = gt.size, int(gt.sum())
    bins = np.linspace(0, 256, 257)
    fg_hist = np.histogram(prediction[gt], bins=bins)[0]
    bg_hist = np.histogram(prediction[~gt], bins=bins)[0]
    fg_fg = np.cumsum(np.flip(fg_hist))  # index i: pixels >= 255 - i
    fg_bg = np.cumsum(np.flip(bg_hist))
    pred_fg = fg_fg + fg_bg
    pred_bg = total - pred_fg
    if gt_fg == 0:
        enhanced = pred_bg.astype(np.float64)
    elif gt_fg == total:
        enhanced = pred_fg.astype(np.float64)
    else:
        bg_fg = gt_fg - fg_fg
        bg_bg = pred_bg - bg_fg
        mean_pred, mean_gt = pred_fg / total, gt_fg / total
        parts = [(fg_fg, 1 - mean_pred, 1 - mean_gt), (fg_bg, 1 - mean_pred, -mean_gt),
                 (bg_fg, -mean_pred, 1 - mean_gt), (bg_bg, -mean_pred, -mean_gt)]
        enhanced = np.zeros(256)
        for count, p, g in parts:
            align = 2 * p * g / (p ** 2 + g ** 2 + EPS)
            enhanced += ((align + 1) ** 2 / 4) * count
    return float((enhanced / (total - 1 + EPS)).max())


def score(prediction: torch.Tensor, probability: torch.Tensor, target: torch.Tensor,
          structures: tuple[str, ...], polyp_metrics: bool) -> dict[str, float]:
    """prediction, target: class maps (H, W); probability: foreground probability (H, W)."""
    scores = {f"{name}_dsc": dice(p, t) for name, p, t in zip(
        structures, structure_masks(prediction, len(structures)), structure_masks(target, len(structures)))}
    if polyp_metrics:
        probability, gt = probability.cpu().numpy(), (target >= 1).cpu().numpy()
        scores["s_alpha"] = s_measure(probability, gt)
        scores["e_phi_max"] = max_e_measure(probability, gt)
    return scores
