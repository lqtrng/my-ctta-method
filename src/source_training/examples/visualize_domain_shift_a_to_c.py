#!/usr/bin/env python3
"""Visualize the Domain A -> Domain C appearance shift on two chosen samples."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.models import ResUnet34  # noqa: E402
from common.paths import DEFAULT_SOURCE_CHECKPOINT, PROJECT_DIR  # noqa: E402
from source_training.train import DiceCrossEntropyLoss  # noqa: E402


DEFAULT_DOMAIN_A_IMAGE = (
    PROJECT_DIR
    / "data/od_oc/domain_A_RIM_ONE_r3/RIM-ONE-r3/testing/images/G-25-L.jpg"
)
DEFAULT_DOMAIN_A_MASK = (
    PROJECT_DIR
    / "data/od_oc/domain_A_RIM_ONE_r3/RIM-ONE-r3/testing/masks/G-25-L.png"
)
DEFAULT_DOMAIN_C_IMAGE = PROJECT_DIR / "data/od_oc/domain_C_ORIGA/Images/004.jpg"
DEFAULT_DOMAIN_C_MASK = PROJECT_DIR / "data/od_oc/domain_C_ORIGA/Masks/004.png"
DEFAULT_CHECKPOINT = DEFAULT_SOURCE_CHECKPOINT
DEFAULT_OUTPUT = Path(__file__).with_name("domain_shift_G-25-L_to_004.png")

OD_COLOR = np.asarray((255, 145, 30), dtype=np.float32)
OC_COLOR = np.asarray((65, 150, 255), dtype=np.float32)
NUM_CLASSES = 3


@dataclass(frozen=True)
class Sample:
    domain: str
    dataset: str
    identifier: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-a-image", type=Path, default=DEFAULT_DOMAIN_A_IMAGE)
    parser.add_argument("--domain-a-mask", type=Path, default=DEFAULT_DOMAIN_A_MASK)
    parser.add_argument("--domain-c-image", type=Path, default=DEFAULT_DOMAIN_C_IMAGE)
    parser.add_argument("--domain-c-mask", type=Path, default=DEFAULT_DOMAIN_C_MASK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def normalize_mask(raw: np.ndarray) -> np.ndarray:
    if set(np.unique(raw).tolist()) <= {0, 1, 2}:
        return raw.astype(np.uint8, copy=True)
    target = np.zeros_like(raw, dtype=np.uint8)
    target[raw < 255] = 1
    target[raw == 0] = 2
    return target


def load_sample(sample: Sample, size: int) -> tuple[Image.Image, np.ndarray, tuple[int, int]]:
    with Image.open(sample.image_path) as opened:
        original_size = opened.size
        image = opened.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    with Image.open(sample.mask_path) as opened:
        raw = np.asarray(
            opened.convert("L").resize((size, size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return image, normalize_mask(raw), original_size


def model_input_image(image: Image.Image) -> Image.Image:
    """Apply the same per-image min-max normalization as image_to_tensor()."""
    array = np.asarray(image, dtype=np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    normalized = (array - minimum) / max(maximum - minimum, 1e-7)
    return Image.fromarray(np.clip(normalized * 255.0, 0, 255).astype(np.uint8))


def overlay_segmentation(image: Image.Image, target: np.ndarray) -> Image.Image:
    base = np.asarray(image, dtype=np.float32)
    result = base.copy()
    disc = target >= 1
    cup = target == 2
    result[disc] = 0.62 * result[disc] + 0.38 * OD_COLOR
    result[cup] = 0.48 * result[cup] + 0.52 * OC_COLOR
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def panel(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    header = 72
    output = Image.new("RGB", (image.width, image.height + header), "white")
    output.paste(image, (0, header))
    draw = ImageDraw.Draw(output)
    draw.text((12, 8), title, fill="black", font=font(21))
    draw.text((12, 40), subtitle, fill=(65, 65, 65), font=font(14))
    return output


def image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    minimum = float(array.min())
    maximum = float(array.max())
    array = (array - minimum) / max(maximum - minimum, 1e-7)
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    return float(
        (2.0 * np.logical_and(prediction, target).sum() + 1e-6)
        / (prediction.sum() + target.sum() + 1e-6)
    )


@torch.inference_mode()
def predict(
    model: ResUnet34,
    criterion: DiceCrossEntropyLoss,
    image: Image.Image,
    target: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    target_tensor = torch.from_numpy(target.copy()).long().unsqueeze(0)
    logits = model(image_tensor(image))
    loss = float(criterion(logits, target_tensor))
    prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    od_dice = binary_dice(prediction >= 1, target >= 1)
    oc_dice = binary_dice(prediction == 2, target == 2)
    return prediction, loss, od_dice, oc_dice


def main() -> None:
    args = parse_args()
    domain_a = Sample(
        "Domain A", "RIM-ONE-r3", "G-25-L", args.domain_a_image, args.domain_a_mask
    )
    domain_c = Sample("Domain C", "ORIGA", "004", args.domain_c_image, args.domain_c_mask)
    samples = (domain_a, domain_c)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = ResUnet34(num_classes=NUM_CLASSES)
    state_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False).eval()
    criterion = DiceCrossEntropyLoss(NUM_CLASSES)

    loaded = [load_sample(sample, args.image_size) for sample in samples]
    inputs = [model_input_image(image) for image, _, _ in loaded]
    predictions = [
        predict(model, criterion, image, target) for image, target, _ in loaded
    ]

    rows: list[list[Image.Image]] = []
    for sample_number, (sample, (_, target, _), input_image, prediction_result) in enumerate(
        zip(samples, loaded, inputs, predictions), start=1
    ):
        prediction, loss, od_dice, oc_dice = prediction_result
        display_name = f"Sample {sample_number}"
        rows.append(
            [
                panel(
                    input_image,
                    f"{display_name}: model input",
                    "Full image -> 512x512 | min-max normalized",
                ),
                panel(
                    overlay_segmentation(input_image, target),
                    f"{display_name}: ground truth",
                    "Orange = optic disc; blue = optic cup",
                ),
                panel(
                    overlay_segmentation(input_image, prediction),
                    f"{display_name}: checkpoint prediction",
                    f"CE+Dice loss {loss:.3f} | OD {od_dice:.3f} | OC {oc_dice:.3f}",
                ),
            ]
        )

    panel_width, panel_height = rows[0][0].size
    title_height = 90
    sheet = Image.new(
        "RGB", (panel_width * 3, title_height + panel_height * 2), (230, 232, 235)
    )
    title_draw = ImageDraw.Draw(sheet)
    title_draw.text(
        (20, 12), "Appearance shift: Sample 1 -> Sample 2", fill="black", font=font(29)
    )
    title_draw.text(
        (20, 53),
        "Same preprocessing and source-trained checkpoint; compare prediction quality after appearance shift",
        fill=(55, 55, 55),
        font=font(16),
    )
    for row_index, row in enumerate(rows):
        for column_index, item in enumerate(row):
            sheet.paste(
                item,
                (column_index * panel_width, title_height + row_index * panel_height),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Sample 1 image: {domain_a.image_path}")
    print(f"Sample 2 image: {domain_c.image_path}")
    for sample_number, (_, loss, od_dice, oc_dice) in enumerate(predictions, start=1):
        print(
            f"Sample {sample_number}: CE+Dice loss={loss:.4f}, "
            f"OD Dice={od_dice:.4f}, OC Dice={oc_dice:.4f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
