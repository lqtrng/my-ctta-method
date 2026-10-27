#!/usr/bin/env python3
"""Visualize ReDi-FM's OD/OC point prompts and all SAM candidates.

The example reproduces the prompting path in redifm_sam/adapt.py:

1. A frozen source ResUNet-34 predicts optic-disc (OD) and optic-cup (OC)
   probabilities.
2. For each structure, a center point and a hard point are extracted from the
   thresholded source prediction.
3. Both points are sent to SAM as positive prompts.
4. SAM returns three masks, and the candidate with the highest Dice agreement
   with the source pseudo-label is selected.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.models import NUM_CLASSES, ResUnet34  # noqa: E402
from common.paths import PROJECT_DIR, SAM_CHECKPOINT  # noqa: E402
from common.protocol import binary_dice_numpy, image_to_tensor  # noqa: E402
from common.redifm import (  # noqa: E402
    largest_component,
    structure_aware_points,
    structure_probabilities,
)


DEFAULT_IMAGE = (
    PROJECT_DIR
    / "data/od_oc/domain_B_REFUGE_train/train/Images_Cropped/g0010.jpg"
)
DEFAULT_SOURCE_CHECKPOINT = (
    PROJECT_DIR / "checkpoints/domain_A_ResUNet34_last_20261017_215955.pth"
)
DEFAULT_SAM_CHECKPOINT = SAM_CHECKPOINT
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

STRUCTURES = (("OD", 0, (255, 145, 30)), ("OC", 1, (75, 145, 255)))
CENTER_COLOR = (40, 220, 70)
HARD_COLOR = (0, 235, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Ground-truth mask; inferred from the image path when omitted",
    )
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--sam-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def draw_center_point(image: Image.Image, point: np.ndarray) -> None:
    draw = ImageDraw.Draw(image)
    radius = max(7, round(min(image.size) * 0.018))
    width = max(3, radius // 3)
    center_x, center_y = map(float, point)
    draw.ellipse(
        (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ),
        fill=CENTER_COLOR,
        outline="white",
        width=width,
    )


def draw_hard_point(image: Image.Image, point: np.ndarray) -> None:
    draw = ImageDraw.Draw(image)
    radius = max(7, round(min(image.size) * 0.018))
    width = max(3, radius // 3)
    hard_x, hard_y = map(float, point)
    arm = radius + 2
    draw.line(
        (hard_x - arm, hard_y - arm, hard_x + arm, hard_y + arm),
        fill="white",
        width=width + 3,
    )
    draw.line(
        (hard_x - arm, hard_y + arm, hard_x + arm, hard_y - arm),
        fill="white",
        width=width + 3,
    )
    draw.line(
        (hard_x - arm, hard_y - arm, hard_x + arm, hard_y + arm),
        fill=HARD_COLOR,
        width=width,
    )
    draw.line(
        (hard_x - arm, hard_y + arm, hard_x + arm, hard_y - arm),
        fill=HARD_COLOR,
        width=width,
    )


def draw_points(image: Image.Image, points: np.ndarray) -> None:
    draw_center_point(image, points[0])
    draw_hard_point(image, points[1])


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.48,
) -> Image.Image:
    base = np.asarray(image, dtype=np.float32)
    tint = np.empty_like(base)
    tint[...] = np.asarray(color, dtype=np.float32)
    opacity = (mask.astype(np.float32) * alpha)[..., None]
    result = base * (1.0 - opacity) + tint * opacity
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def make_panel(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    header_height = 76
    panel = Image.new("RGB", (image.width, image.height + header_height), "white")
    panel.paste(image, (0, header_height))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 9), title, fill="black", font=load_font(21))
    if subtitle:
        draw.text((12, 41), subtitle, fill=(55, 55, 55), font=load_font(15))
    return panel


def make_original_panel(
    image: Image.Image, structure: str, dataset_label: str
) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    draw.rounded_rectangle((12, 12, 310, 92), radius=8, fill=(0, 0, 0, 185))
    draw.ellipse((27, 29, 47, 49), fill=CENTER_COLOR, outline="white", width=2)
    draw.text((58, 27), "Center point", fill="white", font=load_font(17))
    draw.line((28, 69, 46, 87), fill=HARD_COLOR, width=5)
    draw.line((28, 87, 46, 69), fill=HARD_COLOR, width=5)
    draw.text((58, 65), "Hard point", fill="white", font=load_font(17))
    return make_panel(
        marked,
        f"Original fundus image: {structure}",
        f"{dataset_label} | Prompt legend",
    )


def deviation_visualization(
    image: Image.Image, component: np.ndarray
) -> tuple[Image.Image, float, float]:
    """Color foreground pixels by |intensity - foreground mean intensity|."""
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    prototype = float(grayscale[component].mean())
    deviation = np.zeros_like(grayscale)
    deviation[component] = np.abs(grayscale[component] - prototype)
    maximum = float(deviation.max())
    normalized = deviation / max(maximum, 1e-7)

    # Blue -> cyan -> yellow -> red: red pixels deviate most from the prototype.
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * normalized - 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    heatmap = np.stack((red, green, blue), axis=-1) * 255.0

    base = np.asarray(image, dtype=np.float32) * 0.25
    output = base.copy()
    output[component] = (
        0.20 * np.asarray(image, dtype=np.float32)[component]
        + 0.80 * heatmap[component]
    )
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), prototype, maximum


def infer_ground_truth_path(image_path: Path) -> Path:
    mask_directory = image_path.parent.name.replace("Images", "Masks", 1)
    return image_path.parent.parent / mask_directory / f"{image_path.stem}.png"


def load_ground_truth(path: Path, image_size: int) -> np.ndarray:
    raw = np.asarray(
        Image.open(path)
        .convert("L")
        .resize((image_size, image_size), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    if set(np.unique(raw).tolist()) <= {0, 1, 2}:
        return raw

    # REFUGE-style raw convention: 255=background, 128=disc rim, 0=cup.
    target = np.zeros_like(raw)
    target[raw < 255] = 1
    target[raw == 0] = 2
    return target


@torch.inference_mode()
def source_predictions(
    image: Image.Image, checkpoint: Path, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model = ResUnet34(num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False).eval()

    classes = F.softmax(model(image_to_tensor(image, torch.device("cpu"))), dim=1)
    soft = structure_probabilities(classes)[0].cpu().numpy()
    return soft, soft > threshold


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    image = Image.open(args.image).convert("RGB").resize(
        (args.image_size, args.image_size), Image.Resampling.BILINEAR
    )
    ground_truth_path = args.ground_truth or infer_ground_truth_path(args.image)
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground truth not found: {ground_truth_path}. "
            "Pass its location with --ground-truth."
        )
    ground_truth = load_ground_truth(ground_truth_path, args.image_size)
    dataset_label = (
        "Domain B / REFUGE" if "domain_B_REFUGE_train" in str(args.image) else "Dataset"
    )
    source_soft, source_binary = source_predictions(
        image, args.source_checkpoint, args.threshold
    )

    from segment_anything import SamPredictor, sam_model_registry

    sam_device = torch.device(args.sam_device)
    print(f"Loading SAM ViT-H on {sam_device} ...")
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint))
    sam.requires_grad_(False).to(sam_device).eval()
    predictor = SamPredictor(sam)
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if sam_device.type == "cuda"
        else nullcontext()
    )

    raw_results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, int]] = []
    try:
        with torch.inference_mode(), amp_context:
            predictor.set_image(np.asarray(image, dtype=np.uint8))
            for name, index, _ in STRUCTURES:
                points = structure_aware_points(
                    image, source_binary[index], source_soft[index]
                )
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=np.ones(2, dtype=np.int32),
                    multimask_output=True,
                    return_logits=False,
                )
                agreements = np.asarray(
                    [binary_dice_numpy(mask, source_binary[index]) for mask in masks]
                )
                selected = int(np.argmax(agreements))
                raw_results.append((name, points, masks, scores, selected))
    finally:
        predictor.reset_image()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for (name, structure_index, color), result in zip(STRUCTURES, raw_results):
        _, points, masks, scores, selected = result
        component = largest_component(source_binary[structure_index])
        if not component.any():
            row, column = np.unravel_index(
                np.argmax(source_soft[structure_index]),
                source_soft[structure_index].shape,
            )
            component[row, column] = True

        source_overlay = overlay_mask(
            image, source_binary[structure_index], color, alpha=0.48
        )
        source_panel = make_panel(
            source_overlay,
            f"1. Source {name} pseudo-label",
            f"Foreground = probability > {args.threshold:.2f}",
        )

        ground_truth_mask = (
            ground_truth >= 1 if name == "OD" else ground_truth == 2
        )
        ground_truth_overlay = overlay_mask(
            image, ground_truth_mask, color, alpha=0.48
        )
        ground_truth_panel = make_panel(
            ground_truth_overlay,
            f"{dataset_label} ground truth: {name}",
            f"{ground_truth_path.parent.name}/{ground_truth_path.name}; not used for prompt",
        )

        center_image = overlay_mask(image, component, color, alpha=0.42)
        draw_center_point(center_image, points[0])
        center = tuple(map(int, points[0]))
        center_panel = make_panel(
            center_image,
            "2. Why this is the center point",
            f"Centroid of largest component = {center}",
        )

        hard_image, prototype, maximum_deviation = deviation_visualization(
            image, component
        )
        draw_hard_point(hard_image, points[1])
        hard = tuple(map(int, points[1]))
        hard_panel = make_panel(
            hard_image,
            "3. Why this is the hard point",
            f"max |gray - mean| at {hard}; mean={prototype:.3f}, max={maximum_deviation:.3f}",
        )

        prompt_image = image.copy()
        draw_points(prompt_image, points)
        prompt_panel = make_panel(
            prompt_image,
            "4. Exact prompt sent to SAM",
            f"Two positive points: center={center}, hard={hard}",
        )

        candidate_panels: list[Image.Image] = []
        agreements = [
            binary_dice_numpy(mask, source_binary[structure_index]) for mask in masks
        ]
        ground_truth_dice = [
            binary_dice_numpy(mask, ground_truth_mask) for mask in masks
        ]
        for candidate_index, (mask, score, agreement, gt_dice) in enumerate(
            zip(masks, scores, agreements, ground_truth_dice), start=1
        ):
            candidate = overlay_mask(image, mask, color)
            draw_points(candidate, points)
            selected_text = "  SELECTED" if candidate_index - 1 == selected else ""
            candidate_panels.append(
                make_panel(
                    candidate,
                    f"SAM candidate {candidate_index}{selected_text}",
                    f"SAM={score:.3f}  source Dice={agreement:.3f}  GT Dice={gt_dice:.3f}",
                )
            )

        panels = [
            make_original_panel(image, name, dataset_label),
            ground_truth_panel,
            source_panel,
            center_panel,
            hard_panel,
            prompt_panel,
            *candidate_panels,
        ]
        panel_width, panel_height = panels[0].size
        sheet = Image.new(
            "RGB", (panel_width * 3, panel_height * 3), (225, 228, 232)
        )
        for panel_index, panel in enumerate(panels):
            sheet.paste(
                panel,
                ((panel_index % 3) * panel_width, (panel_index // 3) * panel_height),
            )

        output_path = args.output_dir / f"sam_{name.lower()}_prompt_explained.png"
        sheet.save(output_path)
        output_paths.append(output_path)

    print(f"Input: {args.image}")
    print(f"Ground truth: {ground_truth_path}")
    for name, points, _, _, selected in raw_results:
        center = tuple(map(int, points[0]))
        hard = tuple(map(int, points[1]))
        print(
            f"{name}: center={center}, hard={hard}, "
            f"selected=candidate {selected + 1}"
        )
    for output_path in output_paths:
        print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
