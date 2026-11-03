"""Polyp, no adaptation: sharp-crop nnU-Net given perfect crops vs whole-image nnU-Net.

Perfect crops: one per ground-truth polyp region, 2x its extent (min 96), the rule of
``DinoV3Segmenter.crop_box``; pixels from the full-size image; where crops overlap each
pixel keeps the crop most sure of foreground, outside every crop is background."""
import argparse, sys
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage
sys.path.insert(0, "src")
from baselines.dinov3_crop import CropPlacer
from common.models import build_network
from common.paths import POLYP_NNUNET_OFFICIAL_SOURCE_CHECKPOINT, POLYP_NNUNET_SHARP_CROP_CHECKPOINT
from common.protocol import image_to_tensor, prepare_sample
from common.tasks import TASKS

task = TASKS["polyp"]; size = task.image_size; device = torch.device("cuda")
def load(path):
    model = build_network("nnunet_official", task.num_classes, size)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
crop_model, full_model = load(POLYP_NNUNET_SHARP_CROP_CHECKPOINT), load(POLYP_NNUNET_OFFICIAL_SOURCE_CHECKPOINT)
placer = CropPlacer(argparse.Namespace(crop_region="all", crop_scale=2.0, min_crop=96))

def crop_prediction(small, full, truth):
    components, count = ndimage.label(truth)
    merged = None
    for k in range(1, count + 1):
        top, left, side = placer.crop_box(components == k, truth.astype(np.float32))
        sy, sx = full.shape[-2] / size, full.shape[-1] / size
        crop = full[:, :, round(top * sy):round((top + side) * sy), round(left * sx):round((left + side) * sx)]
        crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        pasted = torch.zeros(1, task.num_classes, size, size, device=device); pasted[:, 0] = 1
        pasted[:, :, top:top + side, left:left + side] = F.interpolate(
            F.softmax(crop_model(crop), dim=1), size=(side, side), mode="bilinear", align_corners=False)
        merged = pasted if merged is None else torch.where(pasted[:, 1:] > merged[:, 1:], pasted, merged)
    return merged.argmax(dim=1)[0].cpu()

totals = {}
with torch.inference_mode():
    for domain, samples in task.build_domains(list(task.target_domains), None).items():
        scores = {"crop (perfect crops)": [], "full image": []}
        for sample in samples:
            image, target = prepare_sample(sample, size)
            small = image_to_tensor(image, device); full = image_to_tensor(image.info["original"], device)
            truth = target.numpy() >= 1
            scores["full image"].append(task.score(full_model(small).argmax(dim=1)[0].cpu(), target))
            scores["crop (perfect crops)"].append(task.score(crop_prediction(small, full, truth), target))
        for name, rows in scores.items():
            value = 100 * np.mean([np.mean(list(r.values())) for r in rows])
            totals.setdefault(name, []).append(value)
            print(f"{domain} {name:22} {value:6.2f}", flush=True)
for name, values in totals.items():
    print(f"Do. A {name:22} {np.mean(values):6.2f}")
