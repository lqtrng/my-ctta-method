"""Sharp vs blurry crop ResUNet-34 on their own: both get the same crop, placed from the
ground-truth disc (2x its extent), so crop placement and false DINOv3 regions play no part.
Old model: crop cut from the 512 image.  Sharp model: same box cut from the full-size image."""
import argparse, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src")
from baselines.dinov3_crop import CropPlacer
from common.models import build_network
from common.paths import RESUNET_CROP_CHECKPOINT, RESUNET_SHARP_CROP_CHECKPOINT
from common.protocol import image_to_tensor, prepare_sample
from common.tasks import TASKS

task = TASKS["odoc"]; device = torch.device("cuda")
models = {}
for name, path in (("blurry", RESUNET_CROP_CHECKPOINT), ("sharp", RESUNET_SHARP_CROP_CHECKPOINT)):
    model = build_network("resunet34", 3, 512); model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    models[name] = model.to(device).eval()
placer = CropPlacer(argparse.Namespace(crop_region="largest", crop_scale=2.0, min_crop=96))

def predict(model, source, image, box):
    top, left, side = box
    sy, sx = source.shape[-2] / 512, source.shape[-1] / 512
    crop = source[:, :, round(top * sy):round((top + side) * sy), round(left * sx):round((left + side) * sx)]
    crop = F.interpolate(crop, size=(512, 512), mode="bilinear", align_corners=False, antialias=source is not image)
    probabilities = F.softmax(model(crop), dim=1)
    pasted = torch.zeros(1, 3, 512, 512, device=device); pasted[:, 0] = 1
    pasted[:, :, top:top + side, left:left + side] = F.interpolate(probabilities, size=(side, side), mode="bilinear", align_corners=False)
    return pasted.argmax(dim=1)[0].cpu()

results = {}
with torch.inference_mode():
    for domain, samples in task.build_domains(list(task.target_domains), None).items():
        for sample in samples:
            image, target = prepare_sample(sample, 512)
            small = image_to_tensor(image, device); full = image_to_tensor(image.info["original"], device)
            truth = target.numpy() >= 1
            box = placer.crop_box(truth, truth.astype(np.float32))
            for name, source in (("blurry", small), ("sharp", full)):
                scores = task.score(predict(models[name], source, small, box), target)
                results.setdefault((domain, name), []).append(scores)
        print(domain, " | ".join(f"{name}: " + " ".join(f"{k}={100*np.mean([s[k] for s in results[(domain, name)]]):.2f}"
              for k in results[(domain, name)][0]) for name in ("blurry", "sharp")), flush=True)
