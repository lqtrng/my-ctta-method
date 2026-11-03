"""Does DINOv3 + full-image head miss discs (low recall) or add wrong regions (low precision)?

Each connected region of the predicted disc mask (OD+OC probability > 0.5) is one crop,
as in ``dinov3_crop --crop-region all``.  Uses the cached predictions of
``extras/loss_search/cache_supervision.py`` on OD/OC targets B-E.
"""
import argparse, json, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, "src")
from baselines.dinov3_crop import CropPlacer

C = ".cache/loss_search/odoc/"
index = json.load(open(C + "index.json"))
targets = np.load(C + "targets.npy", mmap_mode="r")
source = np.load(C + "source_classes.npy", mmap_mode="r")
placer = CropPlacer(argparse.Namespace(crop_region="all", crop_scale=2.0, min_crop=96))

def box(mask):
    rows, cols = np.nonzero(mask)
    return rows.min(), cols.min(), rows.max() + 1, cols.max() + 1

def iou(a, b):
    h = max(0, min(a[2], b[2]) - max(a[0], b[0])); w = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = h * w
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

stats = {}
for i, row in enumerate(index):
    truth = np.asarray(targets[i]) >= 1
    disc = (source[i, 1].astype(np.float32) + source[i, 2].astype(np.float32))
    mask = disc > 0.5
    components, count = ndimage.label(mask)
    s = stats.setdefault(row["domain"], dict(images=0, regions=0, none=0, hit_regions=0, hit_regions_iou=0,
                                             found=0, found_iou=0, contained=0, with_false=0, only_false=0))
    s["images"] += 1; s["regions"] += count; s["none"] += count == 0
    truth_box = box(truth)
    hits = 0
    contained = False
    for k in range(1, count + 1):
        region = components == k
        hit = (region & truth).any()
        hits += hit
        s["hit_regions_iou"] += iou(box(region), truth_box) >= 0.5
        top, left, side = placer.crop_box(region, disc)
        inside = np.zeros_like(truth); inside[top:top + side, left:left + side] = True
        contained |= (truth & inside).sum() >= 0.999 * truth.sum()
    s["hit_regions"] += hits
    s["found"] += hits > 0
    s["found_iou"] += any(iou(box(components == k), truth_box) >= 0.5 for k in range(1, count + 1))
    s["contained"] += contained
    s["with_false"] += count - hits > 0
    s["only_false"] += count > 0 and hits == 0

print("dom | regions/img | no region | RECALL: disc touched, box IoU>=.5, crop holds whole disc"
      " | PRECISION: regions on disc, box IoU>=.5 | images with a false region")
total = {k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}
for name, s in [*sorted(stats.items()), ("ALL", total)]:
    n, r = s["images"], max(s["regions"], 1)
    print(f"{name:3} | {s['regions']/n:4.2f} | {100*s['none']/n:5.1f}% | {100*s['found']/n:5.1f}% {100*s['found_iou']/n:5.1f}%"
          f" {100*s['contained']/n:5.1f}% | {100*s['hit_regions']/r:5.1f}% {100*s['hit_regions_iou']/r:5.1f}%"
          f" | {100*s['with_false']/n:5.1f}% (only false: {100*s['only_false']/n:4.1f}%)")
