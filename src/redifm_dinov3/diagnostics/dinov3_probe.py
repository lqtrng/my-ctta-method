"""Offline diagnostic: can frozen DINOv3 refine source OD/OC pseudo-labels on domain B?"""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, timm
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.protocol import prepare_sample, image_to_tensor, binary_dice_numpy
from common.redifm import structure_probabilities, largest_component
from common.models import NUM_CLASSES, ResUnet34
from common.data import build_domains
from common.paths import DEFAULT_SOURCE_CHECKPOINT, MODEL_DIR, dinov3_weights

dev = torch.device("cuda")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
arch = sys.argv[2] if len(sys.argv) > 2 else "vit_large_patch16_dinov3.lvd1689m"
CROP_RES = int(sys.argv[3]) if len(sys.argv) > 3 else 512
SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0

vfm = timm.create_model(arch.split(".")[0], pretrained=True, dynamic_img_size=True,
    pretrained_cfg_overlay=dict(file=str(dinov3_weights(MODEL_DIR, arch)))).to(dev).eval()
src = ResUnet34(NUM_CLASSES); src.load_state_dict(torch.load(
    DEFAULT_SOURCE_CHECKPOINT, map_location="cpu", weights_only=True)); src.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def crop_box(disc_mask, prob, size=512):
    comp = largest_component(disc_mask)
    if not comp.any():
        r, c = np.unravel_index(np.argmax(prob), prob.shape); comp[r, c] = True
    rows, cols = np.nonzero(comp)
    cy, cx = (rows.min() + rows.max()) / 2, (cols.min() + cols.max()) / 2
    side = max(np.ptp(rows) + 1, np.ptp(cols) + 1) * SCALE
    side = int(np.clip(side, 96, size))
    y0 = int(np.clip(round(cy - side / 2), 0, size - side)); x0 = int(np.clip(round(cx - side / 2), 0, size - side))
    return y0, x0, side

@torch.no_grad()
def feats(img_t, box):
    y0, x0, s = box
    crop = img_t[:, :, y0:y0 + s, x0:x0 + s]
    crop = F.interpolate(crop, size=(CROP_RES, CROP_RES), mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16): tok = vfm.forward_features((crop - MEAN) / STD)
    tok = tok[:, vfm.num_prefix_tokens:]
    g = CROP_RES // 16
    return F.normalize(tok.float(), dim=-1).reshape(g, g, -1)

def paste(p_crop, box, size=512):
    """p_crop: (C,g,g) class probs -> (C,size,size) with background outside crop."""
    y0, x0, s = box
    out = torch.zeros(p_crop.shape[0], size, size, device=dev); out[0] = 1
    out[:, y0:y0 + s, x0:x0 + s] = F.interpolate(p_crop[None], size=(s, s), mode="bilinear", align_corners=False)[0]
    return out

def proto_pred(f, sp, T=0.05, conf=0.7, kbg=4):
    g = f.shape[0]; fl = f.reshape(-1, f.shape[-1]); p = sp.reshape(3, -1)
    lab = p.argmax(0); c = p.max(0).values > conf
    logits = []
    for k in range(3):
        m = (lab == k) & c
        if m.sum() < 1: m = lab == k
        if m.sum() < 1: logits.append(torch.full((fl.shape[0],), -1e4, device=dev)); continue
        x = fl[m]
        if k == 0 and x.shape[0] > kbg:  # heterogeneous background: several prototypes
            cent = x[torch.randperm(x.shape[0], device=dev)[:kbg]]
            for _ in range(10):
                a = (x @ cent.T).argmax(1)
                cent = F.normalize(torch.stack([x[a == j].mean(0) if (a == j).any() else cent[j] for j in range(kbg)]), dim=-1)
            logits.append((fl @ cent.T).max(1).values / T)
        else:
            logits.append(fl @ F.normalize(x.mean(0), dim=0) / T)
    return torch.stack(logits).softmax(0).reshape(3, g, g)

def probe_pred(f, sp, steps=100, lr=0.1, wd=1e-3):
    g = f.shape[0]; fl = f.reshape(-1, f.shape[-1]); p = sp.reshape(3, -1).T
    w = (p.max(1).values - 1 / 3) / (2 / 3)
    W = torch.zeros(fl.shape[1], 3, device=dev, requires_grad=True); b = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); out = fl @ W + b
            loss = (-(p * out.log_softmax(1)).sum(1) * w).mean() + wd * (W ** 2).sum()
            loss.backward(); opt.step()
    return (fl @ W + b).softmax(1).T.reshape(3, g, g).detach()

def dsc(cls, gt):
    return binary_dice_numpy(cls >= 1, gt >= 1), binary_dice_numpy(cls == 2, gt == 2)

samples = build_domains(["B"], N)["B"]
rec = {k: [] for k in ["src", "proto", "probe", "avg_proto", "avg_probe"]}
t = time.time()
for s in samples:
    img, gt = prepare_sample(s, 512); x = image_to_tensor(img, dev); gt = gt.numpy()
    with torch.no_grad():
        sc = F.softmax(src(x), 1)
    st = structure_probabilities(sc)[0].cpu().numpy()
    box = crop_box(st[0] > 0.5, st[0])
    f = feats(x, box); g = f.shape[0]
    y0, x0, sz = box
    sp = F.adaptive_avg_pool2d(F.interpolate(sc[:, :, y0:y0 + sz, x0:x0 + sz], size=(g * 4, g * 4), mode="bilinear"), g)[0]
    pp = paste(proto_pred(f, sp), box); pr = paste(probe_pred(f, sp), box)
    rec["src"].append(dsc(sc[0].argmax(0).cpu().numpy(), gt))
    rec["proto"].append(dsc(pp.argmax(0).cpu().numpy(), gt))
    rec["probe"].append(dsc(pr.argmax(0).cpu().numpy(), gt))
    rec["avg_proto"].append(dsc(((pp + sc[0]) / 2).argmax(0).cpu().numpy(), gt))
    rec["avg_probe"].append(dsc(((pr + sc[0]) / 2).argmax(0).cpu().numpy(), gt))
print(f"{arch} crop_res={CROP_RES} scale={SCALE} n={N} {time.time() - t:.1f}s")
for k, v in rec.items():
    v = np.array(v) * 100; print(f"{k:10s} OD={v[:, 0].mean():.2f} OC={v[:, 1].mean():.2f} avg={v.mean():.2f}")
