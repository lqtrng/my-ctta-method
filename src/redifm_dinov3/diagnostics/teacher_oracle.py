"""Offline teacher-quality check: DSC of the selected DINOv3 masks vs GT on domain B."""
import sys, argparse, itertools, json
sys.path.insert(0, __file__.rsplit("/redifm_dinov3/", 1)[0])
import numpy as np, torch, torch.nn.functional as F
import redifm_dinov3.adapt as R
from common.models import ResUnet34
from common.data import build_domains
grid = json.loads(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
ORACLE=True
dev = torch.device("cuda")
base = dict(dinov3_dir=R.DEFAULT_DINOV3_DIR, dinov3_arch=R.DEFAULT_DINOV3_ARCH, crop_scale=2.0, min_crop=96,
            crop_resolution=512, temperature=0.05, prototype_confidence=0.9, background_prototypes=4, background_margin=0)
src = ResUnet34(3); src.load_state_dict(torch.load(R.DEFAULT_SOURCE_CHECKPOINT, weights_only=True)); src.to(dev).eval()
samples = build_domains(["B"], n)["B"]
data = []
for s in samples:
    img, gt = R.prepare_sample(s, 512); x = R.image_to_tensor(img, dev)
    with torch.no_grad(): st = torch.stack([torch.from_numpy(gt.numpy()>=1),torch.from_numpy(gt.numpy()==2)])[None].float().to(dev).clamp(1e-3,1-1e-3) if ORACLE else R.structure_probabilities(F.softmax(src(x), 1))
    data.append((img, x, st, gt.numpy()))
seg = None
keys = list(grid)
for vals in itertools.product(*grid.values()):
    cfg = dict(base, **dict(zip(keys, vals))); a = argparse.Namespace(**cfg)
    if seg is None or seg.args.dinov3_arch != a.dinov3_arch: seg = R.DinoV3Segmenter(a, dev)
    seg.args = a
    rs, rf = [], []
    for img, x, st, gt in data:
        m, p, d = seg.supervision(img, x, st, 0.5)
        m = m[0].cpu().numpy(); sb = st[0].cpu().numpy() > 0.5
        g = [gt >= 1, gt == 2]
        rf.append([R.binary_dice_numpy(m[i], g[i]) for i in range(2)])
        rs.append([R.binary_dice_numpy(sb[i], g[i]) for i in range(2)])
    rf, rs = np.array(rf) * 100, np.array(rs) * 100
    print(dict(zip(keys, vals)), f"FM OD={rf[:,0].mean():.2f} OC={rf[:,1].mean():.2f} avg={rf.mean():.2f} | src avg={rs.mean():.2f}", flush=True)
