import sys; sys.argv=[sys.argv[0], sys.argv[1], "vit_large_patch16_dinov3.lvd1689m", sys.argv[2], sys.argv[3]]
exec(open(__file__.replace("dinov3_sweep.py", "dinov3_probe.py")).read().split("samples = build_domains")[0])
from scipy import ndimage
def lcc_cls(cls):
    disc = largest_component(cls >= 1); out = np.zeros_like(cls); out[disc] = 1
    cup = largest_component((cls == 2) & disc); out[cup] = 2; return out
def em(f, sp, T, conf, it):
    p = proto_pred(f, sp, T=T, conf=conf)
    for _ in range(it): p = proto_pred(f, p, T=T, conf=conf)
    return p
cfgs = [(T, c, it) for T in (0.03, 0.05, 0.1) for c in (0.5, 0.7, 0.9) for it in (0, 2)]
samples = build_domains(["B"], N)["B"]
R = {k: [] for k in ["src", "src_lcc"] + [str(c) for c in cfgs] + [str(c)+"_lcc" for c in cfgs]}
for s in samples:
    img, gt = prepare_sample(s, 512); x = image_to_tensor(img, dev); gt = gt.numpy()
    with torch.no_grad(): sc = F.softmax(src(x), 1)
    st = structure_probabilities(sc)[0].cpu().numpy()
    box = crop_box(st[0] > 0.5, st[0]); f = feats(x, box); g = f.shape[0]; y0, x0, sz = box
    sp = F.adaptive_avg_pool2d(F.interpolate(sc[:, :, y0:y0+sz, x0:x0+sz], size=(g*4, g*4), mode="bilinear"), g)[0]
    c0 = sc[0].argmax(0).cpu().numpy(); R["src"].append(dsc(c0, gt)); R["src_lcc"].append(dsc(lcc_cls(c0), gt))
    for c in cfgs:
        torch.manual_seed(0)
        cl = paste(em(f, sp, *c), box).argmax(0).cpu().numpy()
        R[str(c)].append(dsc(cl, gt)); R[str(c)+"_lcc"].append(dsc(lcc_cls(cl), gt))
print(f"crop_res={CROP_RES} scale={SCALE} n={N}")
for k, v in sorted(R.items(), key=lambda kv: np.mean(kv[1])):
    v = np.array(v) * 100; print(f"{k:22s} OD={v[:,0].mean():.2f} OC={v[:,1].mean():.2f} avg={v.mean():.2f}")
