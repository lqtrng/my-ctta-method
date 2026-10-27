import sys; sys.path.insert(0, __file__.rsplit("/redifm_dinov3/", 1)[0])
import numpy as np, torch, torch.nn.functional as F, argparse
import redifm_dinov3.adapt as R
from common.models import ResUnet34
from common.data import build_domains
dev=torch.device("cuda")
a=argparse.Namespace(dinov3_dir=R.DEFAULT_DINOV3_DIR, dinov3_arch=R.DEFAULT_DINOV3_ARCH, crop_scale=2.0, min_crop=96, crop_resolution=512, temperature=0.05, prototype_confidence=0.9, background_prototypes=4)
seg=R.DinoV3Segmenter(a,dev)
src=ResUnet34(3); src.load_state_dict(torch.load(R.DEFAULT_SOURCE_CHECKPOINT, weights_only=True)); src.to(dev).eval()
rows=[]
for s in build_domains(["B"],400)["B"][::4]:
    img,gt=R.prepare_sample(s,512); x=R.image_to_tensor(img,dev); gt=gt.numpy()
    with torch.no_grad(): st=R.structure_probabilities(F.softmax(src(x),1))
    m,p,_=seg.supervision(img,x,st,0.5); m=m[0].cpu().numpy(); sb=st[0].cpu().numpy()>0.5
    g=[gt>=1,gt==2]
    rows.append([g[0].sum(), sb[0].sum(), m[0].sum(), g[1].sum(), sb[1].sum(), m[1].sum(),
       R.binary_dice_numpy(sb[0],g[0]), R.binary_dice_numpy(m[0],g[0]), R.binary_dice_numpy(sb[1],g[1]), R.binary_dice_numpy(m[1],g[1])])
r=np.array(rows,float)
print("median areas  GT_OD src_OD fm_OD GT_OC src_OC fm_OC:", np.median(r[:,:6],0))
print("median ratio src/GT OD %.2f fm/GT OD %.2f src/GT OC %.2f fm/GT OC %.2f"%tuple(np.median(r[:,[1,2,4,5]]/r[:,[0,0,3,3]],0)))
print("dice src OD %.3f fm OD %.3f src OC %.3f fm OC %.3f"%tuple(r[:,6:].mean(0)))
