"""Box precision and recall of the prostate detector on target domains B-F."""
import json,sys,numpy as np,torch,argparse
sys.path.insert(0,'src')
from common.paths import MODEL_DIR,PROSTATE_DINOV3_DETECTOR_CHECKPOINT
from methods.detect_crop_tta import Detector,crop_of
from training.train_dinov3_detector import boxes_of
C='.cache/loss_search/prostate/';idx=json.load(open(C+'index.json'));I=np.load(C+'images.npy',mmap_mode='r');T=np.load(C+'targets.npy',mmap_mode='r');B=np.load(C+'boxes.npy',mmap_mode='r')
det=Detector(argparse.Namespace(detector_checkpoint=PROSTATE_DINOV3_DETECTOR_CHECKPOINT,dinov3_dir=MODEL_DIR),torch.device('cuda'))
def corners(b):  # (score, cy, cx, h, w) or (cy, cx, h, w) -> y0 x0 y1 x1
    cy,cx,h,w=b[-4:];return cy-h/2,cx-w/2,cy+h/2,cx+w/2
def iou(a,b):
    a,b=corners(a),corners(b);ih=max(0,min(a[2],b[2])-max(a[0],b[0]));iw=max(0,min(a[3],b[3])-max(a[1],b[1]));i=ih*iw
    return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i+1e-9)
R={}
for i,r in enumerate(idx):
    a=np.asarray(I[i],dtype=np.float32).transpose(2,0,1);a=(a-a.min())/max(a.max()-a.min(),1e-7);x=torch.from_numpy(a.copy())[None].cuda()
    g=np.asarray(T[i])>=1;truth=boxes_of(g)
    for mode,keep in (('all',True),('best',False)):
        pred=det.boxes(x,0.3,keep);s=R.setdefault((r['domain'],mode),{k:0 for k in ('pred','truth','tp5','tp3','found5','found3','full','dino_full','images','best_score')})
        s['pred']+=len(pred);s['truth']+=len(truth);s['images']+=1
        for t,key in ((0.5,'5'),(0.3,'3')):
            s['tp'+key]+=sum(any(iou(p,q)>=t for q in truth) for p in pred);s['found'+key]+=sum(any(iou(p,q)>=t for p in pred) for q in truth)
        inside=np.zeros_like(g)
        for p in pred:
            t0,l0,sd=crop_of(p,384,2.0,96);inside[t0:t0+sd,l0:l0+sd]=True
        s['full']+=int((g&inside).sum()>=g.sum()*.999)
        t0,l0,sd=[int(v) for v in B[i]];old=np.zeros_like(g);old[t0:t0+sd,l0:l0+sd]=True;s['dino_full']+=int((g&old).sum()>=g.sum()*.999)
print('domain mode | boxes/img | precision@0.5 recall@0.5 | precision@0.3 recall@0.3 | crop fully contains prostate: detector vs DINOv3-mask crop')
tot={}
for (dom,mode),s in sorted(R.items(),key=lambda z:(z[0][1],z[0][0])):
    print(f"{dom} {mode:4} | {s['pred']/s['images']:.2f} | {100*s['tp5']/max(s['pred'],1):5.1f} {100*s['found5']/max(s['truth'],1):5.1f} | {100*s['tp3']/max(s['pred'],1):5.1f} {100*s['found3']/max(s['truth'],1):5.1f} | {100*s['full']/s['images']:5.1f} vs {100*s['dino_full']/s['images']:5.1f}")
    t=tot.setdefault(mode,{k:0 for k in s});[t.__setitem__(k,t[k]+s[k]) for k in s]
for mode,s in tot.items(): print(f"ALL {mode:4} | {s['pred']/s['images']:.2f} | {100*s['tp5']/max(s['pred'],1):5.1f} {100*s['found5']/max(s['truth'],1):5.1f} | {100*s['tp3']/max(s['pred'],1):5.1f} {100*s['found3']/max(s['truth'],1):5.1f} | {100*s['full']/s['images']:5.1f} vs {100*s['dino_full']/s['images']:5.1f}")
