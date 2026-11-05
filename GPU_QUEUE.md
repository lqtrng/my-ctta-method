# GPU queue (rented 16 GB GPU)

Crop-Teach on the ReDi-FM paper's protocol, all code in [crop_teach/](crop_teach/) (self-contained;
see its README). Every source domain, 3 seeds, the paper's networks (R34 for OD/OC and prostate,
Segformer-B0 for polyp), native-res crop models, paper loss, lr 2e-5, frozen batch-norm.

## 0. Machine setup and smoke test

1. Push the code to a branch, clone it on the machine.
2. Copy `crop_teach/data/` (3.0 GB) and the DINOv3 ViT-L weights (1.2 GB) to
   `crop_teach/models/timm-vit_large_patch16_dinov3.lvd1689m/`.
3. `./setup.sh`: packages, ImageNet weights, checks data, weights, GPU and bfloat16.
4. `./run_queue.sh smoke`: every training and evaluation path on a few images.

**Memory.** The DINOv3 head is the 20M one (hidden 1024). At 1024 px (OD/OC) it ran out of
memory on 8 GB at batch 8; batch 4 used 7.3 GB, so batch 8 needs an estimated 13–14 GB.
`run_queue.sh` lets only one head training run at a time (`flock`), so two lanes don't both
train heads at once.

**GPU must support bfloat16** (Ampere or newer). A V100, T4 or P100 would make DINOv3 very slow.

## 1–3. Training, main tables, long-term

Two lanes in parallel, each `run_queue.sh <problem>`:
- lane 1: `odoc`
- lane 2: `prostate`, then `polyp`

| Phase | What | Runs |
|---|---|---|
| 1 | Train target, native-res crop model and DINOv3 head, for each of the 15 source domains | 45 trainings |
| 2 | Source-only (1 run) and Crop-Teach (seeds 0, 1, 2) for each source domain: Tables 1, 2 | 15 + 45 |
| 3 | Crop-Teach, 3 rounds, seeds 0, 1, 2, for each OD/OC and prostate source domain: Table 3 | 33 |

`./run_queue.sh tables` writes `runs/results/tables.md`, with the paper's numbers alongside.

## Time estimate

These are from per-image costs measured on the local RTX 5060 (8 GB). Training times for
Segformer-B0, the OD/OC native-res crop models and the 1024 px OD/OC heads are extrapolated.

| Problem | Phase 1 (train) | Phase 2 | Phase 3 | Total |
|---|---|---|---|---|
| OD/OC | 12.9 h | 2.0 h | 5.7 h | 20.6 h |
| Prostate | 6.0 h | 1.2 h | 3.5 h | 10.8 h |
| Polyp | 11.1 h | 1.0 h | – | 12.1 h |
| **All** | **30.0 h** | **4.2 h** | **9.2 h** | **43.4 h** |

Each 16 GB GPU below is scaled by a rough speed factor relative to the RTX 5060. The two-lane
column assumes about 1.3× from overlapping data loading, since DINOv3 already keeps the GPU busy.

| 16 GB GPU | Speed vs RTX 5060 | One lane | Two lanes |
|---|---|---|---|
| RTX 4060 Ti 16 GB, RTX A4000 | ~1.0× | ~43 h | ~33 h |
| RTX 5060 Ti 16 GB | ~1.2× | ~36 h | ~28 h |
| RTX 4070 Ti Super | ~2× | ~22 h | ~17 h |
| RTX 4080 (Super), RTX 5070 Ti | ~2.3× | ~19 h | ~15 h |
| RTX 5080 | ~2.7× | ~16 h | ~12 h |

The smoke test (phase 0) measures the real speed; re-estimate from it before starting phases 1–3.
