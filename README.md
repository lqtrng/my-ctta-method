# ReDi-FM: continual test-time adaptation for medical segmentation

## Results

### 1. Crop-Teach

| ID | Crop Model | Refine Model | Target Model | LR | Order | Seed | OD/OC | Prostate | Polyp |
|---|---|---|---|---|---|---|---|---|---|
| CT1 | DINOv3, all regions | R34, native-res | R34 | 2e-5 | default | 0 | **82.66** | – | – |
| CT2 | DINOv3, all regions | nnU, native-res | nnU | 2e-5 | default | 0 | – | – | **80.42** |
| CT3 | DINOv3, all regions | R34, low-res | R34 | 2e-5 | default | 0 | 82.19 | – | 72.04 |
| CT4 | DINOv3, all regions | nnU, low-res | nnU | 2e-5 | default | 0 | – | 77.22 | – |
| CT5 | DINOv3, largest region | R34, low-res | R34 | 2e-5 | default | 0, 1, 2 | 81.38 ± 0.08 | – | – |
| CT6 | DINOv3, largest region | nnU, low-res | nnU | 2e-5 | default | 0, 1, 2 | – | 78.85 ± 0.33 | – |
| CT7 | DINOv3, largest region | R34, low-res | R34 | 1e-4 | default | 0 | 77.09 | – | – |
| CT8 | DINOv3, largest region | nnU, low-res | nnU | 1e-4 | default | 0 | – | 77.01 | – |
| CT9 | DINOv3, largest region | R34, low-res | R34 | 2e-5 | D→B→C→E | 0 | 81.51 | – | – |
| CT11 | DINOv3, largest region | nnU, low-res | nnU | 2e-5 | default | 0 | – | **79.05** | – |
| CT12 | DINOv3, largest region | R34, low-res | R34 | 2e-5 | default | 0 | – | – | 72.00 |
| CT13 | DINOv3, largest region | nnU whole-image model (not crop-trained) | DINOv3 full-image head | 1e-4 | default | 0 | – | 62.13 | – |

### 2. ReDi-FM

| ID | Source / Target Model | Foundation Model | Prompt | LR | Order | Seed | OD/OC | Prostate | Polyp |
|---|---|---|---|---|---|---|---|---|---|
| RF0 | Paper: R34 (Segformer-B0 for polyp) | SAM ViT-H | points from source mask | – | default | 3 seeds | 73.11 | 66.10 | **77.47** |
| RF1 | R34 | SAM ViT-H | points from source mask | not recorded | default | 42 | 71.73 | – | – |
| RF2 | R34 | DINOv3 + crop head | 2× crop around source mask | 1e-4 | default | 0, 1, 2 (OD/OC); 0 (prostate) | **77.54 ± 0.10** | 54.92 | – |
| RF3 | nnU | DINOv3 + crop head | 2× crop around source mask | 1e-4 | default | 0 | – | **72.88** | – |
| RF5 | DINOv3 + full-image head | DINOv3 + crop head | 2× crop around source mask | 1e-4 | default | 0 | 75.53 | 71.41 | – |

### 3. Crop-Infer (no adaptation)

| ID | Crop Model | Refine Model | OD/OC | Prostate | Polyp |
|---|---|---|---|---|---|
| CI1 | DINOv3, all regions | R34, native-res | **66.01** | – | – |
| CI2 | DINOv3, all regions | R34, low-res | 64.87 | – | **81.79** |
| CI3 | DINOv3, largest region | R34, low-res | 57.72 | – | 81.50 |
| CI4 | DINOv3, all regions | R34 trained low-res, tested native-res | 35.10 (domain D only) | – | – |
| CI5 | DINOv3, largest region | R34 trained low-res, tested native-res | 43.28 (domain D only) | – | – |

### 4. Crop-Teach on the paper's protocol: prostate

[crop_teach/](crop_teach/): each domain in turn is the source, the other five are the stream (A→B→…), 3 seeds. SAM ViT-H full-image head (1.25M) places the crops, R34 native-res crop model, R34 target, paper loss, lr 2e-5, frozen BN. Do. X = mean DSC over the five target domains.

**Table 1: Do. X DSC (Crop-Teach: mean ± std over seeds 0, 1, 2)**

| Method | Do. A | Do. B | Do. C | Do. D | Do. E | Do. F | Average |
|---|---|---|---|---|---|---|---|
| Source only (paper) | 51.02 | 58.79 | 26.88 | 24.15 | 60.42 | 53.33 | 45.77 |
| ReDi-FM (paper) | 66.10 | 66.44 | 49.89 | 41.24 | 73.50 | 73.69 | 61.81 |
| Source only (ours) | 65.17 | 67.88 | 63.96 | 35.54 | 71.96 | 72.90 | 62.90 |
| **Crop-Teach** | **76.12 ± 0.10** | **78.54 ± 0.56** | **74.42 ± 0.27** | 39.24 ± 21.31 ⚠ | **77.93 ± 0.37** | **77.04 ± 0.14** | **70.55 ± 3.64** |
| Crop-Teach, collapsed run excluded | 76.12 | 78.54 | 74.42 | 54.29 (seeds 0, 2) | 77.93 | 77.04 | 73.06 |

Our source model is much stronger than the paper's (62.90 vs 45.77, from nnU-Net intensity augmentation), so the gain over source-only is smaller: +7.65 (paper +16.04).

**⚠ Collapse: source D, seed 1 (Do. D 9.13; seeds 0 and 2: 55.65, 52.93).** In domain B, images 113–118 had near-empty teachers (on image 114 both the head and the crop teacher predicted no prostate, so the gate r = 1 and both hard terms pushed the target to all-background at full weight). From B image 117 the target predicts empty masks on every remaining image (C, E, F: 0.00), even when the crop teacher is good again (DSC 80–93). It never recovers because the hard loss clamps the target probability to [1e-6, 1 − 1e-6], which kills the gradient once the target saturates. The clamp is the same as in our ReDi-FM reproduction (`src/common/redifm.py`). Not fixed yet; candidate fixes: hard loss from logits, and skip the update when both teachers are empty. No collapse in the other 17 runs or in the long-term runs.

**Teachers without adaptation (seed 0, Do. X)**

| Source | Head alone | Crop teacher (head boxes) | Crop model on oracle boxes (ground truth) | Crop-Teach |
|---|---|---|---|---|
| A | 59.23 | 64.46 | 80.96 | 76.10 |
| B | 65.06 | 68.18 | 80.43 | 79.29 |
| C | 52.09 | 65.83 | 81.70 | 74.66 |
| D | 35.16 | 43.72 | 52.81 | 55.65 |
| E | 62.10 | 73.64 | 83.35 | 78.26 |
| F | 52.65 | 69.50 | 83.37 | 76.91 |
| Average | 54.38 | 64.22 | 77.10 | 73.48 |

Oracle boxes are a diagnostic only (they read the labels). Box placement costs 12.9 points; the paper reports the same kind of error for box prompts (Figure 3).

**Table 3: long-term, 3 rounds**

| Method | Round 1 | Round 2 | Round 3 | Average | Gain | Degradation |
|---|---|---|---|---|---|---|
| ReDi-FM (paper, all sources) | 61.81 | 64.85 | 65.33 | 64.00 | +18.23 | −2.19 |
| Crop-Teach (source C only, seeds 0, 1, 2) | 74.60 | 76.94 | 76.87 | 76.14 | +12.17 | −1.54 |

### OD/OC — targets B REFUGE · C ORIGA · D REFUGE val/test · E Drishti-GS

**Adaptation**

| # | Method | B | C | D | E | **Do. A** |
|---|---|---|---|---|---|---|
| O1 | Crop-teach, blurry, all | 83.13 | 85.03 | 77.87 | 82.71 | 82.19 |
| O2 | Crop-teach, blurry, largest (3 seeds) | 80.70 | 85.33 | 76.75 | 82.73 | 81.38 ± 0.08 |
| O3 | Crop-teach, blurry, largest, lr 1e-4 | 76.92 | 80.86 | 73.64 | 76.96 | 77.09 |
| O4 | Crop-teach, blurry, largest, stream D→B→C→E | 81.64 | 85.79 | 76.04 | 82.59 | 81.51 |
| O5 | Crop-teach, sharp, all | **83.16** | 85.56 | 79.17 | 82.74 | **82.66** |
| O6 | ReDi-FM + D3, nnU\* | 82.52 | 80.71 | **82.67** | 82.50 | 82.10 |
| O7 | ReDi-FM + D3, R34 (3 seeds) | 78.88 | 77.61 | 79.22 | 74.49 | 77.54 ± 0.10 |
| O8 | ReDi-FM + SAM (our run) | 72.46 | 69.94 | 68.76 | 75.75 | 71.73 |
| O9 | ReDi-FM, D3 only | 57.28 | **86.66** | 71.32 | **86.86** | 75.53 |

**No adaptation**

| # | Method | B | C | D | E | **Do. A** |
|---|---|---|---|---|---|---|
| O10 | D3 dual (Eq. 16 fusion) | 41.52 | **85.10** | 61.75 | 84.01 | 68.09 |
| O11 | D3 single | 51.17 | 82.80 | 59.83 | **84.62** | **69.61** |
| O12 | `dinov3_crop`, blurry, all | 79.01 | 67.06 | 35.17 | 78.22 | 64.87 |
| O13 | `dinov3_crop`, blurry, largest | 40.52 | 67.69 | 44.40 | 78.25 | 57.72 |
| O16 | `dinov3_crop`, sharp, all | 72.69 | 69.65 | 45.12 | 76.60 | 66.01 |
| O17 | Source only, nnU\* | **75.80** | 60.20 | **67.43** | 74.78 | 69.55 |
| O18 | Source only, R34 | 54.54 | 47.62 | 39.31 | 73.20 | 53.67 |

**OD and OC per domain**

| # | B OD | B OC | C OD | C OC | D OD | D OC | E OD | E OC | **Mean OD** | **Mean OC** |
|---|---|---|---|---|---|---|---|---|---|---|
| O1 | 85.63 | 80.63 | 90.65 | 79.41 | 77.63 | 78.11 | 89.34 | 76.08 | 85.81 | 78.56 |
| O2 | 82.34 | 79.06 | 90.88 | 79.78 | 76.53 | 76.97 | 89.23 | 76.24 | 84.75 | 78.01 |
| O3 | 75.88 | 77.96 | 86.68 | 75.03 | 72.82 | 74.45 | 83.90 | 70.01 | 79.82 | 74.36 |
| O4 | 82.89 | 80.38 | 91.23 | 80.35 | 76.61 | 75.47 | 89.78 | 75.39 | 85.13 | 77.90 |
| O5 | 85.78 | 80.55 | 90.69 | 80.43 | 78.24 | 80.09 | 89.20 | 76.28 | 85.98 | 79.34 |
| O6 | 87.55 | 77.49 | 90.69 | 70.73 | 86.28 | 79.07 | 90.88 | 74.12 | 88.85 | 75.35 |
| O7 | 81.99 | 75.76 | 85.02 | 70.20 | 80.14 | 78.29 | 83.46 | 65.53 | 82.65 | 72.45 |
| O8 | 84.52 | 60.41 | 84.27 | 55.62 | 81.43 | 56.08 | 83.99 | 67.51 | 83.55 | 59.91 |
| O9 | 61.32 | 53.24 | 92.79 | 80.54 | 68.49 | 74.15 | 90.16 | 83.55 | 78.19 | 72.87 |
| O10 | 42.51 | 40.52 | 92.16 | 78.03 | 62.31 | 61.18 | 91.17 | 76.84 | 72.04 | 64.14 |
| O11 | 59.89 | 42.45 | 90.75 | 74.85 | 60.99 | 58.67 | 89.90 | 79.34 | 75.38 | 63.83 |
| O12 | 83.41 | 74.61 | 74.93 | 59.19 | 36.72 | 33.62 | 88.78 | 67.67 | 70.96 | 58.77 |
| O13 | 42.29 | 38.76 | 75.50 | 59.88 | 42.60 | 46.20 | 88.83 | 67.68 | 62.30 | 53.13 |
| O16 | 80.11 | 65.26 | 75.83 | 63.46 | 43.47 | 46.77 | 88.97 | 64.22 | 72.09 | 59.93 |
| O17 | 77.68 | 73.93 | 75.00 | 45.40 | 65.43 | 69.44 | 84.33 | 65.24 | 75.61 | 63.50 |
| O18 | 62.39 | 46.69 | 56.63 | 38.60 | 38.60 | 40.03 | 82.49 | 63.92 | 60.03 | 47.31 |

**Where DINOv3 + full-image head places crops**

| Domain | Regions / image | Some region touches the disc | A crop holds the whole disc | Region box IoU ≥ 0.5 with the disc | Regions that are on the disc | Images with a false region |
|---|---|---|---|---|---|---|
| B | 2.04 | 100% | 100% | 99.2% | 49.1% | 92.0% |
| C | 1.11 | 100% | 100% | 99.8% | 90.2% | 8.2% |
| D | 2.00 | 100% | 99.8% | 76.1% | 50.0% | 80.4% |
| E | 1.01 | 100% | 100% | 100% | 99.0% | 1.0% |
| All | 1.66 | 100% | 99.9% | 90.0% | 60.2% | 54.6% |

**Domain-B-only development runs: ReDi-FM + D3, R34**

| Run | Setting | B (OD / OC) |
|---|---|---|
| [exp01_default](results/redifm_dinov3/exp01_default.json) | training-free prototype teacher, seed 42 | 69.18 (76.68 / 61.69) |
| [exp02_head](results/redifm_dinov3/exp02_head.json) | D3-crop head teacher, seed 42 | 78.50 (81.94 / 75.05) |
| [exp03 seed 0](results/redifm_dinov3/exp03_head_seed0.json) · [1](results/redifm_dinov3/exp03_head_seed1.json) · [2](results/redifm_dinov3/exp03_head_seed2.json) | head, seeds 0, 1, 2 | 78.45 · 78.91 · 79.27 |
| [exp04](results/redifm_dinov3/exp04_head_l11-17-23.json) | head on blocks 11, 17 and 23, seed 42 | 77.94 (81.51 / 74.37) |
| [final B seed 0](results/redifm_dinov3/final_domain_B_seed0.json) · [1](results/redifm_dinov3/final_domain_B_seed1.json) · [2](results/redifm_dinov3/final_domain_B_seed2.json) | the final setting, B only | 78.45 · 78.91 · 79.27 |

### Prostate — targets B BMC · C HK · D I2CVB · E RUNMC · F UCL

**Adaptation**

| # | Method | B | C | D | E | F | **Do. A** |
|---|---|---|---|---|---|---|---|
| P1 | Crop-teach, blurry, all | 84.26 | 84.14 | 66.61 | 75.95 | 75.14 | 77.22 |
| P2 | Crop-teach, blurry, largest (3 seeds) | 84.03 | 84.34 | 68.82 | **79.11** | **77.94** | 78.85 ± 0.33 |
| P3 | Crop-teach, blurry, largest, lr 1e-4 | 82.65 | 82.38 | 67.50 | 75.08 | 77.45 | 77.01 |
| P4 | Crop-teach, first version | 74.18 | 76.07 | 61.92 | 55.39 | 43.08 | 62.13 |
| P5 | Crop-teach, blurry, largest (seed 0, same as P2) | 84.21 | **84.52** | **70.72** | 78.99 | 76.80 | **79.05** |
| P6 | ReDi-FM + D3, nnU | 82.47 | 81.56 | 57.25 | 71.92 | 71.18 | 72.88 |
| P7 | ReDi-FM + D3, nnU\* | 70.08 | 72.06 | 62.64 | 64.47 | 63.69 | 66.59 |
| P8 | ReDi-FM + D3, R34 | 52.21 | 77.61 | 43.02 | 64.79 | 36.96 | 54.92 |
| P9 | ReDi-FM, D3 only | **84.60** | 83.61 | 63.92 | 70.67 | 54.24 | 71.41 |

**No adaptation**

| # | Method | B | C | D | E | F | **Do. A** |
|---|---|---|---|---|---|---|---|
| P10 | D3 single | **82.34** | **79.98** | **60.63** | 64.84 | 47.70 | **67.10** |
| P11 | Source only, nnU | 77.99 | 64.45 | 32.18 | **68.35** | **62.23** | 61.04 |
| P12 | Source only, nnU\* | 58.05 | 66.94 | 23.15 | 51.86 | 39.28 | 47.86 |
| P13 | Source only, R34 | 19.07 | 72.54 | 9.60 | 23.47 | 7.75 | 26.49 |

**DINOv3 detector head as crop placer: box accuracy only, no DSC**

| Domain | Boxes / image | P / R @ IoU 0.5 | P / R @ IoU 0.3 | Crop holds the whole prostate: detector vs D3-full mask |
|---|---|---|---|---|
| B | 1.02 | 86.5 / 86.7 | 96.7 / 96.6 | 99.7% vs 96.6% |
| C | 1.00 | 83.5 / 83.5 | 93.7 / 93.7 | 98.1% vs 100% |
| D | 1.04 | 63.9 / 64.7 | 79.8 / 80.1 | 83.1% vs 86.8% |
| E | 1.01 | 71.5 / 71.5 | 80.0 / 79.8 | 87.2% vs 87.6% |
| F | 1.01 | 54.0 / 54.3 | 67.6 / 68.0 | 81.1% vs 74.9% |
| All | 1.02 | 72.1 / 72.5 | 83.9 / 84.0 | 89.4% vs 89.4% |

### Polyp — targets B CVC-ClinicDB · C ETIS-Larib · D Kvasir-SEG (domain A: BKAI-IGH NeoPolyp)

**Adaptation**

| # | Method | B | C | D | **Do. A** |
|---|---|---|---|---|---|
| Y1 | Crop-teach, blurry, all, R34 | 77.65 | 57.63 | 80.85 | 72.04 |
| Y2 | Crop-teach, sharp, all, nnU | **82.22** | **73.96** | **85.07** | **80.42** |
| Y3 | Crop-teach, blurry, largest, R34 | 76.69 | 58.26 | 81.04 | 72.00 |

**No adaptation**

| # | Method | B | C | D | **Do. A** |
|---|---|---|---|---|---|
| Y4 | `dinov3_crop`, blurry, all | **80.94** | **79.89** | **84.55** | **81.79** |
| Y5 | `dinov3_crop`, blurry, largest | 80.68 | 79.82 | 84.02 | 81.50 |
| Y6 | Source only, nnU | 73.90 | 73.18 | 81.14 | 76.07 |
| Y7 | Source only, R34 | 72.12 | 60.99 | 80.08 | 71.06 |

### Loss search (dev / held-out split)

**All loss variants**

| Loss | prostate dev | OD/OC dev | mean dev | prostate held-out | OD/OC held-out | prostate all | OD/OC all |
|---|---|---|---|---|---|---|---|
| **pg_distill_bn_lr2e5** | 79.39 | 81.94 | **80.67** | 78.93 | 81.43 | 79.01 | 81.69 |
| agree_lr2e5 | 79.47 | 81.36 | 80.41 | 79.13 | 81.22 | 79.10 | 81.29 |
| agree_lr4e5 | 79.17 | 81.49 | 80.33 | 78.91 | 81.53 | 78.86 | 81.51 |
| pg_bn_lr2e5 | 79.23 | 81.90 | 80.56 | 78.29 | 81.43 | 78.70 | 81.67 |
| pgd_double_bn_lr2e5 | 79.39 | 81.64 | 80.52 | 78.47 | 81.31 | 78.86 | 81.48 |
| pgd_confident_bn_lr2e5 | 79.45 | 81.57 | 80.51 | 79.15 | 81.32 | 79.17 | 81.45 |
| pg_half_bn_lr2e5 | 79.38 | 81.51 | 80.44 | 78.82 | 80.90 | 78.92 | 81.21 |
| pgd_half_bn_lr2e5 | 79.09 | 81.75 | 80.42 | 78.72 | 81.34 | 78.69 | 81.55 |
| pgs_bn_lr1e5 | 79.68 | 81.07 | 80.37 | 78.50 | 80.71 | 79.02 | 80.89 |
| pgs_bn_lr2e5 | 79.32 | 81.29 | 80.31 | 78.99 | 81.06 | 79.01 | 81.18 |
| pgs_agree_bn_lr2e5 | 79.07 | 80.67 | 79.87 | 79.71 | 80.09 | 79.18 | 80.39 |
| fused_bn_lr2e5 | 78.96 | 80.50 | 79.73 | 79.17 | 80.13 | 78.88 | 80.33 |
| pgs_bn_lr5e6 | 78.79 | 80.50 | 79.64 | 77.84 | 80.10 | 78.30 | 80.30 |
| pgs_bn_lr5e5 | 78.06 | 81.11 | 79.58 | 78.77 | 80.80 | 78.18 | 80.96 |
| fused_bn_lr5e5 | 77.74 | 80.71 | 79.23 | 78.51 | 80.26 | 77.91 | 80.49 |
| baseline | 76.56 | 78.16 | 77.36 | 77.25 | 77.83 | 76.88 | 77.99 |
| pixel_fused | 76.91 | 76.93 | 76.92 | 77.33 | 77.22 | 77.05 | 77.07 |
| pixel_agree_both | 77.11 | 76.40 | 76.76 | 76.23 | 76.12 | 76.70 | 76.26 |
| crop_hard | 75.39 | 75.95 | 75.67 | 75.65 | 75.97 | 75.45 | 75.95 |
| fused_soft | 75.21 | 75.74 | 75.47 | 77.32 | 75.17 | 76.10 | 75.45 |
| crop_soft | 75.77 | 73.41 | 74.59 | 77.15 | 73.60 | 76.23 | 73.49 |
| distill_only_bn_lr2e5 | 74.42 | 63.00 | 68.71 | 73.40 | 63.24 | 73.92 | 63.12 |
| pgs_bn | – | 80.52 | – | – | 79.94 | – | 80.24 |
| per_gate_soft | – | 78.80 | – | – | 78.49 | – | 78.64 |
| baseline_frozen_bn | – | 78.79 | – | – | 78.76 | – | 78.77 |
| fused_lr5e5 | 77.74 | – | – | 78.51 | – | 77.91 | – |
| fused_agree | 77.03 | – | – | 76.50 | – | 76.76 | – |
| agree_both_half | 76.53 | – | – | 76.53 | – | 76.63 | – |
| fused_distill | 76.51 | – | – | 76.73 | – | 76.56 | – |
| agree_both_gate | 76.11 | – | – | 76.68 | – | 76.35 | – |
| fused_nogate | 75.90 | – | – | 77.02 | – | 76.52 | – |
| pixel_agree | 75.89 | – | – | 74.74 | – | 75.35 | – |
| crop_distill | 75.70 | – | – | 76.32 | – | 75.80 | – |
| agree_both_zero | 75.35 | – | – | 76.92 | – | 76.04 | – |
| image_gate | 75.18 | – | – | 76.30 | – | 75.64 | – |
| crop_hard_plain | 74.64 | – | – | 75.50 | – | 75.08 | – |
| agree_both_soft | 73.34 | – | – | 75.98 | – | 74.30 | – |
| fused_lr3e4 | 72.06 | – | – | 73.10 | – | 72.40 | – |

| Loss | B | C | D | all | dev | held-out |
|---|---|---|---|---|---|---|
| agree_lr2e5 | 78.60 | 55.55 | 80.98 | 71.71 | 71.06 | 72.35 |
| pg_distill_bn_lr2e5 | 77.35 | 54.15 | 80.70 | 70.73 | 70.93 | 70.53 |
| agree_lr4e5 | 75.28 | 54.49 | 80.03 | 69.93 | 69.65 | 70.21 |
