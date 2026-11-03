# ReDi-FM: continual test-time adaptation for medical segmentation

Every result computed in this repository, then how to reproduce it. Three problems: OD/OC (fundus), prostate (MRI), and polyp (colonoscopy). Each trains on labeled domain A, then adapts online, one unlabeled image at a time, to the target domains in order, never resetting. The protocol is from `0140.pdf`.

## Results

**How to read the tables**
- **Score:** per-image DSC, averaged per domain. **Do. A** is the unweighted mean over the target domains.
- **Row IDs:** O = OD/OC, P = prostate, Y = polyp. Each row's ID links its score, its OD/OC breakdown and its setup.
- **Tables per problem:** scores (adaptation, no adaptation, checks), then OD/OC per structure (OD includes the cup), then *Setup and source*. *Setup and source* gives the pipeline and links the result JSON, whose `config` holds every option and checkpoint.
- **3 seeds:** mean ± std over seeds 0, 1 and 2. The per-domain cells are seed means, and each seed's Do. A is linked under *Setup and source*.
- **† rows:** analysis scripts, not `src/evaluate.py`. The script is linked.
- **Bold:** the best value in the column of that table.

**Short names**

| Name | Meaning |
|---|---|
| **D3** | frozen DINOv3 ViT-L/16 |
| **D3-full** | D3 + head trained on whole domain-A images |
| **D3-crop** | D3 + head trained on crops |
| **R34** | ResUNet-34 |
| **nnU** | nnU-Net 2D with official training config |
| **nnU\*** | nnU-Net 2D with wrong training config |
| **crop-CNN** | a CNN trained on enlarged domain-A crops |
| **blurry** | crop cut from the resized image |
| **sharp** | the same crop cut from the full-size image |
| **largest / all** | crop the largest predicted region, or every region |
| **Crop-teach** | D3-full places the crops, a frozen crop-CNN on them is the teacher, a CNN target adapts (`redifm_dinov3_nnunet_crop`) |

### OD/OC — targets B REFUGE · C ORIGA · D REFUGE val/test · E Drishti-GS

Paper (ReDi-FM + SAM): **73.11**.

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
| O14 | `dinov3_crop`, blurry-trained crop-CNN given sharp test crops, all (D only) | | | 35.10 | | |
| O15 | `dinov3_crop`, blurry-trained crop-CNN given sharp test crops, largest (D only) | | | 43.28 | | |
| O16 | `dinov3_crop`, sharp, all | 72.69 | 69.65 | 45.12 | 76.60 | 66.01 |
| O17 | Source only, nnU\* | **75.80** | 60.20 | **67.43** | 74.78 | 69.55 |
| O18 | Source only, R34 | 54.54 | 47.62 | 39.31 | 73.20 | 53.67 |

**Checks: the crop-CNN alone, crops placed from the ground truth (2× the disc)**

| # | Method | B | C | D | E | **Mean** |
|---|---|---|---|---|---|---|
| O19 | † Blurry R34 crop-CNN | **81.26** | 65.28 | 60.58 | **76.35** | 70.87 |
| O20 | † Sharp R34 crop-CNN | 80.73 | **68.39** | **68.39** | 74.74 | **73.06** |

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
| O14 |  |  |  |  | 35.96 | 34.24 |  |  | – | – |
| O15 |  |  |  |  | 40.90 | 45.66 |  |  | – | – |
| O16 | 80.11 | 65.26 | 75.83 | 63.46 | 43.47 | 46.77 | 88.97 | 64.22 | 72.09 | 59.93 |
| O17 | 77.68 | 73.93 | 75.00 | 45.40 | 65.43 | 69.44 | 84.33 | 65.24 | 75.61 | 63.50 |
| O18 | 62.39 | 46.69 | 56.63 | 38.60 | 38.60 | 40.03 | 82.49 | 63.92 | 60.03 | 47.31 |
| O19 | 85.08 | 77.43 | 73.77 | 56.78 | 60.54 | 60.61 | 88.60 | 64.10 | 77.00 | 64.73 |
| O20 | 85.34 | 76.11 | 74.69 | 62.08 | 68.17 | 68.60 | 88.70 | 60.77 | 79.22 | 66.89 |

**Setup and source**

| # | Places crop → predicts crop → target | Run |
|---|---|---|
| O1 | D3-full → R34 crop-CNN → R34 | [seed 0](results/redifm_dinov3_nnunet_crop/odoc_all_regions_seed0.json) |
| O2 | D3-full → R34 crop-CNN → R34 | [81.45](results/redifm_dinov3_nnunet_crop/odoc_final_seed0.json) · [81.27](results/redifm_dinov3_nnunet_crop/odoc_final_seed1.json) · [81.41](results/redifm_dinov3_nnunet_crop/odoc_final_seed2.json) |
| O3 | D3-full → R34 crop-CNN → R34 | [seed 0](results/redifm_dinov3_nnunet_crop/odoc_resunet34_seed0.json) |
| O4 | D3-full → R34 crop-CNN → R34 | [seed 0](results/redifm_dinov3_nnunet_crop/odoc_order_DBCE_seed0.json) |
| O5 | D3-full → R34 crop-CNN (sharp) → R34 | [seed 0](results/sharp_crop/adapt_all_seed0.json) |
| O6 | nnU\* source → D3-crop → nnU\* | [seed 0](results/redifm_dinov3_nnunet/all_domains_seed0.json) |
| O7 | R34 source → D3-crop → R34 | [77.57](results/redifm_dinov3/final_all_domains_seed0.json) · [77.42](results/redifm_dinov3/final_all_domains_seed1.json) · [77.65](results/redifm_dinov3/final_all_domains_seed2.json) |
| O8 | R34 source → SAM ViT-H → R34 | [seed 42](results/redifm_sam/redifm_sam_domain_A.json) |
| O9 | D3-full source → D3-crop → D3-full head | [seed 0](results/redifm_dinov3_dual/all_domains_seed0.json) |
| O10 | D3-full + D3-crop | [json](results/dinov3_dual/dinov3_dual_domain_A.json) |
| O11 | D3-full | [json](results/dinov3_single/dinov3_single_domain_A.json) |
| O12 | D3-full → R34 crop-CNN | [json](results/dinov3_crop/all_regions_odoc.json) |
| O13 | D3-full → R34 crop-CNN | [json](results/dinov3_crop/dinov3_crop_domain_A.json) |
| O14 | D3-full → R34 crop-CNN, sharp pixels at test only | [json](results/dinov3_crop/original_pixels_D_all.json) |
| O15 | D3-full → R34 crop-CNN, sharp pixels at test only | [json](results/dinov3_crop/original_pixels_D_largest.json) |
| O16 | D3-full → R34 crop-CNN (sharp) | [json](results/sharp_crop/no_adapt_all.json) |
| O17 | nnU\* | [json](results/source_only_nnunet/source_only_nnunet_domain_A.json) |
| O18 | R34 | [json](results/source_only/source_only_domain_A.json) |
| O19 | ground truth → R34 crop-CNN | [script](results/sharp_crop/crop_model_only.py) |
| O20 | ground truth → R34 crop-CNN (sharp) | [script](results/sharp_crop/crop_model_only.py) |

**Long-term, 3 passes of B→E with no reset** (O7, seed 0, [json](results/redifm_dinov3/long_term_seed0.json)): the rounds give 77.57 / 75.97 / 77.39, an average of 76.98. That's +23.31 over R34 source only, with 0.59 degradation.

<details><summary>Where DINOv3 + full-image head places crops (†, <a href="results/dinov3_crop/region_accuracy.py">script</a>)</summary>

| Domain | Regions / image | Some region touches the disc | A crop holds the whole disc | Region box IoU ≥ 0.5 with the disc | Regions that are on the disc | Images with a false region |
|---|---|---|---|---|---|---|
| B | 2.04 | 100% | 100% | 99.2% | 49.1% | 92.0% |
| C | 1.11 | 100% | 100% | 99.8% | 90.2% | 8.2% |
| D | 2.00 | 100% | 99.8% | 76.1% | 50.0% | 80.4% |
| E | 1.01 | 100% | 100% | 100% | 99.0% | 1.0% |
| All | 1.66 | 100% | 99.9% | 90.0% | 60.2% | 54.6% |

High recall, low precision: no image loses its disc, but most B and D images get one extra false region.
</details>

<details><summary>Domain-B-only development runs: ReDi-FM + D3, R34 (<code>results/redifm_dinov3/</code>)</summary>

| Run | Setting | B (OD / OC) |
|---|---|---|
| [exp01_default](results/redifm_dinov3/exp01_default.json) | training-free prototype teacher, seed 42 | 69.18 (76.68 / 61.69) |
| [exp02_head](results/redifm_dinov3/exp02_head.json) | D3-crop head teacher, seed 42 | 78.50 (81.94 / 75.05) |
| [exp03 seed 0](results/redifm_dinov3/exp03_head_seed0.json) · [1](results/redifm_dinov3/exp03_head_seed1.json) · [2](results/redifm_dinov3/exp03_head_seed2.json) | head, seeds 0, 1, 2 | 78.45 · 78.91 · 79.27 |
| [exp04](results/redifm_dinov3/exp04_head_l11-17-23.json) | head on blocks 11, 17 and 23, seed 42 | 77.94 (81.51 / 74.37) |
| [final B seed 0](results/redifm_dinov3/final_domain_B_seed0.json) · [1](results/redifm_dinov3/final_domain_B_seed1.json) · [2](results/redifm_dinov3/final_domain_B_seed2.json) | the final setting, B only | 78.45 · 78.91 · 79.27 |
</details>

### Prostate — targets B BMC · C HK · D I2CVB · E RUNMC · F UCL

Paper: source only **51.02**, ReDi-FM + SAM **66.10**.

**Adaptation**

| # | Method | B | C | D | E | F | **Do. A** |
|---|---|---|---|---|---|---|---|
| P1 | Crop-teach, blurry, all | 84.26 | 84.14 | 66.61 | 75.95 | 75.14 | 77.22 |
| P2 | Crop-teach, blurry, largest (3 seeds) | 84.03 | 84.34 | 68.82 | **79.11** | **77.94** | 78.85 ± 0.33 |
| P3 | Crop-teach, blurry, largest, lr 1e-4 | 82.65 | 82.38 | 67.50 | 75.08 | 77.45 | 77.01 |
| P4 | Crop-teach, first version | 74.18 | 76.07 | 61.92 | 55.39 | 43.08 | 62.13 |
| P5 | † `crop_teach_tta.py` (standalone Crop-teach, largest) | 84.21 | **84.52** | **70.72** | 78.99 | 76.80 | **79.05** |
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

**Setup and source**

| # | Places crop → predicts crop → target | Run |
|---|---|---|
| P1 | D3-full → nnU crop-CNN → nnU | [seed 0](results/redifm_dinov3_nnunet_crop/prostate_all_regions_seed0.json) |
| P2 | D3-full → nnU crop-CNN → nnU | [79.13](results/redifm_dinov3_nnunet_crop/prostate_final_seed0.json) · [79.02](results/redifm_dinov3_nnunet_crop/prostate_final_seed1.json) · [78.39](results/redifm_dinov3_nnunet_crop/prostate_final_seed2.json) |
| P3 | D3-full → nnU crop-CNN → nnU | [seed 0](results/redifm_dinov3_nnunet_crop/prostate_crop_trained_seed0.json) |
| P4 | D3-full → nnU (whole-image, not crop-trained) → D3-full head; lr 1e-4 | [seed 0](results/redifm_dinov3_nnunet_crop/prostate_all_domains_seed0.json) |
| P5 | D3-full → nnU crop-CNN → nnU | [scores](results/crop_teach_tta/prostate_check/scores.json) |
| P6 | nnU source → D3-crop → nnU | [seed 0](results/redifm_dinov3_nnunet_official/prostate_all_domains_seed0.json) |
| P7 | nnU\* source → D3-crop → nnU\* | [seed 0](results/redifm_dinov3_nnunet/prostate_all_domains_seed0.json) |
| P8 | R34 source → D3-crop → R34 | [seed 0](results/redifm_dinov3/prostate_all_domains_seed0.json) |
| P9 | D3-full source → D3-crop → D3-full head | [seed 0](results/redifm_dinov3_dual/prostate_all_domains_seed0.json) |
| P10 | D3-full | [json](results/dinov3_single/prostate_dinov3_single_domain_A.json) |
| P11 | nnU | [json](results/source_only_nnunet_official/prostate_source_only_nnunet_official_domain_A.json) |
| P12 | nnU\* | [json](results/source_only_nnunet/prostate_source_only_nnunet_domain_A.json) |
| P13 | R34 | [json](results/source_only/prostate_source_only_domain_A.json) |

**Long-term, 3 passes of B→F** (Crop-teach, blurry, largest, lr 1e-4, seed 0, [json](results/redifm_dinov3_nnunet_crop/prostate_long_term_seed0.json)): the rounds give 77.01 / 78.70 / 78.77, an average of 78.16.

Prostate slices are 384×384 in the public release, so prostate has no sharp-crop variant.

<details><summary>DINOv3 detector head as crop placer: box accuracy only, no DSC (†, <a href="results/detect_crop_tta/box_accuracy.py">script</a>)</summary>

CenterNet-style head on D3, trained 40 epochs on domain A. It predicts about 1 box per image, so "all boxes" and "best box" are the same. P/R = precision / recall of boxes matched to the ground truth at the given IoU.

| Domain | Boxes / image | P / R @ IoU 0.5 | P / R @ IoU 0.3 | Crop holds the whole prostate: detector vs D3-full mask |
|---|---|---|---|---|
| B | 1.02 | 86.5 / 86.7 | 96.7 / 96.6 | 99.7% vs 96.6% |
| C | 1.00 | 83.5 / 83.5 | 93.7 / 93.7 | 98.1% vs 100% |
| D | 1.04 | 63.9 / 64.7 | 79.8 / 80.1 | 83.1% vs 86.8% |
| E | 1.01 | 71.5 / 71.5 | 80.0 / 79.8 | 87.2% vs 87.6% |
| F | 1.01 | 54.0 / 54.3 | 67.6 / 68.0 | 81.1% vs 74.9% |
| All | 1.02 | 72.1 / 72.5 | 83.9 / 84.0 | 89.4% vs 89.4% |
</details>

### Polyp — targets B CVC-ClinicDB · C ETIS-Larib · D Kvasir-SEG (domain A: BKAI-IGH NeoPolyp)

Paper (ReDi-FM + SAM, Segformer-B0 target): **77.47**. Polyp's nnU uses 5 poolings at 352 px; nnU-Net itself would pad to 384 and use 6 (`NnUNet2D` in `src/common/models.py`).

**Adaptation**

| # | Method | B | C | D | **Do. A** |
|---|---|---|---|---|---|
| Y1 | Crop-teach, blurry, all, R34 | 77.65 | 57.63 | 80.85 | 72.04 |
| Y2 | Crop-teach, sharp, all, nnU | **82.22** | **73.96** | **85.07** | **80.42** |
| Y3 | † `crop_teach_tta.py` (standalone Crop-teach, largest), R34 | 76.69 | 58.26 | 81.04 | 72.00 |

**No adaptation**

| # | Method | B | C | D | **Do. A** |
|---|---|---|---|---|---|
| Y4 | `dinov3_crop`, blurry, all | **80.94** | **79.89** | **84.55** | **81.79** |
| Y5 | `dinov3_crop`, blurry, largest | 80.68 | 79.82 | 84.02 | 81.50 |
| Y6 | Source only, nnU | 73.90 | 73.18 | 81.14 | 76.07 |
| Y7 | Source only, R34 | 72.12 | 60.99 | 80.08 | 71.06 |

**Check: the crop-CNN alone, crops placed from the ground truth (2× each polyp)**

| # | Method | B | C | D | **Mean** |
|---|---|---|---|---|---|
| Y8 | † Sharp nnU crop-CNN | 81.64 | 83.15 | 85.50 | 83.43 |

**Setup and source**

| # | Places crop → predicts crop → target | Run |
|---|---|---|
| Y1 | D3-full → R34 crop-CNN → R34 | [seed 0](results/redifm_dinov3_nnunet_crop/polyp_all_regions_seed0.json) |
| Y2 | D3-full → nnU crop-CNN (sharp) → nnU | [seed 0](results/polyp_nnunet_sharp/adapt_all_seed0.json) |
| Y3 | D3-full → R34 crop-CNN → R34 | [scores](results/crop_teach_tta/polyp/scores.json) |
| Y4 | D3-full → R34 crop-CNN | [json](results/dinov3_crop/all_regions_polyp.json) |
| Y5 | D3-full → R34 crop-CNN | [json](results/dinov3_crop/polyp_dinov3_crop_domain_A.json) |
| Y6 | nnU | [json](results/polyp_nnunet_sharp/source_only_nnunet.json) |
| Y7 | R34 | [json](results/source_only/polyp_source_only_domain_A.json) |
| Y8 | ground truth → nnU crop-CNN (sharp) | [log](results/polyp_nnunet_sharp/crop_vs_full.log) |

### Loss search (dev / held-out split)

<details><summary>All loss variants (<code>extras/loss_search/</code>, cached supervision, seed 0)</summary>

The teachers' outputs were cached once (crop placement and pseudo ground truth fixed). Each variant then re-ran only the target model's updates. `pg_distill_bn_lr2e5` is the loss Crop-teach uses. A dash means the variant was not run on that problem.

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

Three polyp variants were also run ([agree_lr2e5](results/loss_search/polyp_agree_lr2e5_seed0_r1.json), [agree_lr4e5](results/loss_search/polyp_agree_lr4e5_seed0_r1.json), [pg_distill_bn_lr2e5](results/loss_search/polyp_pg_distill_bn_lr2e5_seed0_r1.json)):

| Loss | B | C | D | all | dev | held-out |
|---|---|---|---|---|---|---|
| agree_lr2e5 | 78.60 | 55.55 | 80.98 | 71.71 | 71.06 | 72.35 |
| pg_distill_bn_lr2e5 | 77.35 | 54.15 | 80.70 | 70.73 | 70.93 | 70.53 |
| agree_lr4e5 | 75.28 | 54.49 | 80.03 | 69.93 | 69.65 | 70.21 |
</details>

## Methods

| `--method` | File | What it does |
|---|---|---|
| `detect_crop_tta` | `src/methods/detect_crop_tta.py` | Crop-teach with a D3 detector placing the crops (prostate; not evaluated) |
| `dinov3_crop` | `src/baselines/dinov3_crop.py` | D3-full places crops, a crop-CNN predicts them. `--crop-region`, `--crop-from` |
| `dinov3_dual` | `src/baselines/dinov3_dual.py` | D3-full and D3-crop fused with Eq. 16 |
| `dinov3_single` | `src/baselines/dinov3_single.py` | D3-full on the whole image |
| `redifm_dinov3` | `src/methods/redifm_dinov3.py` | ReDi-FM with D3-crop in SAM's place. `--arch` picks the source/target CNN |
| `redifm_dinov3_dual` | `src/methods/redifm_dinov3_dual.py` | ReDi-FM with D3 only: the D3-full head is source and target |
| `redifm_dinov3_nnunet_crop` | `src/methods/redifm_dinov3_nnunet_crop.py` | Crop-teach. `--cnn`, `--crop-region`, `--crop-from`, `--no-augmentation` |
| `redifm_sam` | `src/methods/redifm_sam.py` | The paper's ReDi-FM, with frozen SAM ViT-H |
| `source_only` | `src/baselines/source_only.py` | Domain-A model, no updates. `--arch resunet34 / nnunet / nnunet_official` |

`crop_teach_tta.py` (repository root) is Crop-teach as one independent file, with no training: pass three models and a data folder.

<details><summary>The ReDi-FM protocol and loss in detail</summary>

**Setting.** A model trained on labeled domain A adapts online to the unlabeled target stream, one image at a time (batch size 1). Each image gets one optimizer step and is then predicted, and that prediction is scored. There's no reset between domains. OD/OC scores OD as prediction ≥ 1 against ground truth ≥ 1, and OC as prediction = 2 against ground truth = 2.

**Samples.** Images are resized (512 OD/OC, 384 prostate, 352 polyp) and min-max normalized per image. The frozen source gives nested structure probabilities, $P^{OD} = p_{rim} + p_{cup}$ and $P^{OC} = p_{cup}$, with source mask $M_s = P_s > 0.5$. The foundation teacher gives $P_{fm}$ and $M_{fm}$. The student sees an augmented view: ColorJitter (0.2, 0.2, 0.2, 0.05), then RandomErasing (p = 0.5, area 2–20%).

**ReDi-FM loss (Eqs. 6–16).**
- **Confidence weight:** $\Phi(P) = 1.3 - \mathcal{N}(P; 0.5, 0.1)$. It is about 0.04 at P = 0.5 and about 0.94 at P = 0 or 1.
- **Hard loss:** $L_{hard}(M, \Phi) = \overline{\Phi \cdot \mathrm{BCE}(P_t, M)} + 1 - \frac{1}{K}\sum_k \frac{2\sum \Phi P_t M}{\sum \Phi P_t + \sum \Phi M}$.
- **Consensus:** $r$ = the mean Dice of $M_s$ and $M_{fm}$ over structures.
- **Soft loss:** a hard class-balanced soft loss with entropy threshold 0.5, annealed by $r$.
- **Total:** $L = L_{hard}(M_{fm}, \Phi_{fm}) + r\, L_{hard}(M_s, \Phi_s) + L_{soft}$, with Adam at lr 1e-4.

**DINOv3 as the foundation teacher.**
- **The crop:** the largest source region, 2× its extent, at least 96 px.
- **The teacher:** the crop is resized to 512 and encoded by D3; a small convolutional head trained on domain-A crops predicts the classes.
- **Pasting back:** the prediction goes back into the image, with background outside the crop.

**Crop-teach loss** (`teacher_loss`, found with `extras/loss_search/`), summed over structures $k$:

$L = L_{hard}(M_{crop}, \Phi_{crop}) + \sum_k a_k\, L_{hard}(M^k_{D3}, \Phi^k_{D3}) - \overline{\sum_c q_c \log p_{t,c}}$

- $a_k$ is the Dice agreement of the two teachers on structure $k$.
- $q$ fuses the teachers per pixel: $(\Phi_c P_c + r\,\Phi_s P_s)/(\Phi_c + r\,\Phi_s)$.
- Updates use Adam at lr 2e-5, with BatchNorm statistics frozen.
</details>

## Reproducing

```bash
./reproduce.sh                  # setup → assets → odoc → prostate → long_term
./reproduce.sh odoc             # or only some stages (also: polyp)
```

| Stage | What it does |
|---|---|
| `setup` | Install `requirements.txt`. PyTorch is preinstalled on Kaggle; install it first locally |
| `assets` | `src/download/download.py`: DINOv3 and SAM from their official sources; data and checkpoints from the public bundle [`wiedergeburt/redifm-assets`](https://www.kaggle.com/datasets/wiedergeburt/redifm-assets) |
| `odoc`, `prostate`, `polyp` | The evaluations of each problem. A checkpoint missing from the bundle is trained on domain A first |
| `long_term` | The 3-round forgetting check |

Results from the latest experiments are not in `reproduce.sh` yet:
- sharp crops (`--crop-from original`);
- all regions;
- polyp nnU.

Their JSON `config` has the exact options to rerun them with `src/evaluate.py`.

- **Kaggle:** turn on GPU and Internet, attach `wiedergeburt/redifm-assets`, then run `git clone https://github.com/lqtrng/my-ctta-method.git /kaggle/working/ReDi-FM && cd /kaggle/working/ReDi-FM && ./reproduce.sh`. Run one stage per session if time runs out.
- **Bundle:** to publish a new version, run `python src/download/bundle.py build`, then `kaggle datasets version -p .cache/kaggle-bundle --dir-mode zip -m "..."`.
- **MLflow:** every run is logged to `mlruns/`. Browse it with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`. `REDIFM_MLFLOW=0` turns tracking off.
- **Files:** data lives in `data/`. Weights live in `/run/media/lqtrung/1tb-ssd/Model` when it exists, otherwise in `checkpoints/`; `REDIFM_MODEL_DIR` overrides that. This project's checkpoints are in its `ReDi-FM/` subfolder. Methods read only the paths in `src/common/paths.py` and never download.

```bash
python src/evaluate.py --method redifm_dinov3_nnunet_crop --task odoc --crop-from original \
    --foundation-checkpoint "$(python src/common/paths.py RESUNET_SHARP_CROP_CHECKPOINT)" --seed 0
python src/evaluate.py --method source_only --task prostate --arch nnunet_official
python src/evaluate.py --method redifm_dinov3 --rounds 3          # long-term CTTA
python src/evaluate.py --method <name> --help                     # every option of a method
```

## Layout

```
src/evaluate.py   the one evaluation script: streams the target domains, scores every method
                  the same way, writes results/<method>/ and logs to MLflow
src/common/       paths, data streams, tasks and scoring, networks, DINOv3 heads, losses, tracking
src/download/     the only code that uses the network: data, weights, prostate/polyp preparation, bundle
src/training/     train_source.py (R34 / nnU*), train_nnunet.py (nnU; --crop, --crop-from),
                  train_dinov3_head.py (D3 heads), train_dinov3_detector.py
src/baselines/    no-adaptation methods        src/methods/   adaptation methods
crop_teach_tta.py Crop-teach as one file     extras/        loss search, MedCLIP-SAMv2, figures
demo_dinov3_prostate/, demo_adaptation_D/      local viewers of predictions and adaptation steps
```
