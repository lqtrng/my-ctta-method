# ReDi-FM with DINOv3: experiment log (source A → target B, OD/OC)

Protocol: unchanged from the SAM method (`src/redifm_sam/adapt.py`). The data stream, 512×512 preprocessing, one Adam update per image (batch size 1, lr 1e-4), scoring after the update, per-image DSC, and the domain mean all come from the shared `src/common/` modules that both methods import. Every run starts from `domain_A_ResUNet34_last_20261017_215955.pth` (originally `checkpoints/`, now `/run/media/lqtrung/1tb-ssd/Model/ReDi-FM/`). Target: domain B (REFUGE train, 400 images). Avg = (OD + OC) / 2.

Reference points on domain B:

| Method | OD | OC | Avg |
|---|---|---|---|
| Source only (`results/source_only/baseline_no_adaptation_domain_A.json`) | 62.39 | 46.69 | 54.54 |
| ReDi-FM with SAM ViT-H (`results/redifm_sam/proposed_redifm_domain_A.json`) | 84.52 | 60.41 | 72.46 |

## Environment decisions

- **DINOv3 weights.** The official `facebook/dinov3-*` HF repos are gated, and the logged-in account (`lqtrung06`) isn't on the access list. I used the ungated timm ports of the same official LVD-1689M weights (`timm/vit_large_patch16_dinov3.lvd1689m`). They're saved in `/run/media/lqtrung/1tb-ssd/Model/timm-vit_large_patch16_dinov3.lvd1689m`, outside the repo. ViT-B/16 was downloaded too but not used.
- **Variant.** ViT-L/16 (300M) fits the 8 GB RTX 5060 alongside the trainable ResUNet.
- **Precision.** DINOv3 overflows to NaN in float16, so inference uses bfloat16 autocast.
- **Install.** `timm==1.0.30` was installed with `--no-deps`, so torch and transformers are unchanged.
- **Device.** All models run on CUDA. The original script's CPU default only affects speed, not the protocol.

## Full online runs (400 images, domain B)

| Exp | Output | Config change (all others default) | OD | OC | Avg |
|---|---|---|---|---|---|
| 01 | `exp01_default.json` | `--candidates prototype`: training-free prototype matching, crop scale 2.0, T=0.05, confidence 0.9, 4 bg prototypes | 76.68 | 61.69 | 69.18 |
| 02 | `exp02_head.json` | `--candidates head`: decoder trained on domain A, last DINOv3 block, seed 42 | 81.94 | 75.05 | **78.50** |
| 03 | `exp03_head_seed{0,1,2}.json` | as 02, seeds 0/1/2 | 81.87 / 81.85 / 82.24 | 75.02 / 75.97 / 76.29 | 78.45 / 78.91 / 79.27 |
| 04 | `exp04_head_l11-17-23.json` | as 02, decoder on blocks 11+17+23 | 81.51 | 74.37 | 77.94 |
| final | `final_domain_B_seed{0,1,2}.json` (`src/redifm_dinov3/scripts/run_dinov3_domain_B.sh`) | as 03 (default config) | **81.99** | **75.76** | **78.88 ± 0.41** |

The final configuration beats the >74 target on every seed.

## Full Domain-A protocol (continual stream B → C → D → E, no reset)

Command: `src/redifm_dinov3/scripts/run_dinov3_all_domains.sh`. Outputs: `final_all_domains_seed{0,1,2}.json` and `all_domains_run.log`. The config is the final one above, the same as the original script's `--domains B C D E` run. Values are averaged over seeds 0/1/2.

| Domain | OD | OC | Mean | SAM ReDi-FM (mean) |
|---|---|---|---|---|
| B (REFUGE) | 81.99 | 75.76 | 78.88 | 72.46 |
| C (ORIGA) | 85.02 | 70.20 | 77.61 | 69.94 |
| D (REFUGE val/test) | 80.14 | 78.29 | 79.22 | 68.76 |
| E (Drishti-GS) | 83.46 | 65.53 | 74.49 | 75.75 |
| **Do. A** | | | **77.55 ± 0.10** (77.57 / 77.42 / 77.65) | 71.72 (paper: 73.11) |

Pooled per-image DSC (diagnostic): 78.27 / 78.21 / 78.61. Domain E is the only domain below the SAM version (−1.3), because of OC.

## Long-term CTTA / catastrophic forgetting (0140.pdf Table 3 protocol)

Command: `src/redifm_dinov3/scripts/run_dinov3_long_term.sh` (`src/redifm_dinov3/long_term.py`). The run is 3 rounds of B → C → D → E. The target model and Adam state are never reset. The per-image protocol is the same as above, with the final config. Output: `long_term_seed0.json`.

**Only seed 0 was run.** The user asked to stop after seed 0, so seeds 1 and 2 are not done.

| Round | B | C | D | E | Do. A |
|---|---|---|---|---|---|
| 1 | 81.87 / 75.02 / 78.45 | 84.15 / 69.84 / 77.00 | 80.72 / 78.49 / 79.61 | 83.90 / 66.57 / 75.24 | 77.57 |
| 2 | 79.87 / 73.55 / 76.71 | 85.20 / 69.14 / 77.17 | 79.58 / 78.11 / 78.84 | 80.12 / 62.23 / 71.17 | 75.97 |
| 3 | 80.90 / 73.36 / 77.13 | 85.68 / 69.59 / 77.63 | 79.98 / 78.84 / 79.41 | 84.68 / 66.11 / 75.39 | 77.39 |

Cells are OD / OC / mean.

Summary: Average 76.98, Gain +23.31 over Source Only (53.67), Perform. Degra. (Round 1 − Average) **0.59**.

For reference, the paper reports these Perform. Degra. values on OD/OC, averaged over 5 sources: ReDi-FM 0.43, SAR 0.08, SURGEON 0.02, PALM 1.16, TENT 6.87, CoTTA 15.78.

Reading: there is no catastrophic forgetting. Round 3 is back within 0.2 of round 1. The round-2 dip comes mostly from domain E (−4.1, mostly OC), with a smaller drop on B (−1.7). E is the last and smallest domain. C improves slightly with every round.

## Offline teacher-quality diagnostics (no adaptation)

These runs use `src/redifm_dinov3/diagnostics/teacher_quality.py` and `dinov3_probe.py` in the same folder. They score DSC of the selected DINOv3 mask (the "foundation teacher") against GT on all 400 B images. Source-model pseudo-label avg: 54.38.

| Variant | FM OD | FM OC | FM Avg | Note |
|---|---|---|---|---|
| Linear probe fitted per image on source pseudo-labels | 36.7 | 37.2 | 37.0 | first 40 images; overfits source errors, dropped |
| Prototypes, crop scale 1.5 / **2.0** / 3.0 | 69.1 / 73.8 / 73.1 | 53.5 / 53.8 / 52.1 | 61.3 / **63.8** / 62.6 | |
| + 8 bg prototypes | 70.3 | 52.9 | 61.6 | worse |
| + bg margin 2 / 4 / 6 patches | 71.9 / 74.6 / 75.7 | 53.9 / 51.0 / 44.8 | 62.9 / 62.8 / 60.2 | trades OC for OD |
| + EM refinement 1 / 3 iterations | 72.9 / 75.6 | 51.2 / 43.6 | 62.1 / 59.6 | worse |
| + cup contrasted with rim only | 73.8 | 49.8 | 61.8 | worse |
| Oracle: prototypes from GT, first 200 images | 92.9 | 70.8 | 81.9 | upper bound: the features are good, the source prompts are the bottleneck |
| **Decoder head (domain A), last block** | **80.85** | **73.22** | **77.04** | chosen |
| Decoder head, blocks 11+17+23 | 80.59 | 72.18 | 76.39 | |
| Head + 3 prototype candidates, Eq. 5 selection | 72.8 | 56.5 | 64.6 | selection prefers source-like (under-segmented) masks |

Diagnostic finding: on B the source model under-segments, with median area/GT of 0.69 for both OD and OC. Prototypes derived from source pseudo-labels inherit this bias. A decoder trained on source labels over frozen DINOv3 features does not.

## Decisions on unspecified points (resolved per §6 of `goal.md`)

1. **The DINOv3 prediction comes from a decoder trained on source data.** SAM's usefulness in the paper comes from its pretrained mask decoder, and DINOv3 has none. The goal lists "a lightweight head" as an option. The head is trained once, before deployment, on the same labeled domain-A data as the source model. DINOv3 is never updated, the head is frozen at test time, and no target labels or target data are used for training. This matches the paper's "source data are not accessible during adaptation".
2. **Prompting = source-guided crop.** The paper prompts SAM with points from the source pseudo-label. Here, the source disc mask sets the region-of-interest crop: 2× its extent, at least 96 px, encoded at 512 px. The crop size was chosen a priori and confirmed by diagnostics.
3. **Eq. 5 candidate selection is kept, with one candidate.** The `all` mode applies Eq. 5 over the head and prototype candidates. It is implemented but hurts, because Dice-to-source favors the source's own bias.
4. **The learning objective is unchanged** (Eqs. 6–16). The DINOv3 output has the same form as SAM's: per-structure binary mask plus probability. So hard distillation, class-balanced soft distillation and the consensus gate `r` apply directly.
5. **Tuning caveat.** Choices were guided by domain-B GT diagnostics, as the goal requested. The head's training recipe (40 epochs, crop jitter, color jitter) was not tuned on B.
