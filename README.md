# ReDi-FM: OD/OC continual test-time adaptation

This repository reproduces ReDi-FM (`0140.pdf`) for optic disc (OD) and optic cup (OC) segmentation under continual test-time adaptation (CTTA). It also contains a variant in which the frozen SAM foundation model is replaced with a frozen DINOv3. The variant keeps ReDi-FM's idea of combining knowledge from the source model and a foundation model.

## Method

### Setting
- **Source domain.** A ResUNet-34 is trained on labeled domain A (RIM-ONE-r3).
- **Target stream.** At test time the model adapts online, without labels, to the unlabeled target stream B (REFUGE train) → C (ORIGA) → D (REFUGE val/test) → E (Drishti-GS). The stream runs in one pass with no reset between domains.
- **Per-image procedure.** Each image arrives alone (batch size 1). The model takes one Adam step on it (lr 1e-4), then predicts the same image, and that prediction is scored.
- **Metric.** Per-image Dice (DSC), averaged over OD and OC. OD includes the cup region. **Do. A** is the unweighted mean over domains B–E.

This protocol follows `0140.pdf` and CoTTA. It is implemented once in `src/common/` and shared by every method, so results are directly comparable.

### ReDi-FM (original, `src/redifm_sam/`)
For each target image, two teachers supervise the adapting ResUNet-34 (the student):

1. **Source teacher.** A frozen copy of the source model gives OD/OC probabilities. Thresholding at τ = 0.5 gives the source pseudo-masks.
2. **Foundation-model teacher.** Structure-aware point prompts are derived from the source prediction: a center point and a hard point for each structure (Eqs. 2–4). They prompt a frozen SAM ViT-H. Among SAM's candidate masks, the one with the highest Dice against the source pseudo-mask is kept (Eq. 5).
3. **Learning objective.**
   - Hard distillation from both teachers, weighted by each teacher's pixel confidence Φ (Eqs. 6–8).
   - Hard class-balanced soft distillation (Eqs. 10–12).
   - A consensus gate *r*, the agreement between the two teachers' masks, which decides how much the source pseudo-label is trusted (Eqs. 13–16).

   Total loss: `L = L_hard(FM) + r · L_hard(source) + L_soft`.

### ReDi-FM with DINOv3 (`src/redifm_dinov3/`)
Only the foundation-model teacher changes. The protocol, student, optimizer, augmentation, source pseudo-labels and the whole learning objective (Eqs. 6–16) are imported unchanged from `src/common/`.

DINOv3 is a feature backbone with no prompt encoder or mask decoder, so the teacher is built as follows:

1. **The prompt becomes a crop.** The largest connected component of the source OD mask defines a square region of interest. It has 2× the disc's extent and is at least 96 px wide.
2. **Frozen DINOv3 encoding.** The crop is resized to 512×512 and encoded by DINOv3 ViT-L/16 (LVD-1689M weights, bf16). This gives a 32×32 grid of patch tokens from the last block.
3. **Frozen decoder head.** A small convolutional head maps the tokens to background, disc-rim and cup probabilities. It has a 1×1 reduction to 256 channels, 2× upsampling and two 3×3 conv blocks. The head is the counterpart of SAM's mask decoder. It was trained once, before deployment, on labeled domain A only (`train_head.py`), and no target data is ever used. Training samples are disc-centered crops with jittered scale and position, so the head tolerates the imperfect crops the source model produces under domain shift.
4. **Teacher output.** The OD/OC probabilities are pasted back into the full image, with everything outside the crop treated as background. The result enters the unchanged ReDi-FM loss in place of SAM's mask and probability.

DINOv3, the decoder head and the source teacher stay frozen. Only the student ResUNet-34 is updated, and only the student's prediction is scored.

A training-free alternative is also available (`--candidates prototype`). It builds foreground prototypes from DINOv3 features at the structure-aware points, background prototypes from k-means over the confident source background, and then applies Eq. 5. On domain B it inherited the source model's under-segmentation (teacher DSC 63.8, against 77.0 for the head), so the head is the default. See `results/redifm_dinov3/EXPERIMENT_LOG.md` for the full history.

### One adaptation step in detail
Both methods run the same steps for every target image. They differ only in step 4. Code: `adaptation_step` in each `adapt.py`, plus `src/common/protocol.py` and `src/common/redifm.py`.

#### Creating the samples
1. **Load.** Read the RGB fundus image and its mask (`common/data.py`). Masks that use the benchmark convention (255 = background, 128 = disc rim, 0 = cup) are mapped to the classes {0 = background, 1 = rim, 2 = cup}. The mask is used only for scoring, never for adaptation.
2. **Resize and normalize.** Resize the image to 512×512 with bilinear interpolation (the mask uses nearest-neighbour). Convert it to a tensor *x* and min-max normalize it per image to [0, 1] (`prepare_sample`, `image_to_tensor`).
3. **Source sample.** The frozen source model gives class probabilities $p_s = \mathrm{softmax}(f_{src}(x))$ over (background, rim, cup). These are converted into nested structure probabilities, $P^{OD} = p_{rim} + p_{cup}$ and $P^{OC} = p_{cup}$, clamped to $[10^{-6}, 1 - 10^{-6}]$. The source pseudo-mask is $M_s = P_s > \tau$ with $\tau = 0.5$.
4. **Foundation-model sample.** The teacher gives OD/OC probabilities $P_{fm}$ and the mask $M_{fm} = P_{fm} > 0.5$. SAM is prompted with structure-aware points. For DINOv3, the image is cropped around the source disc and normalized with ImageNet mean and std before encoding (see above). For the soft loss, $P_{fm}$ is also turned into nested class probabilities: $disc = P^{OD}_{fm}$, $cup = P^{OC}_{fm} \cdot disc$, $rim = disc - cup$, $background = 1 - disc$, renormalized to sum to 1 (`nested_foundation_classes`).
5. **Student sample.** The student sees an augmented view $\tilde{x}$: ColorJitter (brightness 0.2, contrast 0.2, saturation 0.2, hue 0.05), then RandomErasing (p = 0.5, area 2–20 %, aspect 0.3–3.3, random values). The student runs in train mode, so BatchNorm uses the statistics of this image. This gives $p_t$ and $P_t$. Both teachers see the clean *x*; only the student sees $\tilde{x}$.

#### Computing the loss
- **Confidence weight (Eq. 6).** Each pixel is weighted by $\Phi(P) = \gamma - \mathcal{N}(P;\ 0.5,\ 0.1)$ with $\gamma = 1.3$. A teacher probability near 0.5 (uncertain) gets $\Phi \approx 0.04$; a confident one near 0 or 1 gets $\Phi \approx 0.94$. $\Phi_s$ is computed from $P_s$ and $\Phi_{fm}$ from $P_{fm}$.
- **Weighted hard loss (Eqs. 7–8).** For a teacher mask *M* with weights Φ, applied to both the OD and OC channels:

  $$L_{hard}(M, \Phi) = \overline{\Phi \cdot \mathrm{BCE}(P_t, M)} + 1 - \frac{1}{2}\sum_{k \in \{OD, OC\}} \frac{2\sum \Phi P_t M}{\sum \Phi P_t + \sum \Phi M}$$

- **Consensus (Eq. 13).** $r = \frac{1}{2}\sum_{k} \mathrm{Dice}(M_s^k, M_{fm}^k)$, the agreement between the two teachers. It is detached from the gradient.
- **Hard class-balanced soft loss (Eqs. 10–12, annealed by Eq. 15).**
  - Take the argmax labels of the source ($\hat{y}_s$), the student ($\hat{y}_t$, detached) and the foundation model ($\hat{y}_{fm}$).
  - A pixel is *highly uncertain* when the normalized entropy of $p_s$, $H(p_s)/\log 3$, exceeds 0.5.
  - For each class *c*, compute $\eta_c = \big(|\hat{y}_{fm} = c| + |\hat{y}_t \neq \hat{y}_{fm}|\big) / (2HW)$.
  - Pixels where $\hat{y}_s = c$, or that are highly uncertain, are *difficult* and get $\alpha_c = \eta_c$; the rest get $1 - \eta_c$.
  - The weights are annealed by consensus, $\alpha_c \leftarrow r\,\alpha_c + (1 - r)$, so they fade to uniform when the teachers agree.

  $$L_{soft} = -\,\overline{\sum_c \alpha_c\, p_{t,c} \log p_{fm,c}}$$

- **Total loss (Eq. 16).**

  $$L = L_{hard}(M_{fm}, \Phi_{fm}) + r \cdot L_{hard}(M_s, \Phi_s) + L_{soft}$$

  The foundation-model teacher always supervises. The source pseudo-label counts only as much as the two teachers agree.
- **Update and score.** Take one backward pass and one Adam step (lr 1e-4) on all student parameters. Then predict the clean *x* in eval mode, so BatchNorm statistics are not updated a second time, and take the argmax. Score OD as prediction ≥ 1 against ground truth ≥ 1, and OC as prediction = 2 against ground truth = 2. Then move to the next image without resetting.

The paper does not give the entropy threshold (0.5) or the augmentation magnitudes. They are command-line options, saved in each result JSON.

## Results (source domain A, Do. A DSC)

| Method | B | C | D | E | **Do. A** |
|---|---|---|---|---|---|
| Source only | 54.54 | 47.62 | 39.31 | 73.20 | 53.67 |
| ReDi-FM + SAM (our run, seed 42) | 72.46 | 69.94 | 68.76 | **75.75** | 71.73 |
| ReDi-FM + SAM (paper) | | | | | 73.11 |
| **ReDi-FM + DINOv3** (mean of seeds 0/1/2) | **78.88** | **77.61** | **79.22** | 74.49 | **77.55 ± 0.10** |

- **Where the gain comes from.** Almost all of it is in OC: +15 to +22 DSC on B, C and D. OD is on par with SAM.
- **Forgetting check.** In the long-term check (the Table 3 protocol, 3 rounds of B→C→D→E, seed 0), the rounds give 77.57 / 75.97 / 77.39. That is an average of 76.98, a gain of +23.31 over source only, and a performance degradation of 0.59.
- **Caveat.** The decoder head adds domain-A supervision that zero-shot SAM does not have. Design choices were also made with domain-B labels in view.
- **Further detail** is in `results/redifm_dinov3/WRITEUP.md`.

## Repository layout

Each method lives in its own folder under `src/`, with its outputs in the matching folder under `results/`. Code that more than one method uses lives in `src/common/`. All commands run from the repository root.

| Folder | Method | Entry point | Results |
|---|---|---|---|
| `src/common/` | shared: paths, OD/OC data stream, ResUNet-34, online CTTA protocol, ReDi-FM prompts and losses | n/a | n/a |
| `src/download/` | fetch external model weights (the only code that uses the network) | `download.py` | model folder (see below) |
| `src/source_training/` | train the ResUNet-34 source model on domain A | `train.py` | model folder, `ReDi-FM/` |
| `src/source_only/` | source-only baseline (no adaptation) | `evaluate.py` | `results/source_only/` |
| `src/medclip_samv2/` | zero-shot MedCLIP-SAMv2 baseline | `zero_shot.py` | `results/medclip_samv2/` |
| `src/redifm_sam/` | ReDi-FM with frozen SAM ViT-H (paper method) | `adapt.py` | `results/redifm_sam/` |
| `src/redifm_dinov3/` | ReDi-FM with frozen DINOv3 ViT-L/16 | `adapt.py`, `long_term.py`, `train_head.py`, `scripts/*.sh` | `results/redifm_dinov3/` |

```bash
.venv/bin/python src/download/download.py --check  # what is present, what is missing
.venv/bin/python src/download/download.py          # fetch DINOv3, SAM ViT-H, BiomedCLIP config
.venv/bin/python src/source_only/evaluate.py
.venv/bin/python src/redifm_sam/adapt.py --domains B C D E
src/redifm_dinov3/scripts/run_dinov3_all_domains.sh     # DINOv3, full stream, 3 seeds
src/redifm_dinov3/scripts/run_dinov3_long_term.sh       # DINOv3, 3-round forgetting check
```

Methods load weights only from the local paths in `src/common/paths.py` and never download.

**Where weights are stored.** Weights live in the model folder. That is `/run/media/lqtrung/1tb-ssd/Model` when it exists, otherwise `checkpoints/`; set `REDIFM_MODEL_DIR` to override. Shared foundation models (DINOv3, the BiomedCLIP config) sit at the top level of the model folder. This project's checkpoints (ResUNet-34, SAM ViT-H, the fine-tuned BiomedCLIP, the DINOv3 decoder heads) sit in its `ReDi-FM/` subfolder.

**Files you place by hand.** The dataset, the ResUNet-34 checkpoints and the fine-tuned MedCLIP-SAMv2 BiomedCLIP weights have no public download. `download.py --check` lists where each one goes.
