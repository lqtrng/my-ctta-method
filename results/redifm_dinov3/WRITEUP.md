# ReDi-FM with a frozen DINOv3 foundation model

**Result (A → B, OD/OC, 3 seeds):** OD 81.99, OC 75.76, **Avg 78.88 ± 0.41**.

For comparison, source-only gets 54.54 and the original ReDi-FM with SAM ViT-H gets 72.46. The protocol is identical to the SAM method (`src/redifm_sam/adapt.py`); both import it from `src/common/`.

**Full stream B → C → D → E (Do. A, 3 seeds):** **77.55 ± 0.10**. The SAM version gets 71.72 here, and the paper reports 73.11. Per domain: B 78.88, C 77.61, D 79.22, E 74.49.

Reproduce: `src/redifm_dinov3/scripts/run_dinov3_domain_B.sh` (B only) or `src/redifm_dinov3/scripts/run_dinov3_all_domains.sh` (full stream). Code: `src/redifm_dinov3/adapt.py` (method) and `src/redifm_dinov3/train_head.py` (one-off decoder training).

## 1. DINOv3 prediction pipeline (replaces Section 3.2 prompting and Eq. 5)

1. **Source prompt → region of interest.** The frozen source ResUNet gives `p_s` and `ŷ_s = I[p_s > 0.5]` (Eq. 1, unchanged). Take the largest component of the source disc mask, or the probability maximum if the mask is empty, and define a square crop around it: 2× its extent, at least 96 px. This is the counterpart of the paper's structure-aware prompts. SAM is prompted with points; DINOv3 is "prompted" by where it looks.
2. **Frozen DINOv3 ViT-L/16** (LVD-1689M) encodes the crop at 512×512, giving 32×32 patch tokens of dimension 1024.
3. **Frozen decoder.** A small conv decoder (1×1 reduction, ×2 upsampling, two 3×3 convs) maps the tokens to {background, disc-rim, cup}. These are converted to nested OD/OC probabilities `p_fm`, upsampled into the crop, and set to background outside it. The decoder is trained once, before deployment, on labeled source domain A only, using random crop scale and shift plus color jitter. It plays the role of SAM's pretrained mask decoder; DINOv3 never receives gradients.
4. **Candidate selection (Eq. 5, kept).** `ŷ_fm = argmax_m Dice(m, ŷ_s)` over the candidate set. The default set is the decoder output alone. The training-free prototype candidates (center/hard-point and region prototypes vs. k-means background prototypes) remain available via `--candidates prototype|all`.

## 2. Combination rule (unchanged from the paper)

- **Consensus** `r = Dice(ŷ_s, ŷ_fm)` (Eq. 13) gates the source supervision. When source and DINOv3 disagree, the shift-sensitive source term is down-weighted.
- **Teacher classes.** `p_fm` is mapped to nested background/rim/cup classes (cup ≤ disc) for the soft term, as was done for SAM.

## 3. Learning objective (unchanged form, DINOv3 in SAM's slot)

For each target image, one Adam step on the target model, using a color-jitter and random-erasing view:

- `L_hard = L_seg(p_t, ŷ_fm; Φ(p_fm)) + r · L_seg(p_t, ŷ_s; Φ(p_s))`. This is weighted BCE + Dice with the Gaussian confidence weights Φ of Eq. 6 (Eqs. 7–8, 14).
- `L_soft = −(1/HW) Σ α · p_t log p_fm`, with the hard class-balanced weights α from η (Eqs. 10–12). These are annealed by `r` (Eq. 15).
- `L_total = L_hard + L_soft` (Eq. 16).

The objective can stay as it is because the redesigned DINOv3 branch emits exactly what SAM emitted: a binary mask and a probability map per structure.

## 4. Mapping to the original method

| ReDi-FM (0140.pdf) | This variant | Why |
|---|---|---|
| Frozen SAM ViT-H | Frozen DINOv3 ViT-L/16 | the swap the goal asks for |
| SAM's pretrained promptable mask decoder | Small decoder trained on source domain A, frozen at test time | DINOv3 has no decoder. Training-free prototype matching was tried; it inherits the source's under-segmentation (teacher 63.8 vs. 77.0 for the decoder). |
| Center and hard point prompts (Eqs. 2–4) | Source-guided disc crop (decoder); points used as prototypes (`prototype` mode) | the source model still tells the VFM where to look |
| Candidate selection by Dice with `ŷ_s` (Eq. 5) | Same rule | kept |
| Uncertainty-weighted hard distillation, class-balanced soft distillation, consensus gate (Eqs. 6–16) | Same | DINOv3 output has the same form as SAM's |
| One online update per image, bs = 1, Adam lr 1e-4, after-update scoring | Same (imported) | protocol identity |

**Why it beats the SAM version on OC.** SAM segments the high-contrast disc well (OD 84.5) but is weak on the low-contrast cup (OC 60.4). The DINOv3 decoder carries cup semantics learned from source labels on domain-robust features (teacher OC 73.2), and adaptation lifts OC further to 75.8.

**Caveat.** The DINOv3 branch now holds task knowledge learned from source labels, in addition to DINOv3's general knowledge. This differs from the paper's use of zero-shot SAM. It is allowed by the goal ("a lightweight head") and uses no target labels.
The crop is resized to 512×512, which gives DINOv3 a 32×32 grid of patches (16 px each).
