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
3. **Frozen decoder head.** A small convolutional head maps the tokens to background, disc-rim and cup probabilities. It has a 1×1 reduction to 256 channels, 2× upsampling and two 3×3 conv blocks. The head is the counterpart of SAM's mask decoder. It was trained once, before deployment, on labeled domain A only (`src/training/train_dinov3_head.py`), and no target data is ever used. Training samples are disc-centered crops with jittered scale and position, so the head tolerates the imperfect crops the source model produces under domain shift.
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
| DINOv3 single (one full-image DINOv3, no ResUNet, no adaptation) | 51.17 | 82.80 | 59.83 | 84.62 | 69.61 |
| DINOv3 dual (full-image DINOv3 + crop DINOv3, fused, no adaptation) | 41.52 | 85.10 | 61.75 | 84.01 | 68.09 |
| nnU-Net only (nnU-Net 2D source model, no adaptation) | 75.80 | 60.20 | 67.43 | 74.78 | 69.55 |
| ReDi-FM + DINOv3 with nnU-Net (seed 0) | 82.52 | 80.71 | 82.67 | 82.50 | 82.10 |

- **Where the gain comes from.** Almost all of it is in OC: +15 to +22 DSC on B, C and D. OD is on par with SAM.
- **Forgetting check.** In the long-term check (the Table 3 protocol, 3 rounds of B→C→D→E, seed 0), the rounds give 77.57 / 75.97 / 77.39. That is an average of 76.98, a gain of +23.31 over source only, and a performance degradation of 0.59.
- **Caveat.** The decoder head adds domain-A supervision that zero-shot SAM does not have. Design choices were also made with domain-B labels in view.
- **Further detail** is in `results/redifm_dinov3/WRITEUP.md`.

## Results (prostate, source domain A = BIDMC, prostate DSC)

The source models are trained on domain A with the repository recipe (`training/train_source.py --task prostate`). The target stream is B → F, with no reset and seed 0.

| Method | B (BMC) | C (HK) | D (I2CVB) | E (RUNMC) | F (UCL) | **Do. A** |
|---|---|---|---|---|---|---|
| Source only, ResUNet-34 | 19.07 | 72.54 | 9.60 | 23.47 | 7.75 | 26.49 |
| nnU-Net only (no adaptation) | 58.05 | 66.94 | 23.15 | 51.86 | 39.28 | 47.86 |
| ReDi-FM + DINOv3, ResUNet-34 | 52.21 | **77.61** | 43.02 | **64.79** | 36.96 | 54.92 |
| **ReDi-FM + DINOv3, nnU-Net** | **70.08** | 72.06 | **62.64** | 64.47 | **63.69** | **66.59** |
| Paper (`0140.pdf` Table 1): Source Only / ReDi-FM + SAM | | | | | | 51.02 / 66.10 |

## Reproducing the results

One script reproduces everything, from a fresh clone, locally or in a Kaggle notebook:

```bash
./reproduce.sh                  # setup → assets → odoc → prostate → long_term
./reproduce.sh odoc             # or only some stages
```

| Stage | What it does |
|---|---|
| `setup` | install `requirements.txt` (PyTorch itself is preinstalled on Kaggle; install it first locally) |
| `assets` | `src/download/download.py`: public foundation models (DINOv3, SAM ViT-H) from their official sources, and the data and this project's checkpoints from the private asset bundle |
| `odoc` | every row of the OD/OC table |
| `prostate` | every row of the prostate table |
| `long_term` | the 3-round forgetting check |

A checkpoint that the bundle lacks (a source model or a DINOv3 head) is trained on domain A before it is first used, with the recipe that produced it. `SEEDS`, `PYTHON` and the other settings are listed at the top of `reproduce.sh`.

### On Kaggle

1. In a new notebook, turn on a GPU and Internet, then use **Add Input** to attach the asset dataset `wiedergeburt/redifm-assets`. The owner shares it with you.
2. Run:
   ```bash
   !git clone https://github.com/lqtrng/my-ctta-method.git /kaggle/working/ReDi-FM
   !cd /kaggle/working/ReDi-FM && ./reproduce.sh
   ```
   The attached dataset is linked from `/kaggle/input`, not copied. Results (`results/`) and the MLflow store (`mlruns/`) are written under `/kaggle/working`, so they are kept with the notebook's output. A Kaggle session is limited in time, so you can run one stage per session (`./reproduce.sh setup assets odoc`, then `./reproduce.sh setup assets prostate`, …).

### Locally

The same commands work. `download.py` downloads the asset bundle once into `.cache/`, using the Kaggle CLI and your API token in `~/.kaggle/kaggle.json`. Files that are already in place are never fetched again.

### The asset bundle

The OD/OC datasets, the prepared prostate slices and polyp images and this project's checkpoints have no single public download. They are published once as a private Kaggle dataset, holding only the files the code reads (about 4.4 GB):

```bash
python src/download/bundle.py build                                   # stage into .cache/kaggle-bundle
kaggle datasets create -p .cache/kaggle-bundle --dir-mode zip         # first upload (private by default)
kaggle datasets version -p .cache/kaggle-bundle --dir-mode zip -m "new checkpoints"   # later
```

Keep it private: the OD/OC datasets carry their own licenses. `REDIFM_KAGGLE_DATASET` selects another dataset id.

## Experiment tracking (MLflow)

Every entry point logs one MLflow run to `mlruns/`, in the experiment `redifm-odoc` or `redifm-prostate` (`src/common/tracking.py`):

- **Parameters:** every command-line option.
- **Tags:** method, task, seed, git commit, and whether the tree had uncommitted changes.
- **Metrics:** Do. A and every per-domain number (`B/disc_dsc`, `C/mean_dsc`, …). Each per-image number (DSC, consensus *r*, teacher agreement) is also logged as a curve along the target stream (`stream/...`), so drift and forgetting are visible. Training scripts log the loss per epoch.
- **Artifacts:** the result JSON, which is still written to `results/` as before.

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db      # then open http://127.0.0.1:5000
```

Runs with `--max-images` are tagged `smoke_test=True`. Set `REDIFM_MLFLOW=0` to turn tracking off, or `MLFLOW_TRACKING_URI` to log to another server.

## Repository layout

`src/` is grouped by role. All commands run from the repository root.

```
src/
  evaluate.py  the one evaluation script: streams the target domains, scores every method
               the same way (per-image DSC, domain means, Do. A, long-term rounds), writes
               results/<method>/ and logs to MLflow
  common/      shared code: paths, data streams, tasks and their scoring, networks, DINOv3
               backbone and head, ReDi-FM prompts and losses, MLflow tracking, seed report
  download/    the only code that uses the network: fetch data, weights and checkpoints,
               prepare the prostate and polyp benchmarks, build the Kaggle asset bundle
  training/    train_source.py        ResUNet-34 or nnU-Net source model on domain A
               train_dinov3_head.py   DINOv3 decoder head on domain A (--full-image for the single/dual baselines)
  baselines/   source_only.py         no adaptation (--arch nnunet: the "nnU-Net only" row)
               dinov3_single.py       one frozen DINOv3 with a full-image head, no ResUNet, no adaptation
               dinov3_dual.py         full-image DINOv3 + DINOv3 crop teacher fused with Eq. 16, no adaptation
  methods/     redifm_sam.py          ReDi-FM with frozen SAM ViT-H (the paper's method)
               redifm_dinov3.py       ReDi-FM with frozen DINOv3 ViT-L/16 (--arch nnunet: nnU-Net source/target)
extras/        not part of reproduce.sh: zero-shot MedCLIP-SAMv2, DINOv3 teacher diagnostics, figure scripts
```

Every file in `baselines/` and `methods/` is a black box for `evaluate.py`: `build(args, task)` returns a segmenter that maps one target image to a class map (a CTTA method updates itself inside that call). The results folder and MLflow run are named `<method>[_<arch>]`, e.g. `results/source_only_nnunet/`.

```bash
python src/download/download.py --check                                   # what is present, what is missing
python src/evaluate.py --method source_only --task prostate
python src/evaluate.py --method redifm_sam --domains B C D E
python src/evaluate.py --method redifm_dinov3 --domains B --seed 0        # one domain, one seed
python src/evaluate.py --method redifm_dinov3 --rounds 3                  # long-term CTTA (Table 3)
python src/evaluate.py --method redifm_dinov3 --help                      # every option of a method
python src/common/report.py results/redifm_dinov3/final_all_domains_seed*.json   # mean over seeds
```

Methods load files only from the local paths in `src/common/paths.py` and never download.

**Where files live.** Data is in `data/`. Weights live in the model folder: `/run/media/lqtrung/1tb-ssd/Model` when it exists, otherwise `checkpoints/`; set `REDIFM_MODEL_DIR` to override. Shared foundation models (DINOv3, the BiomedCLIP config) sit at the top level of the model folder. This project's checkpoints (source models, SAM ViT-H, the fine-tuned BiomedCLIP, the DINOv3 decoder heads) sit in its `ReDi-FM/` subfolder. On Kaggle the bundle's files are symlinks into `/kaggle/input`.

**Prostate dataset.** The six-site prostate MRI benchmark of `0140.pdf` (Section 4.1) is in the asset bundle, already prepared. `download.py prostate-public` fetches the public release instead, and `src/download/prepare_prostate.py` lays it out. It lives in `data/prostate-6/domain_{A..F}_{BIDMC,BMC,HK,I2CVB,RUNMC,UCL}/{Images,Masks}`, with the same folder pattern as the OD/OC domains. Each domain holds the T2 slices that contain prostate, at 384×384: 261 / 384 / 158 / 468 / 421 / 175 slices, matching the paper. Masks are class-index PNGs (0 = background, 1 = prostate). `common.data.build_prostate_domains` loads them.
