# Crop-Teach refactor: architecture

The design for `refactor_crop_teach/`, from [GOAL.md](../GOAL.md). The folder is independent: it
imports nothing from `crop_teach/` and has its own data, checkpoints and weights.

Scope:
- adaptation and evaluation only, no training: every model is loaded from a trained checkpoint;
- OD/OC and prostate (ResUNet-34); no SegFormer, so no polyp or COD;
- no DINOv3;
- no command-line arguments: settings are constants inside the class that uses them. Anything
  that varies becomes a new subclass of an interface. Exceptions, passed as constructor
  arguments: the problem and source domain, and the seed, which is given to `Adapting`.

`S` below is the problem's image size: 512 for OD/OC, 384 for prostate. It is not a fixed 512.

## 1. Pipeline: one adaptation step

```
 [1] DataLoader (the subclass sets the order)
      yields TestImage: image (S×S, stretched), full (original resolution), label (S×S)
      │
      ▼
 [2] full-image Model: SamHead (frozen)
      image ──► supervision_1 (S×S class probabilities)
      │
      ▼
 [3] RegionProposal
      a. supervision_1 ──► threshold ──► regions ──► square boxes (2× extent), in the S frame
      b. each box ──► scaled per axis to full resolution (×H/S, ×W/S)
      c. that rectangle is cut from `full`
      d. the cut is resized to S×S
      ──► list[Region(crop: S×S tensor, box: position in the S frame)]
      │
      ▼
 [4] crop Model: ResUNet34 (frozen, trained on source full-resolution crops)
      each region.crop ──► crop prediction (S×S class probabilities, in the crop's own frame)
      │
      ▼
 [5] Fusion
      each crop prediction ──► resized to its box size ──► pasted at region.box in the S frame
      overlaps: the most-foreground crop wins
      ──► supervision_2 (S×S nested structure probabilities)
      │
      ▼
 [6] Learning
      a. Augmentation(image) ──► augmented image (color jitter + random erasing)
      b. target Model(augmented) ──► student prediction
      c. ReDi-FM loss(student, supervision_1, supervision_2)
      d. one Adam step, lr 2e-5, batch-norm statistics frozen
      │
      ▼
 [7] Adapting
      target Model(image), clean, eval mode ──► prediction
      │
      ▼
 [8] EvaluateAdapting
      score(prediction, label) ──► per image ──► per domain ──► per round ──► JSON / CSV
```

- `Adapting` owns steps [2] to [7] for one image. It keeps the three models, the strategies and
  the optimizer state across the whole stream; it is never reset. Its `seed` argument is set
  once, when `Adapting` is created, and never again.
- `EvaluateAdapting` runs [1] → `Adapting` → [8]. It passes only `image` and `full` to
  `Adapting`, never `label`.
- `EvaluateNonAdapting` runs [1] → one frozen Model → [8]: source only (target ResUNet34) and
  SAM head alone.

### Resolution: where full resolution is used

Full resolution is used only at the input of the crop model. Everything else works in the
stretched S×S frame.

```
  S frame                   full resolution             crop model
  (shared canvas)           (e.g. 2048×2048)            (input = output = S×S)

  ┌──────────────┐          ┌──────────────┐
  │   ┌──┐       │  ×H/S    │   ┌───────┐  │  resize    ┌──────────┐
  │   │B │       │ ───────► │   │  cut  │  │ ─────────► │  input   │
  │   └──┘       │  ×W/S    │   │480×480│  │            └────┬─────┘
  │   120×120    │          │   └───────┘  │                 │ crop model
  └──────────────┘          └──────────────┘                 ▼
         ▲                                              ┌──────────┐
         │        resize S → 120, paste at B            │  output  │
         └──────────────────────────────────────────────└──────────┘
```

- Non-square images are stretched to S×S, not padded. The label is stretched the same way, and
  scoring happens in the S frame. Because the full-resolution cut is scaled per axis and then
  resized to S×S, the crop model sees the same stretch as the full-image model, so everything
  stays aligned.
- Cutting from the S×S image and upscaling (the "blurry" variant) scored lower; the cut must come
  from `full`.

## 2. Class diagram

Legend: `◆───►` has (a field) · `- - ->` uses (an argument) · `───▷` / `△` extends

### 2.1 Evaluation layer

```
┌──────────────────────────┐                     ┌──────────────────────────┐
│     EvaluateAdapting     │                     │   EvaluateNonAdapting    │
├──────────────────────────┤                     ├──────────────────────────┤
│ + run() -> Result        │                     │ + run() -> Result        │
└──┬────────────────────┬──┘                     └──┬────────────────────┬──┘
   ◆                    ◆                           ◆                    ◆
   │                    │                           │                    │
   ▼                    ▼                           ▼                    ▼
DataLoader           Adapting                   DataLoader             Model
   ┆                    ┆
   ┆ yields             ┆ receives image, full (never label)
   ▼                    ▼
TestImage - - - - - - - ┘
```

### 2.2 Adapting

```
                    ┌───────────────────────────────────────┐
                    │               Adapting                │
                    ├───────────────────────────────────────┤
                    │ + __call__(image, full) -> prediction │
                    └─┬──────┬──────┬──────┬──────┬──────┬──┘
                      ◆      ◆      ◆      ◆      ◆      ◆
                      │      │      │      │      │      │
     full_model ◄─────┘      │      │      │      │      └─────► Learning
     : Model (SamHead)       │      │      │      │
                             │      │      │      └────► Fusion
     crop_model ◄────────────┘      │      │
     : Model (ResUNet34)            │      └────► RegionProposal
                                    │
     target_model ◄─────────────────┘
     : Model (ResUNet34)

     Learning.step(target_model, ...) - - -> Model        (uses; Adapting owns it)
     RegionProposal.propose(sup1, full) - - -> Region     (creates and returns)
     Fusion.fuse(predictions, regions) - - -> Region      (uses)
```

### 2.3 Learning

```
┌───────────────────────────────────────────────┐
│             «abstract» Learning               │
├───────────────────────────────────────────────┤
│ + step(model, image, sup1, sup2)              │
│   -> dict (loss, consensus)                   │
└──┬─────────────────────────┬──────────────────┘
   ◆                         ◆
   ▼                         ▼
Augmentation           torch.optim.Adam
```

Learning owns everything after the two supervisions exist: augmentation, the student forward
pass, the loss, the optimizer and the batch-norm freezing.

### 2.4 Interfaces and their implementations

```
«abstract» Model          «abstract» RegionProposal     «abstract» Fusion
 + __call__(image)         + propose(sup1, full)         + fuse(predictions, regions, size)
   -> class probs            -> list[Region]               -> supervision_2, merged
 + load(checkpoint)               △          △                     △
    △          △                  │          │                     │
    │          │           PerRegionSquare  UnionSquare      MostForeground
 ResUNet34   SamHead

«abstract» Learning       «abstract» Augmentation       «abstract» DataLoader
 + step(...)               + __call__(image) -> image    rounds = 1
    △                            △                       + __iter__() -> TestImage
    │                            │                       # domains(), # one_pass()
 CropTeachLearning          JitterErase                     △               △
 (ReDi-FM loss,                                             │               │
  Adam 2e-5, frozen BN)                          AlphabeticalStream   LongTermStream
                                                  rounds = 1           rounds = 3
```

### 2.5 Data classes

```
TestImage                              Region
  domain, name, round                    crop: Tensor 1×3×S×S (cut from full resolution)
  image: S×S          → models           box:  (top, left, h, w) in the S frame
  full:  original res → RegionProposal
  label: S×S mask     → evaluators only
```

`label` holds class indices per pixel (int64), resized with nearest-neighbor: OD/OC 0 = background,
1 = disc rim, 2 = cup; prostate 0 = background, 1 = prostate. It is called `label`, not `target`,
because "target" already means the target model and the target domain.

## 3. Components

### Model

| Class | What it is |
|---|---|
| `ResUNet34` | the ReDi-FM paper's ResUNet-34, at S×S. Used as the crop model and the target model. |
| `SamHead` | SAM ViT-H image encoder (frozen, bfloat16) plus a small trained convolutional head. Resizes the S×S input to 1024×1024 for SAM and returns S×S class probabilities. |

### RegionProposal

Both strategies threshold the first structure (disc for OD/OC, prostate for prostate) of
supervision_1 at 0.5, then make square boxes with side = 2 × the region's extent, at least 96 px,
kept inside the image. If nothing passes the threshold, there is one box around the most confident
pixel.

| Class | Boxes |
|---|---|
| `PerRegionSquare` (default) | one box per connected region, in `ndimage.label` order (raster order of each region's first pixel) |
| `UnionSquare` | one box around all regions together |

### Fusion: `MostForeground`, exact order

```
merged = None
inside = zeros(1×1×S×S)

for each region, in proposal order:
    1. probs  = crop prediction (C×S×S class probabilities)
    2. pasted = zeros(C×S×S)                               # all classes 0, not background = 1
    3. pasted[box] = bilinear_resize(probs, h×w)           # placed at (top, left)
    4. inside[box] = 1
    5. if merged is None:
           merged = pasted                                 # first region taken as is
       else:
           better = sum(pasted[1:]) > sum(merged[1:])      # foreground, per pixel, strict >
           merged = where(better, pasted, merged)          # whole class vector swapped

after the loop:
    6. structures    = structure_probabilities(merged)     # classes → nested (disc, cup) or (fg)
    7. supervision_2 = clamp(structures * inside, 1e-6, 1 − 1e-6)
```

- Ties go to the earlier region, because the comparison is a strict `>`.
- Outside every box, supervision_2 is 1e-6, so the crop mask there is background. The loss does
  not ignore those pixels: they count as confident background.
- `merged` (class probabilities) is also returned, for scoring the crop teacher on its own.

Other fusion strategies that could be added later: `Average`, `ConfidenceWeighted`,
`CenterWeighted`.

### Learning: `CropTeachLearning`

The ReDi-FM paper's loss (Eqs. 6–16), with supervision_1 from the SAM head and supervision_2
from the crop model:

```
L = Lhard(crop mask, Φ(crop)) + r · Lhard(SAM-head mask, Φ(SAM head)) + Lsoft
```

- Φ: confidence weight, γ = 1.3, μ = 0.5, σ² = 0.1.
- Lhard: confidence-weighted BCE + Dice against a teacher's binary mask.
- r: Dice of the two teachers' masks, gating the SAM-head term.
- Lsoft: class-balanced soft distillation, annealed by r; entropy threshold 0.5.
- Pseudo-label threshold τ = 0.5. Adam, lr 2e-5, one update per image, over all target-model
  parameters, with batch-norm layers in eval mode (statistics frozen).

### Augmentation: `JitterErase`

`ColorJitter(brightness 0.2, contrast 0.2, saturation 0.2, hue 0.05)` then
`RandomErasing(p 0.5, scale (0.02, 0.2), ratio (0.3, 3.3), value "random")`. Applied to the
student's input only.

### DataLoader

The base class reads and prepares each image and repeats the stream `rounds` times. Each subclass
defines only its single pass and its `rounds`. Subclasses extend `DataLoader` directly; none
extends another.

| Class | Order |
|---|---|
| `AlphabeticalStream` | the paper's protocol: every domain but the source, in alphabetical order, one pass, never reset |
| `LongTermStream` | the same order, repeated 3 times, never reset |

## 4. Files

| Path | Contents |
|---|---|
| `data/<problem>/domain_<letter>/{Images,Masks}/` | OD/OC and prostate, copied from `crop_teach/data/` |
| `checkpoints/<problem>/source_<S>_{head,crop,target}.pth` | trained models, copied from `crop_teach/runs/checkpoints/` |
| `weights/sam_vit_h_4b8939.pth` | SAM ViT-H image encoder weights |
| `weights/imagenet/` | ResNet-34 and MiT-B0 ImageNet weights, for training later (not used now) |
| `results/<problem>/<method>_source<S>.{json,csv}` | written by `run.py`: `source_only`, `sam_head`, `crop_teach_seed<k>` |

Code (run everything from this folder; imports are `from <package>.<module> import ...`):

```
refactor_crop_teach/
├── run.py                      # entry point; selects concrete components
├── evaluation/
│   └── evaluate.py             # combines adaptation, data and models
├── adaptation/
│   ├── adapting.py             # combines the adaptation components
│   ├── learning.py
│   ├── augmentation.py
│   ├── region_proposal.py
│   └── fusion.py
├── modeling/
│   └── models.py
├── data/
│   ├── data_loader.py          # next to the datasets (data/<problem>/)
│   └── <problem>/domain_<letter>/...
├── core/                       # shared definitions
│   ├── problem.py
│   └── data_types.py
├── tests/
│   ├── helpers.py
│   └── test_*.py
└── conftest.py
```

| File | Contents |
|---|---|
| `run.py` | `python run.py`: every problem and source, the three methods; skips results that exist |
| `evaluation/evaluate.py` | `EvaluateAdapting`, `EvaluateNonAdapting`, `Result`, `score` |
| `adaptation/adapting.py` | `Adapting`; `Adapting.last` holds the last step's loss, consensus, crop count and the two teachers' predictions |
| `adaptation/learning.py` | `Learning`, `CropTeachLearning` (the loss lives here) |
| `adaptation/augmentation.py` | `Augmentation`, `JitterErase` |
| `adaptation/region_proposal.py` | `RegionProposal`, `PerRegionSquare`, `UnionSquare` |
| `adaptation/fusion.py` | `Fusion`, `MostForeground` |
| `modeling/models.py` | `Model`, `ResUNet34`, `SamHead` |
| `data/data_loader.py` | `DataLoader`, `AlphabeticalStream`, `LongTermStream` |
| `core/problem.py` | `Problem` (`ODOC`, `PROSTATE`), the paths, classes ↔ nested structures |
| `core/data_types.py` | `TestImage`, `Region` |
| `tests/` | `python -m pytest tests` |

## 5. Tests

Every class that implements an interface has a unit test: `ResUNet34`, `SamHead` (with a small
fake encoder in place of SAM), `PerRegionSquare`, `UnionSquare`, `MostForeground`, `JitterErase`,
`CropTeachLearning`, `AlphabeticalStream` and `LongTermStream`.

Tests worth having beyond that:
- `RegionProposal` with a non-square full-resolution image: the cut rectangle maps back to the
  right pixels.
- `MostForeground`: the strict `>` (ties go to the earlier region), zeros outside every box, the
  clamp.
- `EvaluateAdapting` never passes `label` to `Adapting`.
