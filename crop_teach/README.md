# Crop-Teach: the ReDi-FM paper's experiments

A self-contained folder: it imports nothing from the rest of the repository. It reproduces the
ReDi-FM paper's (IJCAI-26) Tables 1–3 for Crop-Teach:

- **Tables 1 and 2:** each domain in turn is the source, and the model is scored on all the
  other domains.
- **Table 3:** a long-term version, with three passes over the stream.

## Method

For every target image, in stream order and never reset:

1. **Crop placer.** A frozen DINOv3 ViT-L/16 with a full-image head, trained on the source,
   predicts the image. Every region of its mask gets a square crop, 2× the region's size.
2. **Crop model.** A frozen CNN, trained on native-resolution source crops, predicts each crop.
   The crop is cut from the full-size image. The predictions are pasted back, with background
   outside the crops.
3. **Target model.** The whole-image source CNN takes one Adam update (lr 2e-5, batch-norm
   statistics frozen) with the paper's loss: crop teacher + gated DINOv3 teacher + hard
   class-balanced soft term. It then predicts the image, and that prediction is scored.

Networks follow the paper: ResUNet-34 for OD/OC and prostate, Segformer-B0 for polyp. Both start
from ImageNet weights.

## Training

`train.py --problem P --source S --kind K` trains one model on the labeled images of source
domain S only. The queue trains all three kinds for every source domain (15 × 3 = 45 models).

| `--kind` | Role at test time | Network | Input during training | Recipe |
|---|---|---|---|---|
| `target` | Source-only model; the target that adapts | ResUNet-34 (OD/OC, prostate), Segformer-B0 (polyp), from ImageNet | The whole image, resized to 512 / 384 / 352 | 200 epochs, batch 8, Dice + CE, polynomial decay. ResUNet-34: SGD lr 1e-3 (momentum 0.99, Nesterov, wd 5e-4). Segformer-B0: AdamW lr 6e-5 (wd 0.01). Augmentation: rotation, scaling, elastic, mirroring, brightness, noise; prostate also nnU-Net's intensity transforms |
| `crop` | Crop model: the teacher on each crop | Same network as `target` | Native-res crops (always; there is no low-res option, see below) | Same as `target` |
| `head` | Crop placer and second teacher | Frozen DINOv3 ViT-L/16 + a head on block 23: 20M parameters (hidden 1024, the default) or the original 1.4M with `CT_HEAD_HIDDEN=256` | The whole image at 2× size: 1024 / 768 / 704 | 40 epochs, batch 8, AdamW lr 1e-3 one-cycle, Dice + CE, horizontal flip, ColorJitter(0.4, 0.4, 0.4, 0.08) |

**Native-res crops.**
- **Box:** for each training image, a square around the structure, 1.5–3× its extent, with the
  center shifted by up to 0.25× its extent, drawn at random every time.
- **Pixels:** cut from the full-size original (e.g. 2124 × 2056 for REFUGE), not from the
  resized image, then resized to the input size with antialiasing.
- **At test time:** `adapt.py` cuts its crops the same way, so the crop model always sees
  native-res crops.
- **Prostate:** its slices are already stored at 384 × 384, so its native-res and low-res crops are
  the same pixels.

**Checkpoints.** Saved to `runs/checkpoints/<problem>/source_<S>_<kind>.pth`, under that name only
once training finishes. An existing checkpoint is skipped unless `--force`.

## Files

| File | Role |
|---|---|
| `config.py` | Paths (`CT_DATA`, `CT_MODELS`, `CT_RUNS`), the three problems, fixed settings |
| `data.py` | Reads the prepared data; resize + per-image min-max normalization |
| `networks.py` | ResUNet-34, Segformer-B0, frozen DINOv3 and its head |
| `losses.py` | The paper's loss (Eqs. 6–16) |
| `metrics.py` | Dice; S-alpha and max E-phi for polyp (checked against `py_sod_metrics`) |
| `train.py` | Trains one model (`target`, `crop` or `head`) on one source domain |
| `adapt.py` | Source-only or Crop-Teach on one source domain's target stream |
| `summarize.py` | Results → `runs/results/tables.md`, next to the paper's numbers |
| `prepare_data.py` | Run once where the raw data is: writes `data/` |
| `setup.sh`, `run_queue.sh` | Machine setup and the whole experiment queue |

## Running

```bash
./setup.sh                    # packages, ImageNet weights, checks data / DINOv3 / GPU
./run_queue.sh smoke          # phase 0: every code path on a few images (runs_smoke/)
./run_queue.sh odoc     &     # phases 1–3 for one problem; two problems can share a 16 GB GPU
./run_queue.sh prostate && ./run_queue.sh polyp
./run_queue.sh tables         # runs/results/tables.md
```

- **Resuming.** Every step skips its output if it already exists, so a stopped queue resumes
  where it left off.
- **Logs.** `runs/logs/`.
- **MLflow.** `mlflow ui --backend-store-uri sqlite:///runs/mlruns.db`.

## Choices not fixed by the paper

| Choice | Value | Why |
|---|---|---|
| Learning rate | 2e-5, batch-norm statistics frozen | Our choice; the paper uses 1e-4 (1e-5 polyp) |
| Entropy threshold in the soft loss | 0.5 | Not given by the paper |
| Augmentation strengths | ColorJitter(0.2, 0.2, 0.2, 0.05), RandomErasing(p 0.5, area 2–20%) | The paper names the transforms only |
| CNN initialization | ImageNet | Avoids starting from a model that has seen another benchmark domain |
| Segformer-B0 training | AdamW lr 6e-5 | The SegFormer default |
| Stream for source X | All other domains, alphabetical | The paper's "fixed order (A→B→…)" |
