# ReDi-FM: OD/OC continual test-time adaptation

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

Every method takes its evaluation protocol from `src/common/protocol.py` and its data stream from `src/common/data.py`, so the methods can be compared directly. Methods load weights only from the local paths in `src/common/paths.py` and never download. Weights live in the model folder: `/run/media/lqtrung/1tb-ssd/Model` when it exists, otherwise `checkpoints/` (set `REDIFM_MODEL_DIR` to override). Shared foundation models (DINOv3, BiomedCLIP config) sit at its top level; this project's checkpoints (ResUNet-34, SAM ViT-H, fine-tuned BiomedCLIP, DINOv3 decoder heads) sit in its `ReDi-FM/` subfolder. The dataset, the ResUNet-34 checkpoints and the fine-tuned MedCLIP-SAMv2 BiomedCLIP weights have no public download; place them by hand (`--check` lists where). The DINOv3 method is documented in `results/redifm_dinov3/WRITEUP.md` and `results/redifm_dinov3/EXPERIMENT_LOG.md`.
