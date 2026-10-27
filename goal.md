# Goal: Replace the VFM in the ReDi-FM adaptation method with DINOv3

## 1. Background

The method in `0140.pdf` (implemented in `src/proposed_adaptation_method.py`) adapts a source-trained segmentation model to a new target domain as follows:

1. The **source model** (trained on domain A) predicts on each target sample. It carries task-specific knowledge.
2. A **vision foundation model (VFM)** predicts on the same sample. It carries general visual knowledge.
3. The two predictions are **combined into an augmented training sample** (pseudo-label / supervision signal).
4. The **target model** is updated to learn from that sample using the paper's learning objective.

## 2. Objective

Replace the current VFM with **DINOv3** while keeping the core idea: target supervision must still **combine the source model's task-specific knowledge with the foundation model's general knowledge**.

DINOv3 is a self-supervised feature backbone, not a promptable segmenter. A drop-in swap therefore won't work, and the following must be redesigned:

- **How the DINOv3 prediction is produced.** Turn DINOv3 dense features into a segmentation-relevant signal for OC/OD. Examples include feature-similarity/prototype matching seeded by source-model predictions, a lightweight head, or feature-based refinement of source masks. The choice is open.
- **How the two predictions are combined** into the augmented training sample.
- **The learning formula** the target model optimizes, adapted to the new kind of DINOv3 output.

Keep each design choice as close to the paper's intent as possible (see §6).

## 3. Resources

| Item | Location |
|---|---|
| Method paper (primary reference) | `0140.pdf` |
| Secondary reference | `CoTTA.pdf` |
| Current implementation | `src/proposed_adaptation_method.py` |
| Source checkpoint (domain A) | `checkpoints/domain_A_ResUNet34_last_20261017_215955.pth` |
| Data | `data/` |
| DINOv3 checkpoints (to download) | `1tb-ssd/Model/` |

Download any DINOv3 weights needed (choose a variant that fits the hardware) into `1tb-ssd/Model/`. Don't place them inside the repo.

## 4. Evaluation protocol (strict)

- The evaluation protocol must be **identical to the one in `src/proposed_adaptation_method.py`**: the same data splits/ordering, preprocessing, online/episodic adaptation setting, metric computation (DSC), post-processing and reporting. Don't change anything that affects comparability.
- **Current scope:** run only on **target domain B** for the **OC/OD (optic cup / optic disc) segmentation** task. The full 5-domain run is deferred because it takes too long.
- Always start adaptation from the domain-A checkpoint listed above.

## 5. Success criterion

- **Average DSC (mean of OC and OD DSC, computed as in the existing script) > 74 on domain B.**
- Keep iterating on design and hyperparameters **until this target is met**. Don't stop at a lower result.
- Log every experiment (config, changes, OC DSC, OD DSC, avg DSC) under `results/` so the progress is traceable.

## 6. Resolving ambiguity

When something isn't specified here, follow the references in this priority order:

1. `0140.pdf` (highest priority)
2. `CoTTA.pdf`

If a choice is still unresolved, take the option most consistent with the paper's design philosophy and record the decision and the reasoning in the experiment log.

## 7. Deliverables

1. Updated adaptation code using DINOv3 (new file or clearly separated variant; don't break the original method).
2. A script/command that reproduces the domain-B OC/OD evaluation.
3. An experiment log in `results/` showing the final configuration achieving avg DSC > 74.
4. A short write-up of the DINOv3 prediction pipeline, the combination rule and the learning formula, and how each maps to the original method in `0140.pdf`.
