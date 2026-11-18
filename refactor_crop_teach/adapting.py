from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .config import SAM_CHECKPOINT, checkpoint_path
    from .fusion import MaxForegroundFusion, RefinedRegionFusion
    from .learning import Learning, ReDiFMLearning
    from .models import Model, SAMHeadModel, load_cnn_model
    from .proposal import AllRegionsProposal, CropBox, RegionProposal
    from .structures import structure_probabilities
except ImportError:
    from config import SAM_CHECKPOINT, checkpoint_path
    from fusion import MaxForegroundFusion, RefinedRegionFusion
    from learning import Learning, ReDiFMLearning
    from models import Model, SAMHeadModel, load_cnn_model
    from proposal import AllRegionsProposal, CropBox, RegionProposal
    from structures import structure_probabilities


class Adapting:
    def __init__(
        self,
        coarse_model: Model,
        crop_model: Model,
        target_model: Model,
        region_proposal: RegionProposal | None = None,
        region_fusion: RefinedRegionFusion | None = None,
        learning: Learning | None = None,
    ) -> None:
        self.coarse_model = coarse_model.freeze()
        self.crop_model = crop_model.freeze()
        self.target_model = target_model
        self.region_proposal = AllRegionsProposal() if region_proposal is None else region_proposal
        self.region_fusion = MaxForegroundFusion() if region_fusion is None else region_fusion
        self.learning = ReDiFMLearning() if learning is None else learning
        self.learning.attach(self.target_model)

    @classmethod
    def from_checkpoints(
        cls,
        problem: str,
        source: str,
        device: torch.device,
        sam_checkpoint: Path = SAM_CHECKPOINT,
        region_proposal: RegionProposal | None = None,
        region_fusion: RefinedRegionFusion | None = None,
        learning: Learning | None = None,
    ) -> Adapting:
        coarse_model = SAMHeadModel.from_checkpoint(
            checkpoint_path(problem, source, "head"),
            device=device,
            sam_checkpoint=sam_checkpoint,
        )
        crop_model = load_cnn_model(checkpoint_path(problem, source, "crop"), device)
        target_model = load_cnn_model(checkpoint_path(problem, source, "target"), device)
        return cls(
            coarse_model=coarse_model,
            crop_model=crop_model,
            target_model=target_model,
            region_proposal=region_proposal,
            region_fusion=region_fusion,
            learning=learning,
        )

    @torch.no_grad()
    def _refine_regions(
        self, image: torch.Tensor, full: torch.Tensor, boxes: list[CropBox]
    ) -> list[tuple[CropBox, torch.Tensor]]:
        size = image.shape[-1]
        sy, sx = full.shape[-2] / size, full.shape[-1] / size
        regions: list[tuple[CropBox, torch.Tensor]] = []
        for box in boxes:
            crop = full[
                :,
                :,
                round(box.top * sy) : round((box.top + box.side) * sy),
                round(box.left * sx) : round((box.left + box.side) * sx),
            ]
            crop = F.interpolate(
                crop, size=(size, size), mode="bilinear", align_corners=False, antialias=True
            )
            classes = self.crop_model.predict_probabilities(crop)
            resized = F.interpolate(classes, size=(box.side, box.side), mode="bilinear", align_corners=False)
            regions.append((box, resized))
        return regions

    def adapt_and_predict(
        self, image: torch.Tensor, full_image: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        full = image if full_image is None else full_image
        coarse_classes = self.coarse_model.predict_probabilities(image)
        coarse_structures = structure_probabilities(coarse_classes)
        boxes = self.region_proposal.propose(coarse_structures)
        regions = self._refine_regions(image, full, boxes)
        crop_structures, crop_classes = self.region_fusion.fuse(regions, image.shape[-1])
        metrics = self.learning.step(self.target_model, image, coarse_classes, crop_structures)
        prediction = self.target_model.predict_probabilities(image)[0]
        teachers = {"head_alone": coarse_classes[0], "crop_teacher": crop_classes[0]}
        return prediction, {**metrics, "crops": len(boxes), "teachers": teachers}

    def __call__(
        self, image: torch.Tensor, full_image: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        return self.adapt_and_predict(image, full_image)
