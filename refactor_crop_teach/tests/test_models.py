from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from refactor_crop_teach.models import (
    CropRefineModel,
    Model,
    ResUNet34Model,
    SAMHeadModel,
    SegformerB0Model,
    SegmentationHead,
    build_model,
)
from refactor_crop_teach.proposal import LargestRegionProposal
from refactor_crop_teach.fusion import AverageFusion


class DummySAMEncoder(nn.Module):
    embed_dim = 16

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 16, kernel_size=16, stride=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_model_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        Model()


def test_resunet34_model() -> None:
    model = ResUNet34Model(num_classes=2)
    assert isinstance(model, Model)
    assert model.num_classes == 2
    images = torch.rand(2, 3, 64, 64)
    logits = model(images)
    assert logits.shape == (2, 2, 64, 64)
    probs = model.predict_probabilities(images)
    assert probs.shape == (2, 2, 64, 64)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, 64, 64), atol=1e-5)

    model.set_adaptation_train_mode(freeze_batch_norm=True)
    assert model.training
    assert not model.res.bn1.training


def test_segformer_b0_model() -> None:
    model = SegformerB0Model(num_classes=3)
    assert isinstance(model, Model)
    assert model.num_classes == 3
    images = torch.rand(1, 3, 64, 64)
    logits = model(images)
    assert logits.shape == (1, 3, 64, 64)
    probs = model.predict_probabilities(images)
    assert probs.shape == (1, 3, 64, 64)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 64, 64), atol=1e-5)


def test_sam_head_model_and_checkpoint_loading(tmp_path: Path) -> None:
    encoder = DummySAMEncoder()
    head = SegmentationHead(in_channels=16, hidden=8, num_classes=2)
    model = SAMHeadModel(encoder=encoder, head=head, resolution=64)
    assert isinstance(model, Model)
    assert model.num_classes == 2
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(p.requires_grad for p in model.parameters())

    images = torch.rand(1, 3, 32, 32)
    probs = model.predict_probabilities(images)
    assert probs.shape == (1, 2, 32, 32)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 32, 32), atol=1e-5)

    ckpt_path = tmp_path / "head.pth"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "backbone": "sam_vit_h",
            "layers": [23],
            "hidden": 8,
            "num_classes": 2,
            "resolution": 64,
        },
        ckpt_path,
    )
    loaded = SAMHeadModel.from_checkpoint(ckpt_path, device=torch.device("cpu"), encoder=DummySAMEncoder())
    assert isinstance(loaded, SAMHeadModel)

    bad_ckpt = tmp_path / "bad_head.pth"
    torch.save({"backbone": "other", "layers": [0], "hidden": 8, "num_classes": 2}, bad_ckpt)
    with pytest.raises(ValueError):
        SAMHeadModel.from_checkpoint(bad_ckpt, device=torch.device("cpu"), encoder=DummySAMEncoder())


def test_crop_refine_model() -> None:
    coarse = ResUNet34Model(num_classes=2)
    crop = ResUNet34Model(num_classes=2)
    refine = CropRefineModel(
        coarse_model=coarse,
        crop_model=crop,
        region_proposal=LargestRegionProposal(min_crop=16),
        region_fusion=AverageFusion(),
    )
    assert isinstance(refine, Model)
    assert refine.num_classes == 2
    image = torch.rand(1, 3, 64, 64)
    full = torch.rand(1, 3, 128, 128)
    probs = refine.predict_probabilities(image, full)
    assert probs.shape == (1, 2, 64, 64)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 64, 64), atol=1e-5)
    logits = refine(image)
    assert logits.shape == (1, 2, 64, 64)


def test_build_model_factory() -> None:
    assert isinstance(build_model("resunet34", 2), ResUNet34Model)
    assert isinstance(build_model("segformer_b0", 2), SegformerB0Model)
    with pytest.raises(ValueError):
        build_model("unknown", 2)
