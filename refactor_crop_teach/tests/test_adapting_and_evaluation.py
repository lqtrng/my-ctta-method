from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from refactor_crop_teach import (
    Adapting,
    AlphabeticalDomainDataLoader,
    CustomDomainOrderDataLoader,
    EvaluateAdapting,
    EvaluateNonAdapting,
    IdentityAugmentation,
    LargestRegionProposal,
    MaxForegroundFusion,
    Model,
    Problem,
    ReDiFMLearning,
    evaluate_adapting,
    evaluate_non_adapting,
)


class ToyModel(Model):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self._num_classes = num_classes
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.out = nn.Conv2d(8, num_classes, kernel_size=1)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.bn(self.conv(images))))


@pytest.fixture()
def toy_problem_and_dir(tmp_path: Path) -> tuple[Problem, Path]:
    problem = Problem(
        name="toy",
        domains={"A": "SiteA", "B": "SiteB", "C": "SiteC"},
        counts={"A": 2, "B": 2, "C": 2},
        structures=("prostate",),
        image_size=32,
        network="resunet34",
    )
    rng = np.random.default_rng(0)
    for domain in problem.domains:
        img_dir = tmp_path / "toy" / f"domain_{domain}" / "Images"
        msk_dir = tmp_path / "toy" / f"domain_{domain}" / "Masks"
        img_dir.mkdir(parents=True)
        msk_dir.mkdir(parents=True)
        for index in range(2):
            stem = f"{index:04d}_{domain}{index}"
            pixels = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
            mask = np.zeros((48, 48), dtype=np.uint8)
            mask[12:36, 12:36] = 1
            Image.fromarray(pixels).save(img_dir / f"{stem}.png")
            Image.fromarray(mask).save(msk_dir / f"{stem}.png")
    return problem, tmp_path


def test_adapting_class_updates_target_and_returns_prediction() -> None:
    torch.manual_seed(0)
    coarse = ToyModel(2)
    crop = ToyModel(2)
    target = ToyModel(2)
    before = target.out.weight.detach().clone()

    adapting = Adapting(
        coarse_model=coarse,
        crop_model=crop,
        target_model=target,
        region_proposal=LargestRegionProposal(min_crop=16),
        region_fusion=MaxForegroundFusion(),
        learning=ReDiFMLearning(augmentation=IdentityAugmentation(), learning_rate=1e-2),
    )
    image = torch.rand(1, 3, 32, 32)
    full = torch.rand(1, 3, 48, 48)
    prediction, extras = adapting(image, full)

    assert prediction.shape == (2, 32, 32)
    assert torch.allclose(prediction.sum(dim=0), torch.ones(32, 32), atol=1e-5)
    assert not torch.equal(target.out.weight, before)
    assert "loss" in extras and "consensus" in extras and "crops" in extras
    assert set(extras["teachers"]) == {"head_alone", "crop_teacher"}


def test_evaluate_non_adapting(toy_problem_and_dir: tuple[Problem, Path], tmp_path: Path) -> None:
    assert evaluate_non_adapting is EvaluateNonAdapting
    problem, data_dir = toy_problem_and_dir
    loader = AlphabeticalDomainDataLoader(problem=problem, source="A", data_dir=data_dir)
    model = ToyModel(2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out_json = tmp_path / "results" / "non_adapting.json"

    evaluator = EvaluateNonAdapting(dataloader=loader, model=model)
    result = evaluator(output=out_json)

    assert out_json.is_file() and out_json.with_suffix(".csv").is_file()
    saved = json.loads(out_json.read_text())
    assert saved["do_dsc"] == pytest.approx(result["do_dsc"])
    assert result["stream"] == ["B", "C"]
    assert len(result["rounds"]) == 1
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_evaluate_adapting_orchestrates_dataloader_subclasses_and_rounds(
    toy_problem_and_dir: tuple[Problem, Path], tmp_path: Path
) -> None:
    assert evaluate_adapting is EvaluateAdapting
    problem, data_dir = toy_problem_and_dir
    loader = CustomDomainOrderDataLoader(
        problem=problem, source="A", domain_order=["C", "B"], data_dir=data_dir
    )
    adapting = Adapting(
        coarse_model=ToyModel(2),
        crop_model=ToyModel(2),
        target_model=ToyModel(2),
        region_proposal=LargestRegionProposal(min_crop=16),
        region_fusion=MaxForegroundFusion(),
        learning=ReDiFMLearning(augmentation=IdentityAugmentation(), learning_rate=1e-3),
    )
    out_json = tmp_path / "results" / "adapting.json"
    evaluator = EvaluateAdapting(dataloader=loader, adapting=adapting, rounds=2, seed=0)
    result = evaluator(output=out_json)

    assert out_json.is_file() and out_json.with_suffix(".csv").is_file()
    assert result["stream"] == ["C", "B"]
    assert len(result["rounds"]) == 2
    assert "average_dsc" in result and "degradation" in result
    assert "head_alone_mean_dsc" in result["rounds"][0]["mean"]
    assert "crop_teacher_mean_dsc" in result["rounds"][0]["mean"]


def test_no_forbidden_backbone_in_refactored_package() -> None:
    pkg_dir = Path(__file__).resolve().parents[1]
    forbidden = "dino" + "v3"
    for path in pkg_dir.rglob("*.py"):
        text = path.read_text().lower()
        assert forbidden not in text, f"Found {forbidden} reference in {path}"
    req_text = (pkg_dir / "requirements.txt").read_text().lower()
    assert forbidden not in req_text
    assert "timm" not in req_text
