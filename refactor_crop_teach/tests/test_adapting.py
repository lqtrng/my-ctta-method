import copy

import torch
from conftest import BrightModel, TinyModel

from method.adapting import Adapting
from method.augmentation import NoAugmentation
from method.fusion import MostForegroundFusion
from method.learning import ReDiFMLearning
from method.proposal import AllRegionsProposal, Box


class RecordingRefiner(BrightModel):
    """BrightModel that records the crops it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.crops = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.crops.append(images.clone())
        return super().forward(images)


def bright_square(size: int, top: int, left: int, side: int) -> torch.Tensor:
    image = torch.zeros(1, 3, size, size)
    image[..., top: top + side, left: left + side] = 1.0
    return image


def make_adapting(target=None, refiner=None) -> Adapting:
    return Adapting(target or TinyModel(), BrightModel(), refiner or RecordingRefiner(),
                    AllRegionsProposal(min_crop=8), MostForegroundFusion(), NoAugmentation(),
                    ReDiFMLearning(1e-2, 0.5, 0.5))


def test_adapting_updates_the_target_and_returns_its_prediction():
    torch.manual_seed(0)
    target = TinyModel()
    adapting = make_adapting(target)
    before = copy.deepcopy(target.state_dict())
    image = bright_square(32, 8, 8, 8)
    classes, statistics = adapting(image, image)
    assert classes.shape == (2, 32, 32)
    assert torch.allclose(classes, target.predict(image)[0])
    assert not torch.equal(before["conv.weight"], target.state_dict()["conv.weight"])
    assert statistics["crops"] == 1 and statistics["consensus"] > 0.9 and "loss" in statistics
    assert set(statistics["teachers"]) == {"head_alone", "crop_teacher"}


def test_adapting_leaves_the_teachers_frozen():
    foundation, refiner = TinyModel(), TinyModel()
    states = [copy.deepcopy(foundation.state_dict()), copy.deepcopy(refiner.state_dict())]
    adapting = Adapting(TinyModel(), foundation, refiner, AllRegionsProposal(min_crop=8), MostForegroundFusion(),
                        NoAugmentation(), ReDiFMLearning(1e-2, 0.5, 0.5))
    image = bright_square(32, 8, 8, 8)
    adapting(image, image)
    for model, state in zip((foundation, refiner), states):
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert not any(p.requires_grad for p in adapting.foundation.parameters())
    assert not any(p.requires_grad for p in adapting.refiner.parameters())


def test_supervision_comes_from_one_crop_per_foundation_region():
    adapting = make_adapting()
    image = bright_square(32, 2, 2, 4) + bright_square(32, 20, 20, 6)
    supervision, boxes = adapting.supervise(image, image)
    assert boxes == [Box(0, 0, 8), Box(16, 16, 12)]
    assert supervision.foundation.shape == supervision.refined.shape == (1, 2, 32, 32)
    assert supervision.refined[0, 1, 22, 22] > 0.99 and supervision.refined[0, :, 12, 12].sum() == 0
    assert supervision.inside.sum() == 64 + 144


def test_refine_cuts_the_crop_from_the_full_size_image():
    refiner = RecordingRefiner()
    adapting = make_adapting(refiner=refiner)
    full = torch.zeros(1, 3, 64, 64)
    full[..., 32:48, 16:32] = 1.0
    region = adapting.refine(full, Box(top=16, left=8, side=8), size=32)
    assert region.box == Box(16, 8, 8) and region.classes.shape == (1, 2, 8, 8)
    (crop,) = refiner.crops
    assert crop.shape == (1, 3, 32, 32) and torch.allclose(crop, torch.ones_like(crop))
