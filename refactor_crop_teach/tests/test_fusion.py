import pytest
import torch

from method.fusion import MeanFusion, MostForegroundFusion, RefinedRegion
from method.proposal import Box


def region(top: int, left: int, side: int, foreground: float) -> RefinedRegion:
    classes = torch.tensor([1.0 - foreground, foreground]).view(1, 2, 1, 1).expand(1, 2, side, side).clone()
    return RefinedRegion(Box(top, left, side), classes)


@pytest.mark.parametrize("fusion", [MostForegroundFusion(), MeanFusion()])
def test_a_single_region_is_pasted_with_zeros_outside(fusion):
    fused = fusion([region(2, 3, 4, 0.7)], size=10)
    assert fused.classes.shape == (1, 2, 10, 10) and fused.inside.shape == (1, 1, 10, 10)
    assert torch.allclose(fused.classes[0, :, 2:6, 3:7], torch.tensor([0.3, 0.7]).view(2, 1, 1).expand(2, 4, 4))
    assert fused.classes.sum() == pytest.approx(16.0)
    assert fused.inside.sum() == 16 and fused.inside[0, 0, 2:6, 3:7].all()


@pytest.mark.parametrize("fusion", [MostForegroundFusion(), MeanFusion()])
def test_inside_covers_every_region(fusion):
    fused = fusion([region(0, 0, 4, 0.2), region(2, 2, 4, 0.9)], size=8)
    assert fused.inside.sum() == 16 + 16 - 4


def test_most_foreground_fusion_keeps_the_most_foreground_region_where_they_overlap():
    for regions in ([region(0, 0, 4, 0.2), region(2, 2, 4, 0.9)], [region(2, 2, 4, 0.9), region(0, 0, 4, 0.2)]):
        fused = MostForegroundFusion()(regions, size=8)
        assert fused.classes[0, 1, 3, 3] == pytest.approx(0.9)   # overlap
        assert fused.classes[0, 1, 0, 0] == pytest.approx(0.2)   # first region only
        assert fused.classes[0, 1, 5, 5] == pytest.approx(0.9)   # second region only


def test_mean_fusion_averages_the_regions_where_they_overlap():
    fused = MeanFusion()([region(0, 0, 4, 0.2), region(2, 2, 4, 0.9)], size=8)
    assert fused.classes[0, 1, 3, 3] == pytest.approx(0.55)
    assert fused.classes[0, 0, 3, 3] == pytest.approx(0.45)
    assert fused.classes[0, 1, 0, 0] == pytest.approx(0.2)


@pytest.mark.parametrize("fusion", [MostForegroundFusion(), MeanFusion()])
def test_fusion_needs_a_region(fusion):
    with pytest.raises(ValueError):
        fusion([], size=8)
