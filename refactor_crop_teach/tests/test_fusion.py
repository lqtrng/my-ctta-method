import torch

from core.data_types import Region
from adaptation.fusion import MostForeground

S = 32
EMPTY = torch.empty(0)  # Fusion does not read the crop image.


def uniform(classes: list[float]) -> torch.Tensor:
    """A crop prediction with the same class vector at every pixel."""
    return torch.tensor(classes).view(1, -1, 1, 1).expand(1, len(classes), S, S).clone()


def test_most_foreground_pastes_the_resized_prediction_at_its_box():
    prediction = uniform([0.2, 0.8])
    prediction[:, :, :, S // 2:] = torch.tensor([0.7, 0.3]).view(1, 2, 1, 1)  # right half: background
    supervision_2, merged = MostForeground().fuse([prediction], [Region(EMPTY, (4, 8, 8, 16))], S)
    assert supervision_2.shape == (1, 1, S, S) and merged.shape == (1, 2, S, S)
    assert torch.allclose(merged[0, 1, 4:12, 8:15], torch.tensor(0.8))    # left half of the box
    assert torch.allclose(merged[0, 1, 4:12, 17:24], torch.tensor(0.3))   # right half of the box
    assert torch.allclose(supervision_2[0, 0, 4:12, 8:15], torch.tensor(0.8))


def test_most_foreground_is_background_outside_every_box():
    supervision_2, merged = MostForeground().fuse([uniform([0.2, 0.8])], [Region(EMPTY, (4, 8, 8, 16))], S)
    outside = torch.ones(S, S, dtype=torch.bool)
    outside[4:12, 8:24] = False
    assert torch.all(merged[0][:, outside] == 0)                          # all classes 0, not background = 1
    assert torch.all(supervision_2[0, 0][outside] == 1e-6)                # confident background


def test_most_foreground_clamps_supervision_2():
    supervision_2, _ = MostForeground().fuse([uniform([0.0, 1.0])], [Region(EMPTY, (0, 0, 16, 16))], S)
    assert supervision_2.max() == 1.0 - 1e-6 and supervision_2.min() == 1e-6


def test_most_foreground_keeps_the_most_foreground_crop_where_boxes_overlap():
    regions = [Region(EMPTY, (0, 0, 16, 16)), Region(EMPTY, (8, 8, 16, 16))]
    for first, second, winner in [(0.3, 0.9, 0.9), (0.9, 0.3, 0.9)]:
        predictions = [uniform([1 - first, first]), uniform([1 - second, second])]
        _, merged = MostForeground().fuse(predictions, regions, S)
        assert torch.allclose(merged[0, 1, 8:16, 8:16], torch.tensor(winner))  # the overlap
        assert torch.allclose(merged[0, 1, 0:8, 0:8], torch.tensor(first))     # only the first box
        assert torch.allclose(merged[0, 1, 16:24, 16:24], torch.tensor(second))  # only the second box


def test_most_foreground_ties_go_to_the_earlier_region():
    """Same total foreground (0.6), different class vectors: the comparison is a strict >."""
    regions = [Region(EMPTY, (0, 0, 16, 16)), Region(EMPTY, (8, 8, 16, 16))]
    earlier, later = [0.4, 0.5, 0.1], [0.4, 0.1, 0.5]
    assert sum(earlier[1:]) == sum(later[1:])
    _, merged = MostForeground().fuse([uniform(earlier), uniform(later)], regions, S)
    assert torch.allclose(merged[0, :, 12, 12], torch.tensor(earlier))


def test_most_foreground_swaps_the_whole_class_vector():
    regions = [Region(EMPTY, (0, 0, 16, 16)), Region(EMPTY, (8, 8, 16, 16))]
    earlier, later = [0.5, 0.45, 0.05], [0.3, 0.1, 0.6]  # later has more foreground, but less rim
    supervision_2, merged = MostForeground().fuse([uniform(earlier), uniform(later)], regions, S)
    assert torch.allclose(merged[0, :, 12, 12], torch.tensor(later))
    # Nested structures of the winner: disc = rim + cup, cup.
    assert torch.allclose(supervision_2[0, :, 12, 12], torch.tensor([0.7, 0.6]))


def test_most_foreground_handles_a_non_square_box():
    _, merged = MostForeground().fuse([uniform([0.1, 0.9])], [Region(EMPTY, (2, 3, 5, 20))], S)
    assert torch.count_nonzero(merged[0, 1]) == 5 * 20
