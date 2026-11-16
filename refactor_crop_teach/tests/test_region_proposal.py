import torch
import torch.nn.functional as F

from helpers import blob_probabilities
from adaptation.region_proposal import PerRegionSquare, UnionSquare

S = 256


def full_image(height: int = S, width: int = S) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 3, height, width)


# --- PerRegionSquare ---------------------------------------------------------------------------


def test_per_region_square_gives_one_box_per_connected_region_in_raster_order():
    # The second blob starts on an earlier row, so ndimage.label numbers it first.
    sup1 = blob_probabilities(S, [(100, 20, 60, 50), (40, 150, 70, 80)])
    regions = PerRegionSquare().propose(sup1, full_image())
    assert [region.box for region in regions] == [(0, 96, 160, 160), (70, 0, 120, 120)]


def test_per_region_square_box_is_twice_the_extent_and_centered():
    sup1 = blob_probabilities(S, [(100, 90, 50, 60)])  # extent 60, center (124.5, 119.5)
    (region,) = PerRegionSquare().propose(sup1, full_image())
    assert region.box == (64, 60, 120, 120)
    assert region.crop.shape == (1, 3, S, S)


def test_per_region_square_box_is_at_least_96_and_stays_inside_the_image():
    sup1 = blob_probabilities(S, [(2, 250, 4, 4)])  # a small region in the top-right corner
    (region,) = PerRegionSquare().propose(sup1, full_image())
    assert region.box == (0, S - 96, 96, 96)


def test_per_region_square_box_is_never_larger_than_the_image():
    sup1 = blob_probabilities(S, [(10, 10, 200, 200)])
    (region,) = PerRegionSquare().propose(sup1, full_image())
    assert region.box == (0, 0, S, S)


def test_per_region_square_falls_back_to_the_most_confident_pixel():
    sup1 = blob_probabilities(S, [], value=0.9)          # foreground 0.1 everywhere: below 0.5
    sup1[0, :, 200, 30] = torch.tensor([0.6, 0.4])       # the most confident, still below 0.5
    (region,) = PerRegionSquare().propose(sup1, full_image())
    assert region.box == (152, 0, 96, 96)


def test_per_region_square_thresholds_the_first_structure():
    # Three classes: the first structure (disc) is rim + cup. Rim 0.3 + cup 0.3 passes 0.5.
    sup1 = torch.zeros(1, 3, S, S)
    sup1[:, 0] = 1.0
    sup1[0, :, 100:160, 100:160] = torch.tensor([0.4, 0.3, 0.3]).view(3, 1, 1)
    (region,) = PerRegionSquare().propose(sup1, full_image())
    assert region.box == (70, 70, 120, 120)


# --- UnionSquare -------------------------------------------------------------------------------


def test_union_square_gives_one_box_around_all_regions():
    sup1 = blob_probabilities(S, [(100, 100, 20, 20), (140, 150, 20, 10)])  # together: rows 100-159, columns 100-159
    (region,) = UnionSquare().propose(sup1, full_image())
    assert region.box == (70, 70, 120, 120)
    assert region.crop.shape == (1, 3, S, S)


def test_union_square_falls_back_to_the_most_confident_pixel():
    sup1 = blob_probabilities(S, [], value=0.9)
    sup1[0, :, 128, 128] = torch.tensor([0.6, 0.4])
    (region,) = UnionSquare().propose(sup1, full_image())
    assert region.box == (80, 80, 96, 96)


# --- Full resolution ---------------------------------------------------------------------------


def test_crop_is_cut_from_the_full_resolution_image_scaled_per_axis():
    """Non-square full image (2x in height, 3x in width): the cut rectangle maps back to the
    pixels under the box."""
    height, width = 2 * S, 3 * S
    full = torch.zeros(1, 3, height, width)
    full[:, :, 128:368, 180:540] = 1.0  # exactly the box (64, 60, 120, 120), scaled x2 and x3
    sup1 = blob_probabilities(S, [(100, 90, 50, 60)])
    for proposal in (PerRegionSquare(), UnionSquare()):
        (region,) = proposal.propose(sup1, full)
        assert region.box == (64, 60, 120, 120)
        assert proposal.full_rectangle(region.box, S, height, width) == (128, 368, 180, 540)
        assert torch.allclose(region.crop, torch.ones(1, 3, S, S), atol=1e-5)  # one pixel off would show zeros


def test_crop_keeps_full_resolution_detail():
    """The crop is the full-resolution cut resized to S x S, not an upscaled cut of the S x S image."""
    full = full_image(4 * S, 4 * S)
    sup1 = blob_probabilities(S, [(100, 100, 32, 32)])
    (region,) = PerRegionSquare().propose(sup1, full)
    top, left, side, _ = region.box
    assert side == 96
    cut = full[:, :, 4 * top: 4 * (top + side), 4 * left: 4 * (left + side)]
    expected = F.interpolate(cut, size=(S, S), mode="bilinear", align_corners=False, antialias=True)
    assert torch.equal(region.crop, expected)
