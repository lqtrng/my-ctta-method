import numpy as np

from method.proposal import AllRegionsProposal, Box, LargestRegionProposal


def two_regions() -> np.ndarray:
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[10:20, 10:30] = 0.9   # small: 10 x 20
    probability[50:90, 40:70] = 0.8   # large: 40 x 30
    return probability


def test_all_regions_gives_one_square_crop_per_region_twice_its_extent():
    boxes = AllRegionsProposal(threshold=0.5, crop_scale=2.0, min_crop=8)(two_regions())
    assert boxes == [Box(top=0, left=0, side=40), Box(top=20, left=14, side=80)]


def test_largest_region_gives_one_crop_around_the_largest_region():
    boxes = LargestRegionProposal(threshold=0.5, crop_scale=2.0, min_crop=8)(two_regions())
    assert boxes == [Box(top=20, left=14, side=80)]


def test_crops_have_a_minimum_size_and_stay_inside_the_image():
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[95:98, 1:3] = 1.0
    for proposal in (AllRegionsProposal(min_crop=30), LargestRegionProposal(min_crop=30)):
        (box,) = proposal(probability)
        assert box.side == 30 and box.top == 70 and box.left == 0


def test_without_any_region_one_crop_is_centered_on_the_most_confident_pixel():
    probability = np.full((100, 100), 0.1, dtype=np.float32)
    probability[40, 60] = 0.3
    for proposal in (AllRegionsProposal(min_crop=20), LargestRegionProposal(min_crop=20)):
        assert proposal(probability) == [Box(top=30, left=50, side=20)]
