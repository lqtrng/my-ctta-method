import torch

from adaptation.augmentation import Augmentation, JitterErase


def image() -> torch.Tensor:
    return torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))


def test_jitter_erase_settings():
    augmentation = JitterErase()
    assert isinstance(augmentation, Augmentation)
    assert augmentation.JITTER == dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    assert augmentation.ERASING == dict(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")
    assert augmentation.erasing.p == 0.5 and augmentation.jitter.hue == (-0.05, 0.05)


def test_jitter_erase_returns_a_changed_copy_of_the_same_shape():
    original = image()
    before = original.clone()
    torch.manual_seed(0)
    augmented = JitterErase()(original)
    assert augmented.shape == original.shape and augmented.dtype == original.dtype
    assert torch.equal(original, before)          # the clean image is left as it was
    assert not torch.equal(augmented, original)


def test_jitter_erase_is_random_but_repeatable_under_a_seed():
    augmentation = JitterErase()
    torch.manual_seed(1)
    first = augmentation(image())
    second = augmentation(image())
    torch.manual_seed(1)
    again = augmentation(image())
    assert torch.equal(first, again) and not torch.equal(first, second)


def test_jitter_erase_erases_a_rectangle_about_half_the_time():
    """Erased pixels are random normal values; elsewhere color jitter keeps the image in [0, 1]."""
    augmentation = JitterErase()
    torch.manual_seed(0)
    erased = 0
    for _ in range(200):
        augmented = augmentation(image())
        outside = (augmented < 0) | (augmented > 1)
        if outside.any():
            erased += 1
            rows = outside.any(dim=1)[0].any(dim=1).nonzero().flatten()
            columns = outside.any(dim=1)[0].any(dim=0).nonzero().flatten()
            area = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
            assert area <= 0.2 * 64 * 64 + 64  # the erased rectangle is at most 20% of the image
    assert 70 <= erased <= 130
