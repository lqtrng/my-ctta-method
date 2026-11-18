import torch

import config
from method.augmentation import JitterErasingAugmentation, NoAugmentation


def test_no_augmentation_returns_the_image():
    image = torch.rand(1, 3, 16, 16)
    assert torch.equal(NoAugmentation()(image), image)


def test_jitter_erasing_returns_a_changed_copy_of_the_same_shape():
    image = torch.rand(1, 3, 16, 16)
    original = image.clone()
    augmented = JitterErasingAugmentation(config.JITTER, config.ERASING)(image)
    assert augmented.shape == image.shape
    assert not torch.equal(augmented, image)
    assert torch.equal(image, original)


def test_jitter_erasing_is_reproducible_under_a_seed():
    augmentation = JitterErasingAugmentation(config.JITTER, config.ERASING)
    image = torch.rand(1, 3, 16, 16)
    torch.manual_seed(5)
    first = augmentation(image)
    torch.manual_seed(5)
    assert torch.equal(augmentation(image), first)


def test_jitter_erasing_with_certain_erasing_erases_a_patch():
    augmentation = JitterErasingAugmentation(dict(brightness=0.0), dict(p=1.0, scale=(0.2, 0.2), value=0))
    augmented = augmentation(torch.ones(1, 3, 20, 20))
    assert (augmented == 0).all(dim=1).sum() > 0
