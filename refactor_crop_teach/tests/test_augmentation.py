from __future__ import annotations

import pytest
import torch

from refactor_crop_teach.augmentation import (
    Augmentation,
    ColorJitterAugmentation,
    ColorJitterRandomErasingAugmentation,
    GaussianNoiseAugmentation,
    IdentityAugmentation,
    RandomErasingAugmentation,
    build_augmentation,
)


def test_augmentation_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        Augmentation()


def test_identity_augmentation() -> None:
    strategy = IdentityAugmentation()
    assert isinstance(strategy, Augmentation)
    image = torch.rand(2, 3, 32, 32)
    out = strategy(image)
    assert out.shape == image.shape
    assert torch.equal(out, image)
    assert out.data_ptr() != image.data_ptr()


def test_color_jitter_augmentation() -> None:
    torch.manual_seed(0)
    strategy = ColorJitterAugmentation(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
    assert isinstance(strategy, Augmentation)
    single = torch.full((1, 3, 32, 32), 0.5)
    batch = torch.full((2, 3, 32, 32), 0.5)
    out_single = strategy(single)
    out_batch = strategy(batch)
    assert out_single.shape == single.shape
    assert out_batch.shape == batch.shape
    assert (out_single >= 0.0).all() and (out_single <= 1.0).all()


def test_random_erasing_augmentation() -> None:
    torch.manual_seed(0)
    strategy = RandomErasingAugmentation(p=1.0, scale=(0.1, 0.2), ratio=(1.0, 1.0), value=0.0)
    assert isinstance(strategy, Augmentation)
    image = torch.ones(1, 3, 64, 64)
    out = strategy(image)
    assert out.shape == image.shape
    assert (out == 0.0).any()
    assert (out == 1.0).any()


def test_color_jitter_random_erasing_augmentation() -> None:
    torch.manual_seed(42)
    strategy = ColorJitterRandomErasingAugmentation()
    assert isinstance(strategy, Augmentation)
    image = torch.rand(2, 3, 64, 64)
    out = strategy(image)
    assert out.shape == image.shape
    assert not torch.equal(out, image)


def test_gaussian_noise_augmentation() -> None:
    torch.manual_seed(7)
    strategy = GaussianNoiseAugmentation(std=0.1)
    assert isinstance(strategy, Augmentation)
    image = torch.full((1, 3, 32, 32), 0.5)
    out = strategy(image)
    assert out.shape == image.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()
    assert not torch.equal(out, image)


def test_build_augmentation_registry() -> None:
    for name in ("identity", "color_jitter", "random_erasing", "color_jitter_erasing", "gaussian_noise"):
        assert isinstance(build_augmentation(name), Augmentation)
    with pytest.raises(ValueError):
        build_augmentation("unknown")
