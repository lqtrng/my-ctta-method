from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from refactor_crop_teach.config import Problem
from refactor_crop_teach.dataloader import (
    AlphabeticalDomainDataLoader,
    CustomDomainOrderDataLoader,
    DataLoader,
    ReversedStreamDataLoader,
    ShuffledStreamDataLoader,
)


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> tuple[Problem, Path]:
    problem = Problem(
        name="toy",
        domains={"A": "SiteA", "B": "SiteB", "C": "SiteC", "D": "SiteD"},
        counts={"A": 2, "B": 2, "C": 2, "D": 2},
        structures=("lesion",),
        image_size=32,
        network="resunet34",
    )
    for domain in problem.domains:
        img_dir = tmp_path / "toy" / f"domain_{domain}" / "Images"
        msk_dir = tmp_path / "toy" / f"domain_{domain}" / "Masks"
        img_dir.mkdir(parents=True)
        msk_dir.mkdir(parents=True)
        for index in range(2):
            stem = f"{index:04d}_{domain}{index}"
            pixels = np.full((48, 48, 3), 20 * (index + 1), dtype=np.uint8)
            pixels[0, 0, 0] = 0
            mask = np.zeros((48, 48), dtype=np.uint8)
            mask[12:36, 12:36] = 1
            Image.fromarray(pixels).save(img_dir / f"{stem}.png")
            Image.fromarray(mask).save(msk_dir / f"{stem}.png")
    return problem, tmp_path


def test_dataloader_interface_is_abstract(synthetic_dataset: tuple[Problem, Path]) -> None:
    problem, data_dir = synthetic_dataset
    with pytest.raises(TypeError):
        DataLoader(problem=problem, source="A", data_dir=data_dir)


def test_alphabetical_domain_dataloader(synthetic_dataset: tuple[Problem, Path]) -> None:
    problem, data_dir = synthetic_dataset
    loader = AlphabeticalDomainDataLoader(problem=problem, source="A", data_dir=data_dir)
    assert isinstance(loader, DataLoader)
    assert loader.domains == ["B", "C", "D"]
    assert len(loader) == 6
    items = list(loader)
    assert [(it.domain, it.name) for it in items] == [
        ("B", "0000_B0"),
        ("B", "0001_B1"),
        ("C", "0000_C0"),
        ("C", "0001_C1"),
        ("D", "0000_D0"),
        ("D", "0001_D1"),
    ]
    assert items[0].image.shape == (1, 3, 32, 32)
    assert items[0].full_image.shape == (1, 3, 48, 48)
    assert items[0].target.shape == (32, 32)


def test_custom_domain_order_dataloader(synthetic_dataset: tuple[Problem, Path]) -> None:
    problem, data_dir = synthetic_dataset
    loader = CustomDomainOrderDataLoader(
        problem=problem,
        source="A",
        domain_order=["D", "B", "C"],
        data_dir=data_dir,
    )
    assert isinstance(loader, DataLoader)
    assert loader.domains == ["D", "B", "C"]
    assert [(s.domain, s.name) for s in loader.samples] == [
        ("D", "0000_D0"),
        ("D", "0001_D1"),
        ("B", "0000_B0"),
        ("B", "0001_B1"),
        ("C", "0000_C0"),
        ("C", "0001_C1"),
    ]
    with pytest.raises(ValueError):
        CustomDomainOrderDataLoader(
            problem=problem, source="A", domain_order=["A"], data_dir=data_dir
        )


def test_reversed_stream_dataloader(synthetic_dataset: tuple[Problem, Path]) -> None:
    problem, data_dir = synthetic_dataset
    loader = ReversedStreamDataLoader(problem=problem, source="A", data_dir=data_dir)
    assert isinstance(loader, DataLoader)
    assert loader.domains == ["D", "C", "B"]
    assert [(s.domain, s.name) for s in loader.samples] == [
        ("D", "0001_D1"),
        ("D", "0000_D0"),
        ("C", "0001_C1"),
        ("C", "0000_C0"),
        ("B", "0001_B1"),
        ("B", "0000_B0"),
    ]


def test_shuffled_stream_dataloader(synthetic_dataset: tuple[Problem, Path]) -> None:
    problem, data_dir = synthetic_dataset
    loader_a = ShuffledStreamDataLoader(
        problem=problem, source="A", shuffle_seed=11, data_dir=data_dir
    )
    loader_b = ShuffledStreamDataLoader(
        problem=problem, source="A", shuffle_seed=11, data_dir=data_dir
    )
    loader_c = ShuffledStreamDataLoader(
        problem=problem, source="A", shuffle_seed=99, data_dir=data_dir
    )
    assert isinstance(loader_a, DataLoader)
    assert loader_a.samples == loader_b.samples
    assert loader_a.samples != loader_c.samples
    assert sorted((s.domain, s.name) for s in loader_a.samples) == sorted(
        (s.domain, s.name)
        for s in AlphabeticalDomainDataLoader(problem=problem, source="A", data_dir=data_dir).samples
    )
