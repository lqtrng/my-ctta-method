import numpy as np
import pytest
import torch
from PIL import Image

import core.problem as problem_module
from data.data_loader import AlphabeticalStream, DataLoader, LongTermStream
from core.data_types import TestImage
from helpers import tiny_problem

COUNTS = {"A": 2, "B": 3, "C": 1, "D": 2}
S, SHAPE = 32, (40, 60)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(problem_module, "DATA_DIR", tmp_path)
    return tiny_problem(tmp_path, COUNTS, S, SHAPE)


def order(loader: DataLoader) -> list[tuple[int, str, str]]:
    return [(item.round, item.domain, item.name) for item in loader]


ONE_PASS_FROM_B = [("A", "0000_a"), ("A", "0001_a"), ("C", "0000_c"), ("D", "0000_d"), ("D", "0001_d")]


# --- AlphabeticalStream ------------------------------------------------------------------------


def test_alphabetical_stream_is_one_pass_over_every_domain_but_the_source(tiny):
    loader = AlphabeticalStream(tiny, "B", "cpu")
    assert isinstance(loader, DataLoader) and loader.rounds == 1
    assert loader.domains() == ["A", "C", "D"]
    assert order(loader) == [(1, *item) for item in ONE_PASS_FROM_B]
    assert len(loader) == 5


def test_alphabetical_stream_prepares_each_image(tiny):
    item = next(iter(AlphabeticalStream(tiny, "B", "cpu")))
    assert isinstance(item, TestImage)
    assert item.image.shape == (1, 3, S, S) and item.image.dtype == torch.float32
    assert item.full.shape == (1, 3, *SHAPE)                       # original resolution, not resized
    assert item.label.shape == (S, S) and item.label.dtype == torch.int64
    for tensor in (item.image, item.full):                         # each min-max normalized to [0, 1]
        assert tensor.min() == 0 and tensor.max() == 1


def test_alphabetical_stream_stretches_image_and_label_the_same_way(tiny):
    """Not padded: the image is resized bilinearly and the label with nearest-neighbor, both to S x S."""
    folder = tiny.domain_dir("A")
    item = next(iter(AlphabeticalStream(tiny, "B", "cpu")))
    mask = Image.open(folder / "Masks" / "0000_a.png")
    expected = np.asarray(mask.resize((S, S), Image.Resampling.NEAREST), dtype=np.int64)
    assert np.array_equal(item.label.numpy(), expected)
    assert set(item.label.unique().tolist()) <= {0, 1, 2}
    image = Image.open(folder / "Images" / "0000_a.png").resize((S, S), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    array = (array - array.min()) / (array.max() - array.min())
    assert np.allclose(item.image[0].numpy(), array)


def test_alphabetical_stream_checks_the_number_of_images(tiny):
    (tiny.domain_dir("C") / "Images" / "0000_c.png").unlink()
    with pytest.raises(RuntimeError):
        list(AlphabeticalStream(tiny, "B", "cpu"))


def test_alphabetical_stream_rejects_an_unknown_source(tiny):
    with pytest.raises(ValueError):
        AlphabeticalStream(tiny, "Z", "cpu")


# --- LongTermStream ----------------------------------------------------------------------------


def test_long_term_stream_repeats_the_same_order_three_times(tiny):
    loader = LongTermStream(tiny, "B", "cpu")
    assert isinstance(loader, DataLoader) and loader.rounds == 3
    assert loader.domains() == ["A", "C", "D"]
    assert order(loader) == [(round_index, *item) for round_index in (1, 2, 3) for item in ONE_PASS_FROM_B]
    assert len(loader) == 15


def test_long_term_stream_yields_the_same_images_every_round(tiny):
    items = list(LongTermStream(tiny, "A", "cpu"))
    per_round = len(items) // 3
    for first, later in zip(items[:per_round], items[2 * per_round:]):
        assert torch.equal(first.image, later.image) and torch.equal(first.label, later.label)


# --- The hierarchy -----------------------------------------------------------------------------


def test_each_stream_extends_data_loader_directly():
    assert AlphabeticalStream.__bases__ == (DataLoader,) and LongTermStream.__bases__ == (DataLoader,)
