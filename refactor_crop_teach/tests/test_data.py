import pytest
import torch
from conftest import TOY

from method.data import DomainSequenceLoader, ShuffledLoader


def names(loader):
    return [(sample.domain, sample.name) for sample in loader.order()]


def test_domain_sequence_loader_streams_domain_after_domain_in_the_given_order(toy_data):
    loader = DomainSequenceLoader(TOY, toy_data, ["C", "A"])
    assert [domain for domain, _ in names(loader)] == ["C"] * 4 + ["A"] * 3
    assert names(loader)[:2] == [("C", "0000_C"), ("C", "0001_C")]
    assert len(loader) == 7


def test_shuffled_loader_mixes_all_images_in_a_seeded_order(toy_data):
    sequence = names(DomainSequenceLoader(TOY, toy_data, ["A", "B", "C"]))
    shuffled = names(ShuffledLoader(TOY, toy_data, ["A", "B", "C"], seed=3))
    assert sorted(shuffled) == sorted(sequence) and shuffled != sequence
    assert shuffled == names(ShuffledLoader(TOY, toy_data, ["A", "B", "C"], seed=3))
    assert shuffled != names(ShuffledLoader(TOY, toy_data, ["A", "B", "C"], seed=4))


@pytest.mark.parametrize("loader_class", [DomainSequenceLoader, ShuffledLoader])
def test_loaders_yield_resized_normalized_images_and_full_size_copies(toy_data, loader_class):
    test = next(iter(loader_class(TOY, toy_data, ["B"])))
    assert test.image.shape == (1, 3, 32, 32) and test.full.shape == (1, 3, 40, 48)
    assert test.image.min() >= 0 and test.image.max() <= 1 and test.full.max() == 1
    assert test.mask.shape == (32, 32) and test.mask.dtype == torch.int64 and set(test.mask.unique().tolist()) == {0, 1}


@pytest.mark.parametrize("loader_class", [DomainSequenceLoader, ShuffledLoader])
def test_loaders_limit_images_per_domain(toy_data, loader_class):
    assert len(loader_class(TOY, toy_data, ["A", "C"], max_images=2)) == 4


def test_loader_refuses_an_incomplete_domain(toy_data):
    next((toy_data / "toy" / "domain_A" / "Images").iterdir()).unlink()
    with pytest.raises(RuntimeError, match="expected 3"):
        DomainSequenceLoader(TOY, toy_data, ["A"]).order()
