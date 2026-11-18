import copy

import torch

from adaptation.adapting import Adapting
from adaptation.fusion import MostForeground
from helpers import FixedModel, TinyModel, blob_probabilities
from adaptation.learning import CropTeachLearning
from adaptation.region_proposal import PerRegionSquare, UnionSquare

S = 128


def build(target: TinyModel | None = None, **strategies) -> Adapting:
    full_model = FixedModel(blob_probabilities(S, [(10, 10, 20, 20), (80, 70, 30, 30)]))
    crop_model = FixedModel(blob_probabilities(S, [(40, 40, 50, 50)], value=0.8))
    return Adapting(full_model, crop_model, target or TinyModel(2), **strategies)


def test_defaults_are_the_components_of_the_current_method():
    adapting = build()
    assert type(adapting.region_proposal) is PerRegionSquare
    assert type(adapting.fusion) is MostForeground
    assert type(adapting.learning) is CropTeachLearning


def test_one_step_runs_the_pipeline_and_returns_the_clean_prediction():
    torch.manual_seed(0)
    adapting = build()
    image, full = torch.rand(1, 3, S, S), torch.rand(1, 3, 2 * S, 3 * S)
    prediction = adapting(image, full)
    assert prediction.shape == (2, S, S) and not prediction.requires_grad
    assert torch.allclose(prediction.sum(0), torch.ones(S, S), atol=1e-5)
    # [2] the full-image model saw the stretched image; [4] the crop model saw one S x S crop per region.
    assert len(adapting.full_model.seen) == 1 and adapting.full_model.seen[0] is image
    assert [tuple(crop.shape) for crop in adapting.crop_model.seen] == [(1, 3, S, S)] * 2
    assert adapting.last == {"crops": 2, "loss": adapting.last["loss"], "consensus": adapting.last["consensus"]}
    assert adapting.last["loss"] > 0 and 0 <= adapting.last["consensus"] <= 1
    # [7] the prediction is the updated target model's, on the clean image, in eval mode.
    assert not adapting.target_model.training
    with torch.no_grad():
        assert torch.equal(prediction, adapting.target_model(image)[0])


def test_the_target_model_is_updated_every_step_and_never_reset():
    torch.manual_seed(0)
    adapting = build()
    states = [copy.deepcopy(adapting.target_model.state_dict())]
    for _ in range(2):
        adapting(torch.rand(1, 3, S, S), torch.rand(1, 3, S, S))
        states.append(copy.deepcopy(adapting.target_model.state_dict()))
    assert not torch.equal(states[0]["conv.weight"], states[1]["conv.weight"])
    assert not torch.equal(states[1]["conv.weight"], states[2]["conv.weight"])
    assert next(iter(adapting.learning.optimizer.state.values()))["step"] == 2


def test_the_two_teachers_are_frozen():
    adapting = Adapting(TinyModel(2), TinyModel(2), TinyModel(2))
    before = [copy.deepcopy(model.state_dict()) for model in (adapting.full_model, adapting.crop_model)]
    torch.manual_seed(0)
    adapting(torch.rand(1, 3, S, S), torch.rand(1, 3, S, S))
    for model, state in zip((adapting.full_model, adapting.crop_model), before):
        assert not model.training
        assert not any(parameter.requires_grad for parameter in model.parameters())
        assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())


def stream(seed: int) -> list[torch.Tensor]:
    """Predictions on a fixed three-image stream, from the same initial models."""
    generator = torch.Generator().manual_seed(7)
    images = [torch.rand(1, 3, S, S, generator=generator) for _ in range(3)]
    torch.manual_seed(99)                      # the same initial target model every time
    target = TinyModel(2)
    adapting = build(target, seed=seed)
    return [adapting(image, image) for image in images]


def test_the_seed_fixes_the_random_draws_of_the_stream():
    assert build().seed == 0 and build(seed=3).seed == 3
    first, again, other = stream(1), stream(1), stream(2)
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert not all(torch.equal(a, b) for a, b in zip(first, other))


def test_a_strategy_can_be_swapped():
    torch.manual_seed(0)
    adapting = build(region_proposal=UnionSquare())
    adapting(torch.rand(1, 3, S, S), torch.rand(1, 3, S, S))
    assert adapting.last["crops"] == 1
