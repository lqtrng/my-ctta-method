import copy

import pytest
import torch
from conftest import TinyModel

from method.learning import ReDiFMLearning, Supervision, confidence_weight, redifm_loss


def square_classes(size: int = 16, foreground: float = 0.95) -> torch.Tensor:
    """(1, 2, size, size) class probabilities: a confident square in the middle."""
    probability = torch.full((1, 1, size, size), 1 - foreground)
    probability[..., 4:12, 4:12] = foreground
    return torch.cat([1 - probability, probability], dim=1)


def supervision(foundation: torch.Tensor, refined: torch.Tensor) -> Supervision:
    return Supervision(foundation, refined, torch.ones(1, 1, *refined.shape[-2:]))


def test_supervision_gives_background_outside_the_crops():
    refined = square_classes()
    inside = torch.zeros(1, 1, 16, 16)
    inside[..., :8, :] = 1
    structures = Supervision(refined, refined, inside).refined_structures
    assert structures.shape == (1, 1, 16, 16)
    assert structures[0, 0, 6, 6] == pytest.approx(0.95)
    assert structures[0, 0, 10, 6] == pytest.approx(1e-6)


def test_confidence_weight_is_low_at_one_half_and_high_at_the_extremes():
    weights = confidence_weight(torch.tensor([0.0, 0.5, 1.0]))
    assert weights[1] == pytest.approx(0.0384, abs=1e-3)
    assert weights[0] == pytest.approx(weights[2]) and weights[0] > 0.9


def test_consensus_is_the_dice_of_the_two_teacher_masks():
    teacher = square_classes()
    _, agree = redifm_loss(teacher, teacher, teacher[:, 1:], 0.5, 0.5)
    empty = torch.cat([torch.ones(1, 1, 16, 16), torch.zeros(1, 1, 16, 16)], dim=1)
    _, disagree = redifm_loss(teacher, empty, teacher[:, 1:], 0.5, 0.5)
    assert agree == pytest.approx(1.0) and disagree == pytest.approx(0.0, abs=1e-6)


def test_redifm_learning_takes_one_adam_step_with_batch_norm_statistics_frozen():
    torch.manual_seed(0)
    model = TinyModel()
    learning = ReDiFMLearning(learning_rate=1e-2, pseudo_threshold=0.5, entropy_threshold=0.5)
    learning.attach(model)
    before = copy.deepcopy(model.state_dict())
    statistics = learning.learn(model, torch.rand(1, 3, 16, 16), supervision(square_classes(), square_classes()))
    after = model.state_dict()
    assert set(statistics) == {"loss", "consensus"} and statistics["consensus"] == pytest.approx(1.0)
    assert not torch.equal(before["conv.weight"], after["conv.weight"])
    assert torch.equal(before["bn.running_mean"], after["bn.running_mean"])
    assert model.training and not model.bn.training


def test_redifm_learning_moves_the_target_towards_the_teachers():
    torch.manual_seed(0)
    model = TinyModel()
    learning = ReDiFMLearning(learning_rate=5e-2, pseudo_threshold=0.5, entropy_threshold=0.5)
    learning.attach(model)
    image = torch.zeros(1, 3, 16, 16)
    image[..., 4:12, 4:12] = 1.0
    teachers = supervision(square_classes(), square_classes())
    losses = [learning.learn(model, image, teachers)["loss"] for _ in range(30)]
    assert losses[-1] < losses[0]
    prediction = model.predict(image)[0].argmax(0)
    assert prediction[8, 8] == 1 and prediction[0, 0] == 0
