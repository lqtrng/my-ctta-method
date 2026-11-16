import copy

import pytest
import torch

from augmentation import Augmentation
from helpers import TinyModel, blob_probabilities
from learning import CropTeachLearning, Learning
from problem import structure_probabilities

S = 32


class NoAugmentation(Augmentation):
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return image


def supervisions(num_classes: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    sup1 = blob_probabilities(S, [(8, 8, 12, 12)], num_classes)
    sup2 = structure_probabilities(blob_probabilities(S, [(10, 8, 12, 12)], num_classes))
    return sup1, sup2


def test_crop_teach_learning_settings():
    learning = CropTeachLearning()
    assert isinstance(learning, Learning)
    assert (learning.LEARNING_RATE, learning.PSEUDO_THRESHOLD, learning.ENTROPY_THRESHOLD) == (2e-5, 0.5, 0.5)
    assert (learning.GAMMA, learning.MU, learning.SIGMA2) == (1.3, 0.5, 0.1)


@pytest.mark.parametrize("num_classes", [2, 3])
def test_step_updates_every_parameter_once_with_adam(num_classes):
    torch.manual_seed(0)
    model = TinyModel(num_classes)
    before = copy.deepcopy(model.state_dict())
    learning = CropTeachLearning()
    outcome = learning.step(model, torch.rand(1, 3, S, S), *supervisions(num_classes))
    assert set(outcome) == {"loss", "consensus"}
    assert outcome["loss"] > 0 and 0 <= outcome["consensus"] <= 1
    assert isinstance(learning.optimizer, torch.optim.Adam)
    assert learning.optimizer.param_groups[0]["lr"] == 2e-5
    for name, parameter in model.named_parameters():
        change = (parameter.detach() - before[name]).abs().max()
        assert 0 < change <= 2e-5 * 1.01, name  # Adam's first step moves each weight by at most lr


def test_step_freezes_the_batch_norm_statistics():
    torch.manual_seed(0)
    model = TinyModel(2)
    mean, variance = model.bn.running_mean.clone(), model.bn.running_var.clone()
    CropTeachLearning().step(model, torch.rand(1, 3, S, S), *supervisions())
    assert model.training and not model.bn.training
    assert torch.equal(model.bn.running_mean, mean) and torch.equal(model.bn.running_var, variance)
    assert model.bn.num_batches_tracked == 0


def test_step_augments_the_student_input_only():
    torch.manual_seed(0)
    augmentation = NoAugmentation()
    learning = CropTeachLearning(augmentation)
    model = TinyModel(2)
    learning.step(model, torch.rand(1, 3, S, S), *supervisions())
    assert augmentation.calls == 1


def test_step_keeps_one_optimizer_for_the_whole_stream():
    torch.manual_seed(0)
    learning, model = CropTeachLearning(), TinyModel(2)
    learning.step(model, torch.rand(1, 3, S, S), *supervisions())
    optimizer = learning.optimizer
    learning.step(model, torch.rand(1, 3, S, S), *supervisions())
    assert learning.optimizer is optimizer
    assert next(iter(optimizer.state.values()))["step"] == 2
    with pytest.raises(ValueError):
        learning.step(TinyModel(2), torch.rand(1, 3, S, S), *supervisions())


def test_confidence_weight_is_low_at_one_half_and_high_at_the_ends():
    weight = CropTeachLearning().confidence_weight(torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(weight, torch.tensor([0.9385, 0.0384, 0.9385]), atol=1e-3)


def test_consensus_is_the_dice_of_the_two_teachers_masks():
    learning = CropTeachLearning()
    sup1 = blob_probabilities(S, [(8, 8, 12, 12)])
    student = blob_probabilities(S, [(8, 8, 12, 12)], value=0.6)
    _, same = learning.loss(student, sup1, structure_probabilities(sup1))
    assert same == pytest.approx(1.0)
    _, shifted = learning.loss(student, sup1, structure_probabilities(blob_probabilities(S, [(8, 14, 12, 12)])))
    assert shifted == pytest.approx(0.5)  # 12 x 6 shared, of 12 x 12 each
    _, apart = learning.loss(student, sup1, structure_probabilities(blob_probabilities(S, [(20, 20, 8, 8)])))
    assert apart == pytest.approx(0.0, abs=1e-6)


def test_loss_is_lower_for_a_student_that_agrees_with_both_teachers():
    learning = CropTeachLearning()
    sup1 = blob_probabilities(S, [(8, 8, 12, 12)])
    sup2 = structure_probabilities(sup1)
    agreeing, _ = learning.loss(blob_probabilities(S, [(8, 8, 12, 12)], value=0.99), sup1, sup2)
    opposite, _ = learning.loss(blob_probabilities(S, [(20, 20, 8, 8)], value=0.99), sup1, sup2)
    assert agreeing < opposite


def test_the_sam_head_term_is_gated_by_the_consensus():
    """With no overlap between the teachers (r = 0), the SAM head's mask does not enter the hard loss."""
    learning = CropTeachLearning()
    sup1 = blob_probabilities(S, [(2, 2, 8, 8)])
    sup2 = structure_probabilities(blob_probabilities(S, [(20, 20, 8, 8)]))
    student = blob_probabilities(S, [(20, 20, 8, 8)], value=0.8)
    loss, consensus = learning.loss(student, sup1, sup2)
    prediction = structure_probabilities(student)
    crop_term = learning.hard_loss(prediction, sup2 > 0.5, learning.confidence_weight(sup2))
    head_term = learning.hard_loss(prediction, structure_probabilities(sup1) > 0.5,
                                   learning.confidence_weight(structure_probabilities(sup1)))
    soft_term = loss - crop_term
    assert consensus < 1e-6 and head_term > 1.0
    # r = 0: every soft-loss weight is 1, so the soft term is plain soft cross-entropy to the crop model.
    crop_classes = torch.cat((1 - sup2, sup2), dim=1)
    expected = -(student * crop_classes.log()).sum(1).mean()
    assert soft_term.item() == pytest.approx(expected.item(), rel=1e-4)
