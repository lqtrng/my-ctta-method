import pytest
import torch
from conftest import TOY, BrightModel, TinyModel

from method.adapting import Adapting
from method.augmentation import NoAugmentation
from method.data import DomainSequenceLoader, ShuffledLoader
from method.evaluation import EvaluateAdapting, EvaluateNonAdapting
from method.fusion import MostForegroundFusion
from method.learning import ReDiFMLearning
from method.proposal import AllRegionsProposal


def test_evaluate_non_adapting_scores_every_image_per_domain_and_round(toy_data):
    logged = []
    loader = DomainSequenceLoader(TOY, toy_data, ["B", "A"])
    result = EvaluateNonAdapting(BrightModel(), loader, TOY, rounds=2, log=lambda m, step: logged.append(step)).run()
    assert len(result.rows) == 2 * 5 and logged == [1, 2]
    first = result.rounds[0]
    assert list(first["domains"]) == ["B", "A"]
    assert first["domains"]["B"]["images"] == 2 and first["domains"]["A"]["images"] == 3
    assert first["mean"]["mean_dsc"] > 90  # The bright square is the mask.
    assert set(result.rows[0]) == {"round", "domain", "image", "fg_dsc", "mean_dsc"}
    assert (result.rows[0]["domain"], result.rows[0]["image"], result.rows[-1]["round"]) == ("B", "0000_B", 2)


def test_evaluate_non_adapting_does_not_change_the_model(toy_data):
    model = TinyModel()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    EvaluateNonAdapting(model, DomainSequenceLoader(TOY, toy_data, ["A"]), TOY).run()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())


def test_evaluate_adapting_scores_the_target_and_both_teachers(toy_data):
    torch.manual_seed(0)
    target = TinyModel()
    adapting = Adapting(target, BrightModel(), BrightModel(), AllRegionsProposal(min_crop=8), MostForegroundFusion(),
                        NoAugmentation(), ReDiFMLearning(1e-2, 0.5, 0.5))
    loader = ShuffledLoader(TOY, toy_data, ["A", "C"], seed=0)
    result = EvaluateAdapting(adapting, loader, TOY).run()
    assert len(result.rows) == 7
    row = result.rows[0]
    assert {"mean_dsc", "head_alone_mean_dsc", "crop_teacher_mean_dsc", "loss", "consensus", "crops"} <= set(row)
    assert result.rounds[0]["mean"]["head_alone_mean_dsc"] == pytest.approx(
        EvaluateNonAdapting(BrightModel(), loader, TOY).run().rounds[0]["mean"]["mean_dsc"])
    assert result.ms_per_image > 0
