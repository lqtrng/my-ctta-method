import csv
import json

import pytest
import torch
import torch.nn.functional as F

import core.problem as problem_module
from adaptation.adapting import Adapting
from data.data_loader import AlphabeticalStream, LongTermStream
from evaluation.evaluate import EvaluateAdapting, EvaluateNonAdapting, score
from helpers import FixedModel, TinyModel, blob_probabilities, tiny_problem

COUNTS = {"A": 2, "B": 3, "C": 2}
S = 128


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(problem_module, "DATA_DIR", tmp_path)
    return tiny_problem(tmp_path, COUNTS, S, (150, 200))


class RecordingAdapting:
    """Stands in for Adapting: records exactly what it is given."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        self.calls.append((args, kwargs))
        return torch.full((3, S, S), 1 / 3)


def test_score_is_the_dice_of_each_nested_structure():
    label = torch.zeros(S, S, dtype=torch.int64)
    label[20:60, 20:60] = 1      # rim
    label[30:50, 30:50] = 2      # cup
    perfect = F.one_hot(label, 3).permute(2, 0, 1).float()
    assert score(perfect, label, ("disc", "cup")) == {"disc_dsc": 1.0, "cup_dsc": 1.0, "mean_dsc": 1.0}
    no_cup = perfect.clone()
    no_cup[1] += no_cup[2]
    no_cup[2] = 0
    scores = score(no_cup, label, ("disc", "cup"))   # disc = rim + cup is still right
    assert scores["disc_dsc"] == 1.0 and scores["cup_dsc"] < 1e-6 and scores["mean_dsc"] == pytest.approx(0.5)
    half = torch.zeros(S, S, dtype=torch.int64)
    half[20:60, 20:40] = 1
    assert score(F.one_hot(half, 2).permute(2, 0, 1).float(), (label > 0).long(), ("fg",))["fg_dsc"] == pytest.approx(2 / 3)


def test_evaluate_adapting_never_passes_the_label_to_adapting(tiny):
    loader = AlphabeticalStream(tiny, "A", "cpu")
    adapting = RecordingAdapting()
    EvaluateAdapting(loader, adapting).run()
    items = list(loader)
    assert len(adapting.calls) == len(items) == 5
    for (args, kwargs), item in zip(adapting.calls, items):
        assert not kwargs and len(args) == 2                       # image and full, nothing else
        assert torch.equal(args[0], item.image) and torch.equal(args[1], item.full)
        assert all(argument.is_floating_point() for argument in args)   # the label is int64


def test_evaluate_adapting_scores_per_image_per_domain_and_per_round(tiny):
    torch.manual_seed(0)
    full_model = FixedModel(blob_probabilities(S, [(20, 20, 30, 30)], 3))
    adapting = Adapting(full_model, TinyModel(3), TinyModel(3), seed=4)
    result = EvaluateAdapting(LongTermStream(tiny, "B", "cpu"), adapting).run()
    assert len(result.rows) == 3 * 4
    assert [row["round"] for row in result.rows] == [1] * 4 + [2] * 4 + [3] * 4
    assert {"disc_dsc", "cup_dsc", "mean_dsc", "head_alone_mean_dsc", "crop_teacher_mean_dsc", "loss",
            "consensus", "crops", "domain", "image", "round"} == set(result.rows[0])
    assert [entry["round"] for entry in result.rounds] == [1, 2, 3]
    first = result.rounds[0]
    assert list(first["domains"]) == ["A", "C"] and first["domains"]["A"]["images"] == 2
    in_a = [row["mean_dsc"] for row in result.rows[:2]]
    assert first["domains"]["A"]["mean_dsc"] == pytest.approx(100 * sum(in_a) / 2)
    assert first["mean"]["mean_dsc"] == pytest.approx((first["domains"]["A"]["mean_dsc"] + first["domains"]["C"]["mean_dsc"]) / 2)
    assert result.do_dsc == first["mean"]["mean_dsc"]
    summary = result.to_dict()
    assert summary["stream"] == ["A", "C"] and summary["seed"] == 4 and "degradation" in summary


def test_evaluate_adapting_is_repeatable(tiny):
    def run() -> list[dict]:
        torch.manual_seed(123)  # the same initial models; Adapting then seeds its own draws
        full_model = FixedModel(blob_probabilities(S, [(20, 20, 30, 30)], 3))
        return EvaluateAdapting(AlphabeticalStream(tiny, "B", "cpu"), Adapting(full_model, TinyModel(3), TinyModel(3))).run().rows

    assert run() == run()


def test_evaluate_non_adapting_scores_one_frozen_model(tiny):
    model = TinyModel(3)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = EvaluateNonAdapting(AlphabeticalStream(tiny, "C", "cpu"), model).run()
    assert [(row["domain"], row["image"]) for row in result.rows] == [
        ("A", "0000_a"), ("A", "0001_a"), ("B", "0000_b"), ("B", "0001_b"), ("B", "0002_b")]
    assert set(result.rows[0]) == {"round", "domain", "image", "disc_dsc", "cup_dsc", "mean_dsc"}
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert not model.training and result.info["method"] == "source_only"
    assert len(result.rounds) == 1 and "degradation" not in result.to_dict()


def test_result_is_saved_as_json_and_csv(tiny, tmp_path):
    result = EvaluateNonAdapting(AlphabeticalStream(tiny, "C", "cpu"), TinyModel(3)).run()
    result.save(tmp_path / "out" / "source_only_sourceC")
    summary = json.loads((tmp_path / "out" / "source_only_sourceC.json").read_text())
    assert summary["problem"] == "tiny" and summary["source"] == "C" and summary["do_dsc"] == result.do_dsc
    assert summary["rounds"][0]["domains"]["B"]["images"] == 3
    with open(tmp_path / "out" / "source_only_sourceC.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 and float(rows[0]["mean_dsc"]) == pytest.approx(result.rows[0]["mean_dsc"])
