import pytest
import torch
from torch import nn

from method.architectures import SegmentationHead
from method.models import (SAM_CHANNELS, SAM_RESOLUTION, Model, ResUNet34Model, SAMHeadModel, SegformerB0Model,
                           load_model)


def check_probabilities(model: Model, size: int = 64) -> torch.Tensor:
    model.train()
    probabilities = model.predict(torch.rand(2, 3, size, size))
    assert probabilities.shape == (2, model.num_classes, size, size)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2, size, size), atol=1e-5)
    assert not model.training and not probabilities.requires_grad
    return probabilities


def test_resunet34_predicts_class_probabilities():
    check_probabilities(ResUNet34Model(3))


def test_segformer_b0_predicts_class_probabilities_at_input_size():
    check_probabilities(SegformerB0Model(2))


def test_load_model_restores_a_crop_teach_cnn_checkpoint(tmp_path):
    torch.manual_seed(0)
    original = ResUNet34Model(2)
    path = tmp_path / "source_A_target.pth"
    torch.save({"state_dict": original.network.state_dict(), "network": "resunet34", "num_classes": 2,
                "kind": "target"}, path)
    loaded = load_model(path, torch.device("cpu"))
    assert isinstance(loaded, ResUNet34Model)
    images = torch.rand(1, 3, 64, 64)
    assert torch.equal(loaded.predict(images), original.predict(images))


class FakeSAMEncoder(nn.Module):
    """Records its input; returns SAM's output shape (256 channels on a 64 x 64 grid)."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.inputs = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.inputs.append(images)
        pooled = nn.functional.adaptive_avg_pool2d(images.float(), 64).mean(1, keepdim=True)
        return pooled.expand(-1, SAM_CHANNELS, -1, -1).to(images.dtype)


def test_sam_head_model_feeds_sam_one_normalized_1024_image_at_a_time():
    encoder = FakeSAMEncoder()
    model = SAMHeadModel(encoder, SegmentationHead(SAM_CHANNELS, 8, 2), num_classes=2)
    images = torch.ones(2, 3, 96, 96)
    logits = model(images)
    assert logits.shape == (2, 2, 96, 96)
    assert [x.shape for x in encoder.inputs] == [(1, 3, SAM_RESOLUTION, SAM_RESOLUTION)] * 2
    assert encoder.inputs[0].dtype == torch.bfloat16
    expected_red = (255.0 - 123.675) / 58.395
    assert torch.allclose(encoder.inputs[0][0, 0].float(), torch.full((1024, 1024), expected_red), rtol=1e-2)


def test_sam_head_model_trains_only_the_head():
    model = SAMHeadModel(FakeSAMEncoder(), SegmentationHead(SAM_CHANNELS, 8, 2), num_classes=2)
    names = [name for name, _ in model.named_parameters()]
    assert names and all(name.startswith("head.") for name in names)


def test_load_model_rejects_heads_on_other_backbones(tmp_path):
    path = tmp_path / "head.pth"
    torch.save({"state_dict": {}, "backbone": "dinov3", "layers": [23], "hidden": 8, "num_classes": 2}, path)
    with pytest.raises(ValueError, match="SAM"):
        load_model(path, torch.device("cpu"))
