import pytest
import torch
from torch import nn

from models import Model, ResUNet34, SamHead


class FakeEncoder(nn.Module):
    """Stands in for SAM ViT-H: 1024 x 1024 image -> 256 channels on a 64 x 64 grid."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 16, stride=16)
        self.shapes: list[tuple[int, ...]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(images.shape))
        return self.conv(images)


def save_checkpoint(path, model: nn.Module, num_classes: int, **extra) -> None:
    torch.save({"state_dict": model.state_dict(), "num_classes": num_classes, **extra}, path)


# --- ResUNet34 ---------------------------------------------------------------------------------


@pytest.mark.parametrize("num_classes", [2, 3])
def test_resunet34_returns_class_probabilities_at_input_size(num_classes):
    torch.manual_seed(0)
    model = ResUNet34(num_classes).eval()
    assert isinstance(model, Model)
    with torch.no_grad():
        output = model(torch.rand(2, 3, 64, 64))
    assert output.shape == (2, num_classes, 64, 64)
    assert torch.allclose(output.sum(1), torch.ones(2, 64, 64), atol=1e-5)
    assert output.min() >= 0


def test_resunet34_load_restores_the_trained_weights(tmp_path):
    torch.manual_seed(0)
    trained = ResUNet34(3).eval()
    save_checkpoint(tmp_path / "model.pth", trained, 3)
    loaded = ResUNet34(3).load(tmp_path / "model.pth").eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(loaded(image), trained(image))


def test_resunet34_load_rejects_a_wrong_or_missing_checkpoint(tmp_path):
    save_checkpoint(tmp_path / "model.pth", ResUNet34(2), 2)
    with pytest.raises(ValueError):
        ResUNet34(3).load(tmp_path / "model.pth")
    with pytest.raises(FileNotFoundError):
        ResUNet34(3).load(tmp_path / "missing.pth")


def test_resunet34_is_trainable():
    model = ResUNet34(2)
    model(torch.rand(1, 3, 64, 64))[:, 1].mean().backward()
    assert model.seg_head.weight.grad is not None and model.res.conv1.weight.grad is not None


# --- SamHead (with a small fake encoder in place of SAM) ---------------------------------------


def test_sam_head_resizes_to_1024_and_returns_probabilities_at_input_size():
    torch.manual_seed(0)
    encoder = FakeEncoder()
    model = SamHead(3, encoder)
    assert isinstance(model, Model)
    output = model(torch.rand(2, 3, 96, 96))
    assert output.shape == (2, 3, 96, 96)
    assert torch.allclose(output.sum(1), torch.ones(2, 96, 96), atol=1e-5)
    assert encoder.shapes == [(1, 3, 1024, 1024)] * 2  # one image at a time


def test_sam_head_is_frozen_and_stays_in_eval_mode():
    model = SamHead(2, FakeEncoder())
    model.train()
    assert not model.training and not model.head.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in model._encoder[0].parameters())
    assert not model(torch.rand(1, 3, 64, 64)).requires_grad


def test_sam_head_state_holds_only_the_head():
    model = SamHead(2, FakeEncoder())
    assert all(key.startswith("head.") for key in model.state_dict())


def test_sam_head_uses_the_encoder_dtype():
    encoder = FakeEncoder().double()
    dtypes = []
    encoder.register_forward_pre_hook(lambda module, args: dtypes.append(args[0].dtype))
    output = SamHead(2, encoder)(torch.rand(1, 3, 64, 64))
    assert dtypes == [torch.float64] and output.dtype == torch.float32


def test_sam_head_load_restores_the_trained_head(tmp_path):
    torch.manual_seed(0)
    encoder = FakeEncoder()
    trained = SamHead(3, encoder)
    save_checkpoint(tmp_path / "head.pth", trained.head, 3, backbone="sam_vit_h", hidden=256, resolution=1024)
    loaded = SamHead(3, encoder).load(tmp_path / "head.pth")
    image = torch.rand(1, 3, 64, 64)
    assert torch.equal(loaded(image), trained(image))


def test_sam_head_load_rejects_a_head_of_another_backbone(tmp_path):
    model = SamHead(3, FakeEncoder())
    save_checkpoint(tmp_path / "head.pth", model.head, 3, backbone="dinov3", hidden=256, resolution=1024)
    with pytest.raises(ValueError):
        model.load(tmp_path / "head.pth")
