from __future__ import annotations

import pytest
import torch

from refactor_crop_teach import (
    CustomDomainOrderDataLoader,
    EvaluateNonAdapting,
    load_cnn_model,
)
from refactor_crop_teach.config import DATA_DIR, checkpoint_path


def test_prostate_source_a_non_adapting_matches_baseline() -> None:
    ckpt = checkpoint_path("prostate", "A", "target")
    prostate_dir = DATA_DIR / "prostate" / "domain_B"
    if not ckpt.is_file() or not prostate_dir.is_dir():
        pytest.skip("Prostate data or checkpoints not present")

    device = torch.device("cpu")
    loader = CustomDomainOrderDataLoader(
        problem="prostate",
        source="A",
        domain_order=["B"],
        device=device,
        max_images=2,
    )
    model = load_cnn_model(ckpt, device=device)
    evaluator = EvaluateNonAdapting(dataloader=loader, model=model)
    result = evaluator.evaluate()

    expected_mean = 100.0 * (0.3329764457532246 + 0.36208092500298705) / 2.0
    assert result["do_dsc"] == pytest.approx(expected_mean, abs=1e-4)
