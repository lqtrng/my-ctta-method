from __future__ import annotations

from .adapting import Adapting
from .augmentation import (
    Augmentation,
    ColorJitterAugmentation,
    ColorJitterRandomErasingAugmentation,
    GaussianNoiseAugmentation,
    IdentityAugmentation,
    RandomErasingAugmentation,
)
from .config import PROBLEMS, Problem
from .dataloader import (
    AlphabeticalDomainDataLoader,
    CustomDomainOrderDataLoader,
    DataLoader,
    ReversedStreamDataLoader,
    ShuffledStreamDataLoader,
    Sample,
    TestItem,
)
from .evaluation import (
    EvaluateAdapting,
    EvaluateNonAdapting,
    evaluate_adapting,
    evaluate_non_adapting,
)
from .fusion import (
    AverageFusion,
    ConfidenceWeightedFusion,
    MaxForegroundFusion,
    RefinedRegionFusion,
)
from .learning import (
    CropOnlyLearning,
    FusedDistillLearning,
    Learning,
    ReDiFMLearning,
)
from .models import (
    CropRefineModel,
    Model,
    ResUNet34Model,
    SAMHeadModel,
    SegformerB0Model,
    SegmentationHead,
    build_model,
    load_cnn_model,
)
from .proposal import (
    AllRegionsProposal,
    CropBox,
    LargestRegionProposal,
    MostConfidentRegionProposal,
    RegionProposal,
    WholeImageProposal,
)

__all__ = [
    "Adapting",
    "Augmentation",
    "IdentityAugmentation",
    "ColorJitterAugmentation",
    "RandomErasingAugmentation",
    "ColorJitterRandomErasingAugmentation",
    "GaussianNoiseAugmentation",
    "RegionProposal",
    "CropBox",
    "AllRegionsProposal",
    "LargestRegionProposal",
    "MostConfidentRegionProposal",
    "WholeImageProposal",
    "RefinedRegionFusion",
    "MaxForegroundFusion",
    "AverageFusion",
    "ConfidenceWeightedFusion",
    "Learning",
    "ReDiFMLearning",
    "CropOnlyLearning",
    "FusedDistillLearning",
    "Model",
    "ResUNet34Model",
    "SegformerB0Model",
    "SAMHeadModel",
    "CropRefineModel",
    "SegmentationHead",
    "build_model",
    "load_cnn_model",
    "DataLoader",
    "AlphabeticalDomainDataLoader",
    "CustomDomainOrderDataLoader",
    "ReversedStreamDataLoader",
    "ShuffledStreamDataLoader",
    "Sample",
    "TestItem",
    "EvaluateNonAdapting",
    "EvaluateAdapting",
    "evaluate_non_adapting",
    "evaluate_adapting",
    "Problem",
    "PROBLEMS",
]
