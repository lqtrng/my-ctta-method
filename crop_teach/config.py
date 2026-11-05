"""Paths, problems and fixed settings. Everything else in this folder reads them from here.

The folder is self-contained: it imports nothing from the rest of the repository.

Locations (environment variables override them):
  CT_DATA     prepared data, one folder per problem (see prepare_data.py)
  CT_MODELS   DINOv3 weights: CT_MODELS/timm-<arch>/model.safetensors
  CT_RUNS     everything this folder writes: checkpoints/, results/, logs/, mlruns/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CT_DATA", HERE / "data"))
MODEL_DIR = Path(os.environ.get("CT_MODELS", HERE / "models"))
RUNS_DIR = Path(os.environ.get("CT_RUNS", HERE / "runs"))
CHECKPOINT_DIR = RUNS_DIR / "checkpoints"
RESULTS_DIR = RUNS_DIR / "results"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{RUNS_DIR / 'mlruns.db'}")

DINOV3_ARCH = "vit_large_patch16_dinov3.lvd1689m"
# The frozen backbone under the full-image head: "sam_vit_h" (SAM ViT-H image encoder, the default)
# or "dinov3" (DINOv3 ViT-L/16). SAM always takes a 1024 x 1024 input.
BACKBONE = os.environ.get("CT_BACKBONE", "sam_vit_h")
SAM_CHECKPOINT = Path(os.environ.get("CT_SAM", MODEL_DIR / "sam_vit_h_4b8939.pth"))


@dataclass(frozen=True)
class Problem:
    name: str
    domains: dict[str, str]          # letter -> dataset name, in stream order
    counts: dict[str, int]           # images per domain, checked when loading
    structures: tuple[str, ...]      # nested: structure k is "class >= k + 1"
    image_size: int                  # every image is resized to this square size
    network: str                     # source / target network, as in the ReDi-FM paper
    head_resolution: int             # DINOv3 input size for the full-image head (2x image_size)

    @property
    def num_classes(self) -> int:
        return len(self.structures) + 1

    def stream(self, source: str) -> list[str]:
        """Target domains for a source domain: all the others, in alphabetical order."""
        return [domain for domain in self.domains if domain != source]


PROBLEMS = {
    "odoc": Problem(
        "odoc",
        {"A": "RIM-ONE-r3", "B": "REFUGE", "C": "ORIGA", "D": "REFUGE-Validation-Test", "E": "Drishti-GS"},
        {"A": 159, "B": 400, "C": 650, "D": 800, "E": 101},
        ("disc", "cup"), 512, "resunet34", 1024,
    ),
    "prostate": Problem(
        "prostate",
        {"A": "BIDMC", "B": "BMC", "C": "HK", "D": "I2CVB", "E": "RUNMC", "F": "UCL"},
        {"A": 261, "B": 384, "C": 158, "D": 468, "E": 421, "F": 175},
        ("prostate",), 384, "resunet34", 768,
    ),
    "polyp": Problem(
        "polyp",
        {"A": "BKAI-IGH-NeoPolyp", "B": "CVC-ClinicDB", "C": "ETIS-LaribPolypDB", "D": "Kvasir-SEG"},
        {"A": 1000, "B": 612, "C": 196, "D": 1000},
        ("polyp",), 352, "segformer_b0", 704,
    ),
}


# Full-image head width: 256 = the small head (1.4M parameters on DINOv3, 1.25M on SAM; the
# default); 1024 = 20M parameters.
HEAD_HIDDEN = int(os.environ.get("CT_HEAD_HIDDEN", 256))

# Crop-Teach at test time (fixed; see README.md for where each comes from).
LEARNING_RATE = 2e-5         # Adam, one update per image, batch-norm statistics frozen
PSEUDO_THRESHOLD = 0.5       # tau in the paper
ENTROPY_THRESHOLD = 0.5      # "high uncertainty" in the soft loss; not given by the paper
CROP_SCALE = 2.0             # crop side = 2 x the region's extent
MIN_CROP = 96                # smallest crop side, in pixels of the resized image
JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
ERASING = dict(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")

# Training crops (crop models): jittered around the structure.
TRAIN_CROP_SCALE = (1.5, 3.0)
TRAIN_CROP_SHIFT = 0.25


def checkpoint_path(problem: str, source: str, kind: str) -> Path:
    """kind: target (whole image), crop (native-res crops) or head (DINOv3 full-image head)."""
    return CHECKPOINT_DIR / problem / f"source_{source}_{kind}.pth"


def dinov3_weights(arch: str = DINOV3_ARCH) -> Path:
    return MODEL_DIR / f"timm-{arch}" / "model.safetensors"


def head_resolution(problem: "Problem") -> int:
    """Input size of the backbone under the full-image head."""
    return 1024 if BACKBONE == "sam_vit_h" else problem.head_resolution
