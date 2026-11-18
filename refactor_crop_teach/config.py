"""Paths, problems and the method's fixed settings.

Everything this folder reads or writes is inside it:
  data/<problem>/domain_<letter>/{Images,Masks}   prepared data          (fetch_assets.py)
  models/sam_vit_h_4b8939.pth                     SAM ViT-H weights      (fetch_assets.py)
  checkpoints/<problem>/source_<S>_<kind>.pth     trained source models  (fetch_assets.py)
  runs/results/, runs/mlruns.db                   outputs                (run.py)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"
CHECKPOINT_DIR = HERE / "checkpoints"
RUNS_DIR = HERE / os.environ.get("CT_RUNS", "runs")
RESULTS_DIR = RUNS_DIR / "results"
MLFLOW_URI = f"sqlite:///{RUNS_DIR / 'mlruns.db'}"

SAM_CHECKPOINT = MODEL_DIR / "sam_vit_h_4b8939.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


@dataclass(frozen=True)
class Problem:
    name: str
    domains: dict[str, str]          # letter -> dataset name
    counts: dict[str, int]           # images per domain, checked when loading
    structures: tuple[str, ...]      # nested: structure k is "class >= k + 1"
    image_size: int                  # every image is resized to this square size

    @property
    def num_classes(self) -> int:
        return len(self.structures) + 1

    def stream(self, source: str) -> list[str]:
        """The paper's target stream for a source domain: all other domains, alphabetical."""
        return [domain for domain in self.domains if domain != source]


PROBLEMS = {
    "odoc": Problem(
        "odoc",
        {"A": "RIM-ONE-r3", "B": "REFUGE", "C": "ORIGA", "D": "REFUGE-Validation-Test", "E": "Drishti-GS"},
        {"A": 159, "B": 400, "C": 650, "D": 800, "E": 101},
        ("disc", "cup"), 512,
    ),
    "prostate": Problem(
        "prostate",
        {"A": "BIDMC", "B": "BMC", "C": "HK", "D": "I2CVB", "E": "RUNMC", "F": "UCL"},
        {"A": 261, "B": 384, "C": 158, "D": 468, "E": 421, "F": 175},
        ("prostate",), 384,
    ),
    "polyp": Problem(
        "polyp",
        {"A": "BKAI-IGH-NeoPolyp", "B": "CVC-ClinicDB", "C": "ETIS-LaribPolypDB", "D": "Kvasir-SEG"},
        {"A": 1000, "B": 612, "C": 196, "D": 1000},
        ("polyp",), 352,
    ),
}

# Crop-Teach at test time.
LEARNING_RATE = 2e-5         # Adam, one update per image, batch-norm statistics frozen
PSEUDO_THRESHOLD = 0.5       # tau in the paper: foundation mask = probability > tau
ENTROPY_THRESHOLD = 0.5      # "high uncertainty" in the soft loss; not given by the paper
CROP_SCALE = 2.0             # crop side = 2 x the region's extent
MIN_CROP = 96                # smallest crop side, in pixels of the resized image
JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
ERASING = dict(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")


def checkpoint_path(problem: str, source: str, kind: str) -> Path:
    """kind: target (whole-image CNN), crop (native-res crop CNN) or head (SAM full-image head)."""
    return CHECKPOINT_DIR / problem / f"source_{source}_{kind}.pth"
