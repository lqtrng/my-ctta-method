from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "models"
RUNS_DIR = (HERE / os.environ.get("CT_RUNS", "runs")).resolve()
if not RUNS_DIR.is_relative_to(HERE):
    raise ValueError(f"CT_RUNS must name a folder inside {HERE}, got {RUNS_DIR}")
CHECKPOINT_DIR = RUNS_DIR / "checkpoints"
RESULTS_DIR = RUNS_DIR / "results"
MLFLOW_URI = f"sqlite:///{RUNS_DIR / 'mlruns.db'}"

os.environ["TORCH_HOME"] = str(MODEL_DIR / "torch")
os.environ["HF_HOME"] = str(MODEL_DIR / "huggingface")
if (MODEL_DIR / "huggingface" / "hub" / "models--nvidia--mit-b0").is_dir():
    os.environ["HF_HUB_OFFLINE"] = "1"

SAM_CHECKPOINT = MODEL_DIR / "sam_vit_h_4b8939.pth"
SAM_RESOLUTION = 1024
HEAD_HIDDEN = int(os.environ.get("CT_HEAD_HIDDEN", 256))


@dataclass(frozen=True)
class Problem:
    name: str
    domains: dict[str, str]
    counts: dict[str, int]
    structures: tuple[str, ...]
    image_size: int
    network: str

    @property
    def num_classes(self) -> int:
        return len(self.structures) + 1

    def stream(self, source: str) -> list[str]:
        return [domain for domain in self.domains if domain != source]


PROBLEMS = {
    "odoc": Problem(
        "odoc",
        {"A": "RIM-ONE-r3", "B": "REFUGE", "C": "ORIGA", "D": "REFUGE-Validation-Test", "E": "Drishti-GS"},
        {"A": 159, "B": 400, "C": 650, "D": 800, "E": 101},
        ("disc", "cup"),
        512,
        "resunet34",
    ),
    "prostate": Problem(
        "prostate",
        {"A": "BIDMC", "B": "BMC", "C": "HK", "D": "I2CVB", "E": "RUNMC", "F": "UCL"},
        {"A": 261, "B": 384, "C": 158, "D": 468, "E": 421, "F": 175},
        ("prostate",),
        384,
        "resunet34",
    ),
    "polyp": Problem(
        "polyp",
        {"A": "BKAI-IGH-NeoPolyp", "B": "CVC-ClinicDB", "C": "ETIS-LaribPolypDB", "D": "Kvasir-SEG"},
        {"A": 1000, "B": 612, "C": 196, "D": 1000},
        ("polyp",),
        352,
        "segformer_b0",
    ),
}

LEARNING_RATE = 2e-5
PSEUDO_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 0.5
CROP_SCALE = 2.0
MIN_CROP = 96
JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
ERASING = dict(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")


def checkpoint_path(problem: str, source: str, kind: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> Path:
    return checkpoint_dir / problem / f"source_{source}_{kind}.pth"
