"""Repository locations shared by every method.

External model weights are fetched only by ``download/download.py``; the
methods load them from the paths below and never touch the network.
"""

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data/od_oc"
DOMAIN_A_DIR = DATA_DIR / "domain_A_RIM_ONE_r3/RIM-ONE-r3"
DRISHTI_ARCHIVE = DATA_DIR / "FunduSegmenter_Datasets.zip"
# Six-site prostate MRI benchmark (0140.pdf, Section 4.1), domains A-F.
PROSTATE_DIR = PROJECT_DIR / "data/prostate-6"
PROSTATE_ARCHIVE = PROSTATE_DIR / "raw/Processed_data_nii.zip"
RESULTS_DIR = PROJECT_DIR / "results"

# Model weights and checkpoints are kept on the external SSD when it is mounted,
# otherwise in the local checkpoints folder.  REDIFM_MODEL_DIR overrides both.
# MODEL_DIR holds shared foundation models; CHECKPOINT_DIR this project's files.
EXTERNAL_MODEL_DIR = Path("/run/media/lqtrung/1tb-ssd/Model")
LOCAL_CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
MODEL_DIR = Path(
    os.environ.get("REDIFM_MODEL_DIR")
    or (EXTERNAL_MODEL_DIR if EXTERNAL_MODEL_DIR.is_dir() else LOCAL_CHECKPOINT_DIR)
)
CHECKPOINT_DIR = MODEL_DIR / "ReDi-FM" if MODEL_DIR != LOCAL_CHECKPOINT_DIR else LOCAL_CHECKPOINT_DIR
DEFAULT_SOURCE_CHECKPOINT = CHECKPOINT_DIR / "domain_A_ResUNet34_last_20261017_215955.pth"
NNUNET_SOURCE_CHECKPOINT = CHECKPOINT_DIR / "domain_A_nnUNet2D_last_20261029_211218.pth"
# Trained with nnU-Net intensity augmentation; the earlier prostate_domain_A_ResUNet34.pth
# (OD/OC augmentation only) generalized poorly (26.49 Do. A against the paper's 51.02).
PROSTATE_RESUNET_SOURCE_CHECKPOINT = CHECKPOINT_DIR / "prostate_domain_A_ResUNet34_nnunet_aug.pth"
PROSTATE_NNUNET_SOURCE_CHECKPOINT = CHECKPOINT_DIR / "prostate_domain_A_nnUNet2D.pth"
PROSTATE_DINOV3_HEAD_CHECKPOINT = CHECKPOINT_DIR / "prostate_dinov3_vitl16_head_domain_A.pth"
PRETRAINED_RESUNET_CHECKPOINT = CHECKPOINT_DIR / "pretrain-ResUnet.pth"
DINOV3_ARCH = "vit_large_patch16_dinov3.lvd1689m"
SAM_CHECKPOINT = CHECKPOINT_DIR / "sam_vit_h_4b8939.pth"
BIOMEDCLIP_CHECKPOINT = CHECKPOINT_DIR / "medclip_samv2_biomedclip_pytorch_model.bin"
BIOMEDCLIP_DIR = MODEL_DIR / "BiomedCLIP-vit-bert-hf"
# Official MedDINOv3 (CT-3M pretrained ViT-B/16) and its source code, cloned
# from https://github.com/ricklisz/MedDINOv3 into the git-ignored third_party/.
MEDDINOV3_CHECKPOINT = MODEL_DIR / "MedDINOv3-ViTB-16-CT-3M/model.pth"
MEDDINOV3_REPO_DIR = PROJECT_DIR / "third_party/MedDINOv3"
MEDDINOV3_CODE_DIR = MEDDINOV3_REPO_DIR / "nnUNet/nnunetv2/training/nnUNetTrainer/dinov3"


def dinov3_weights(model_dir: Path, arch: str) -> Path:
    return model_dir / f"timm-{arch}" / "model.safetensors"


def require_file(path: Path, what: str) -> Path:
    """Return ``path`` if it exists, otherwise point at the download step."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            "Fetch it first: .venv/bin/python src/download/download.py --check"
        )
    return path
