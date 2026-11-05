#!/usr/bin/env bash
# Prepare a GPU machine: install packages, fetch the ImageNet weights, check data, weights and GPU.
#   ./setup.sh
# Expects (copied from the machine that ran prepare_data.py):
#   data/<problem>/domain_<letter>/{Images,Masks}                      (or CT_DATA)
#   models/sam_vit_h_4b8939.pth  (SAM ViT-H, the default backbone; or CT_SAM)
#   models/timm-vit_large_patch16_dinov3.lvd1689m/model.safetensors  (only with CT_BACKBONE=dinov3)
set -euo pipefail
cd "$(dirname "$0")"
P=${PYTHON:-python}

$P -m pip install -q -r requirements.txt

$P - <<'EOF'
import torch
from config import BACKBONE, PROBLEMS, SAM_CHECKPOINT, dinov3_weights
from data import domain_samples

assert torch.cuda.is_available(), "No CUDA GPU"
name = torch.cuda.get_device_name(0)
memory = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {name}, {memory:.1f} GB, bf16 supported: {torch.cuda.is_bf16_supported()}")
if not torch.cuda.is_bf16_supported():
    print("WARNING: no native bfloat16 (e.g. V100, T4, P100): DINOv3 will be very slow")
if BACKBONE == "sam_vit_h":
    assert SAM_CHECKPOINT.is_file(), f"Missing SAM ViT-H checkpoint: {SAM_CHECKPOINT}"
else:
    assert dinov3_weights().is_file(), f"Missing DINOv3 weights: {dinov3_weights()}"
print(f"Backbone under the head: {BACKBONE}")
for problem, spec in PROBLEMS.items():
    for domain in spec.domains:
        domain_samples(problem, domain)  # Raises if a domain is missing or incomplete.
print("Data: all 15 domains complete")

# Download once now, so no run needs the network later.
from torchvision.models import ResNet34_Weights, resnet34
resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
from transformers import SegformerForSemanticSegmentation
SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2)
print("ImageNet weights cached (ResNet-34, MiT-B0)")
EOF
echo "Setup done. Next: ./run_queue.sh smoke"
