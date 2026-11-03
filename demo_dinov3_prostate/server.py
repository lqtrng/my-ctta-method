#!/usr/bin/env python3
"""Pick a prostate slice in the browser and see three models segment it side by side.

    python demo_dinov3_prostate/server.py [--port 8000] [--device cuda]

Then open http://127.0.0.1:8000.  All three are frozen, nothing adapts:

* DINOv3 only: DINOv3 on the whole slice with the head trained on whole domain-A
  slices.  Train that head once with
  python src/training/train_dinov3_head.py --task prostate --full-image --crop-resolution 768
* nnU-Net only: the source model trained on domain A.
* nnU-Net + DINOv3: nnU-Net's mask places a crop, DINOv3 encodes the crop and the
  head trained on domain-A crops segments it (the ReDi-FM + DINOv3 teacher).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.dinov3_single import FullImageSegmenter  # noqa: E402
from common.data import PROSTATE_SITES  # noqa: E402
from common.models import build_network  # noqa: E402
from common.paths import PROSTATE_DINOV3_FULL_IMAGE_HEAD_CHECKPOINT, require_file  # noqa: E402
from common.protocol import image_to_tensor, prepare_sample  # noqa: E402
from common.redifm import structure_probabilities  # noqa: E402
from common.tasks import TASKS  # noqa: E402
from methods.redifm_dinov3 import DinoV3Segmenter, add_arguments, resolve_checkpoints  # noqa: E402


PAGE = Path(__file__).with_name("demo_dinov3_prostate.html")


def png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def mask_data_url(mask: np.ndarray) -> str:
    return png_data_url(Image.fromarray(mask.astype(np.uint8) * 255, mode="L"))


class Demo:
    """The dataset and the three models, loaded once; one slice runs at a time."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.task = TASKS[args.task]
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.domains = self.task.build_domains(list(PROSTATE_SITES), None)
        self.lock = threading.Lock()
        resolve_checkpoints(args, self.task)
        checkpoint = require_file(args.source_checkpoint, "nnU-Net source checkpoint")
        self.nnunet = build_network(args.arch, self.task.num_classes, self.task.image_size)
        self.nnunet.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        self.nnunet.requires_grad_(False).to(self.device).eval()
        self.crop_segmenter = DinoV3Segmenter(args, self.device)
        self.full_image_segmenter = FullImageSegmenter(
            self.crop_segmenter.model,
            require_file(PROSTATE_DINOV3_FULL_IMAGE_HEAD_CHECKPOINT, "Full-image DINOv3 head"),
            args.dinov3_arch,
            self.device,
        )

    def catalog(self) -> dict:
        return {
            "device": str(self.device),
            "domains": [
                {
                    "domain": domain,
                    "dataset": PROSTATE_SITES[domain],
                    "source": domain == "A",
                    "images": [sample.identifier for sample in samples],
                }
                for domain, samples in self.domains.items()
            ],
        }

    def segment(self, domain: str, index: int) -> dict:
        sample = self.domains[domain][index]
        image, target = prepare_sample(sample, self.task.image_size)
        with self.lock, torch.inference_mode():
            image_tensor = image_to_tensor(image, self.device)
            logits = self.nnunet(image_tensor)
            structures = structure_probabilities(F.softmax(logits, dim=1))
            combined, _, _ = self.crop_segmenter.supervision(
                image, image_tensor, structures, self.args.pseudo_threshold
            )
            soft = structures[0, 0].cpu().numpy()
            top, left, side = self.crop_segmenter.crop_box(soft > self.args.pseudo_threshold, soft)
            dinov3 = self.full_image_segmenter(image_tensor).argmax(dim=1)[0]
            nnunet = logits.argmax(dim=1)[0]
        return {
            "domain": domain,
            "index": index,
            "identifier": sample.identifier,
            "size": self.task.image_size,
            "image": png_data_url(image),
            "ground_truth": mask_data_url(target.numpy() > 0),
            "dinov3": mask_data_url(dinov3.cpu().numpy() > 0),
            "nnunet": mask_data_url(nnunet.cpu().numpy() > 0),
            "combined": mask_data_url(combined[0, 0].cpu().numpy()),
            "crop": {"top": top, "left": left, "side": side},
        }


def make_handler(demo: Demo) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, content_type: str, body: bytes) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The page moved on to another slice.

        def respond_json(self, status: int, payload: dict) -> None:
            self.respond(status, "application/json", json.dumps(payload).encode("utf-8"))

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self.respond(200, "text/html; charset=utf-8", PAGE.read_bytes())
            elif url.path == "/api/catalog":
                self.respond_json(200, demo.catalog())
            elif url.path == "/api/segment":
                self.segment(parse_qs(url.query))
            else:
                self.respond_json(404, {"error": f"Not found: {url.path}"})

        def segment(self, query: dict[str, list[str]]) -> None:
            domain = query.get("domain", [""])[0]
            index = query.get("index", [""])[0]
            if domain not in demo.domains or not index.isdigit() or int(index) >= len(demo.domains[domain]):
                self.respond_json(400, {"error": f"No image {index!r} in domain {domain!r}"})
                return
            try:
                self.respond_json(200, demo.segment(domain, int(index)))
            except Exception as error:  # Shown on the page; the server keeps running.
                self.log_error("%s", repr(error))
                self.respond_json(500, {"error": repr(error)})

        def log_message(self, format: str, *args: object) -> None:
            pass

        def log_error(self, format: str, *args: object) -> None:
            print(format % args, file=sys.stderr)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", help="PyTorch device (default: cuda when available)")
    add_arguments(parser)
    parser.set_defaults(arch="nnunet_official", task="prostate", candidates="head")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading the dataset and the models...")
    demo = Demo(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(demo))
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
