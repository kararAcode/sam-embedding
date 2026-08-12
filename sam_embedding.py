#!/usr/bin/env python3
"""
Export a SAM image embedding and manifest for GreenSkEye.

Usage:
  python3 sam_embedding.py --input <image-path> --output <output-directory>
  python3 sam_embedding.py --input img.jpg --output artifacts --checkpoint sam_vit_h_4b8939.pth

Requires:
  torch, torchvision, segment-anything, opencv-python-headless, numpy

The array shape is typically (1, 256, 64, 64) float32 for vit_h, matching
SamPredictor.get_image_embedding() after set_image().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def download_checkpoint(model_type: str, checkpoint_path: Path) -> None:
    url = SAM_CHECKPOINT_URLS[model_type]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".part")

    if partial_path.exists():
        partial_path.unlink()

    print(f"Checkpoint not found. Downloading {model_type} weights...")
    print(f"Source: {url}")
    print(f"Target: {checkpoint_path}")
    urlretrieve(url, partial_path)
    partial_path.replace(checkpoint_path)


def main() -> None:
    import cv2
    import torch

    parser = argparse.ArgumentParser(description="Export GreenSkEye SAM artifacts.")
    parser.add_argument("--input", required=True, type=Path, help="Input image path")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for manifest.json and embedding.bin",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("sam_vit_h_4b8939.pth"),
        help="SAM checkpoint .pth file",
    )
    parser.add_argument(
        "--model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="Must match checkpoint",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.checkpoint.is_file():
        try:
            download_checkpoint(args.model_type, args.checkpoint)
        except (OSError, ValueError) as e:
            print(f"Failed to download checkpoint: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as e:
        print(
            "Install: pip install torch torchvision segment-anything opencv-python numpy\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    bgr = cv2.imread(str(args.input))
    if bgr is None:
        print(f"Could not read image: {args.input}", file=sys.stderr)
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device)
    sam.eval()

    predictor = SamPredictor(sam)

    with torch.inference_mode():
        predictor.set_image(rgb, image_format="RGB")
        emb = predictor.get_image_embedding()
        arr = emb.detach().cpu().to(torch.float32).numpy()

    manifest_path, embedding_path = write_artifacts(
        args.output,
        arr,
        model_type=args.model_type,
        checkpoint=args.checkpoint.name,
        original_size=predictor.original_size,
        input_size=predictor.input_size,
    )

    print(
        f"Saved embedding shape={arr.shape} dtype=float32 -> {embedding_path}\n"
        f"Saved manifest -> {manifest_path}"
    )


def write_artifacts(
    output_dir: Path,
    embedding: np.ndarray,
    *,
    model_type: str,
    checkpoint: str,
    original_size: tuple[int, int],
    input_size: tuple[int, int],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = output_dir / "embedding.bin"
    embedding = np.ascontiguousarray(embedding, dtype="<f4")
    embedding_path.write_bytes(embedding.tobytes())

    original_height, original_width = original_size
    input_height, input_width = input_size
    manifest = {
        "imageSize": {"width": original_width, "height": original_height},
        "embedding": embedding_path.name,
        "embeddingShape": list(embedding.shape),
        "embeddingDtype": "float32",
        "modelType": model_type,
        "checkpoint": checkpoint,
        "inputSize": {"width": input_width, "height": input_height},
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest_path, embedding_path


if __name__ == "__main__":
    main()
