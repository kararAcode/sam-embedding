#!/usr/bin/env python3
"""
Export SAM image embedding to a NumPy .npy file and metadata.json.

Usage:
  python3 sam_embedding.py --input <image-path> --output <path.npy>
  python3 sam_embedding.py --input img.jpg --output embed.npy --checkpoint sam_vit_h_4b8939.pth

Requires:
  torch, torchvision, segment-anything, opencv-python-headless, numpy

The array shape is typically (1, 256, 64, 64) float32 for vit_h, matching
SamPredictor.get_image_embedding() after set_image().
"""

from _future_ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch



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
    parser = argparse.ArgumentParser(
        description="Save SAM ViT image embedding as .npy and metadata.json."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input image path")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path (.npy); .npy is appended if missing",
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
        except Exception as e:
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

    out_path = args.output
    if out_path.suffix.lower() != ".npy":
        out_path = out_path.with_suffix(".npy")

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), arr)

    metadata = {
        "modelType": args.model_type,
        "checkpoint": args.checkpoint.name,
        "device": str(device),
        "embeddingFile": out_path.name,
        "embeddingShape": list(arr.shape),
        "embeddingDtype": str(arr.dtype),
        "originalSize": list(predictor.original_size),
        "inputSize": list(predictor.input_size),
    }

    metadata_path = out_path.with_name("metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Saved embedding shape={arr.shape} dtype={arr.dtype} -> {out_path}\n"
        f"Saved metadata -> {metadata_path}"
    )



if _name_ == "_main_":
    main()
