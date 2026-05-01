import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2

SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
}

MODEL_CONFIGS = {
        "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
        }
}

_MODEL = None


def _resolve_device(device_arg: str) -> str:
        requested = str(device_arg).strip().lower()
        if requested in ("", "auto"):
                return "cuda:0" if torch.cuda.is_available() else "cpu"

        if requested.startswith("cpu"):
                return "cpu"

        if not torch.cuda.is_available():
                return "cpu"

        if requested == "cuda":
                return "cuda:0"

        if requested.startswith("cuda:") and ("," in requested):
                first = requested.split(":", 1)[1].split(",", 1)[0].strip()
                first = first if first else "0"
                return f"cuda:{first}"

        return requested


def _get_model(device_arg: str = "auto"):
        global _MODEL
        if _MODEL is not None:
                return _MODEL

        device = _resolve_device(device_arg)
        encoder = "vits"
        model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
        model.load_state_dict(
                torch.load(
                        f"scripts/depth_anything_v2_{encoder}.pth", map_location="cpu"
                )
        )
        _MODEL = model.to(device).eval()
        return _MODEL


def generate_depthmap(input_dir, output_dir, device: str = "auto"):
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists() or not input_path.is_dir():
                raise ValueError(
                        f"Input directory does not exist or is not a directory: {input_dir}"
                )

        output_path.mkdir(parents=True, exist_ok=True)
        model = _get_model(device)

        processed = 0
        skipped = 0
        failed = 0

        for img_path in sorted(input_path.rglob("*")):
                if (
                        not img_path.is_file()
                        or img_path.suffix.lower() not in SUPPORTED_EXTENSIONS
                ):
                        skipped += 1
                        continue

                raw_image = cv2.imread(str(img_path))
                if raw_image is None:
                        skipped += 1
                        continue

                with torch.no_grad():
                        depth = model.infer_image(raw_image)

                depth_min, depth_max = depth.min(), depth.max()
                if depth_max - depth_min > 0:
                        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
                else:
                        depth_normalized = np.zeros_like(depth)

                depth_8bit = (depth_normalized * 255.0).astype(np.uint8)
                save_path = output_path / img_path.relative_to(input_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if cv2.imwrite(str(save_path), depth_8bit):
                        processed += 1
                else:
                        failed += 1

        print(f"Done. processed={processed} skipped={skipped} failed={failed}")
        return {"processed": processed, "skipped": skipped, "failed": failed}


def parse_args():
        parser = argparse.ArgumentParser(
                description="Generate depth maps for images in input directory."
        )
        parser.add_argument(
                "input_dir", type=str, help="Input directory with source images"
        )
        parser.add_argument(
                "output_dir", type=str, help="Output directory for depth maps"
        )
        parser.add_argument(
                "--device",
                type=str,
                default="auto",
                help=(
                        "Compute device: auto | cpu | cuda | cuda:N | cuda:N,M. "
                        "For multi-GPU strings, first GPU is used."
                ),
        )
        return parser.parse_args()


def main():
        args = parse_args()
        generate_depthmap(args.input_dir, args.output_dir, device=args.device)


if __name__ == "__main__":
        main()
