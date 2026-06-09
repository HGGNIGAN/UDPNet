#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
                "Pillow is required. Install it with `pip install Pillow`."
        ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
        ("yolo26n", PROJECT_ROOT / "outputs" / "FSNet_UDPNet_OTS_yolo26n"),
        ("yolov8n", PROJECT_ROOT / "outputs" / "FSNet_UDPNet_OTS_yolov8n"),
        ("yolo26x", PROJECT_ROOT / "outputs" / "FSNet_UDPNet_OTS_yolo26x"),
]
DEFAULT_DATASETS = ("RTTS", "FoggyCityscape", "DAWN")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figs" / "report_comparisons"
PANEL_COLUMNS = ("original", "restored", "original_detection", "restored_detection")
COLUMN_LABELS = {
        "original": "Original",
        "restored": "Restored",
        "original_detection": "Original detection",
        "restored_detection": "Restored detection",
}
RUN_LABELS = {
        "yolo26n": "yolo26n",
        "yolov8n": "yolov8n",
        "yolo26x": "yolo26x",
}
SampleKey = str


def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description="Generate comparison panels from existing UDPNet run visuals."
        )
        parser.add_argument(
                "--seed",
                type=int,
                default=42,
                help="Deterministic seed for sample selection.",
        )
        parser.add_argument(
                "--output-dir",
                type=Path,
                default=DEFAULT_OUTPUT_DIR,
                help="Destination folder for generated panels.",
        )
        parser.add_argument(
                "--run-dir",
                action="append",
                default=None,
                help="Optional run directory override. Repeat for multiple runs.",
        )
        parser.add_argument(
                "--sample",
                action="append",
                default=None,
                help="Override a sample filename. Use DATASET=FILE.",
        )
        return parser.parse_args()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
                return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
                return ImageFont.load_default()


TITLE_FONT = font(24)
LABEL_FONT = font(18)
SMALL_FONT = font(14)


def list_image_names(folder: Path) -> List[str]:
        if not folder.exists():
                return []
        return sorted(
                child.name
                for child in folder.iterdir()
                if child.is_file()
                and child.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )


def resolve_runs(run_dir_overrides: Sequence[Path] | None) -> List[Tuple[str, Path]]:
        if run_dir_overrides:
                labels = [
                        path.name.split("FSNet_UDPNet_OTS_")[-1]
                        for path in run_dir_overrides
                ]
                return list(zip(labels, run_dir_overrides))
        return list(DEFAULT_RUNS)


def parse_sample_overrides(raw_overrides: Sequence[str] | None) -> Dict[SampleKey, str]:
        overrides: Dict[SampleKey, str] = {}

        def process_line(raw_value: str) -> None:
                raw_value = raw_value.strip()
                if not raw_value or raw_value.startswith("#"):
                        return
                if "=" not in raw_value:
                        raise ValueError(
                                f"Invalid sample entry: {raw_value!r}. Expected DATASET=FILE."
                        )
                left, sample_name = raw_value.split("=", 1)
                left = left.strip()
                sample_name = sample_name.strip()
                if not left or not sample_name:
                        raise ValueError(
                                f"Invalid sample entry: {raw_value!r}. Expected DATASET=FILE."
                        )

                if ":" in left:
                        raise ValueError(
                                f"Invalid sample entry: {raw_value!r}. Expected DATASET=FILE."
                        )
                key = left

                overrides[key] = sample_name

        for raw_value in raw_overrides or []:
                if not raw_value:
                        continue
                p = Path(raw_value)
                # If argument is a filepath, read entries from it.
                if p.exists() and p.is_file():
                        for line in p.read_text(encoding="utf-8").splitlines():
                                process_line(line)
                        continue
                # Otherwise treat arg as inline DATASET=FILE entry.
                process_line(raw_value)

        return overrides


def collect_common_names(run_dirs: Sequence[Tuple[str, Path]], dataset: str) -> List[str]:
        common_names: set[str] | None = None
        for _, run_dir in run_dirs:
                folder = run_dir / "visuals" / dataset / "original"
                names = set(list_image_names(folder))
                if common_names is None:
                        common_names = names
                else:
                        common_names &= names
                if not common_names:
                        return []
        return sorted(common_names or [])


def pick_sample(rng: random.Random, candidates: Sequence[str]) -> str:
        if not candidates:
                raise ValueError(
                        "No shared filenames available for this dataset."
                )
        return rng.choice(list(candidates))


def resolve_sample_name(
        rng: random.Random,
        run_dirs: Sequence[Tuple[str, Path]],
        dataset: str,
        overrides: Dict[SampleKey, str],
) -> str:
        override = overrides.get(dataset)
        if override is not None:
                common_names = collect_common_names(run_dirs, dataset)
                if override not in common_names:
                        raise ValueError(
                                f"Override sample {override!r} is not shared across runs for {dataset}."
                        )
                return override

        shared_names = collect_common_names(run_dirs, dataset)
        return pick_sample(rng, shared_names)


def load_panel_image(path: Path, size: Tuple[int, int]) -> Image.Image:
        with Image.open(path) as image:
                image = image.convert("RGB")
                fitted = ImageOps.contain(
                        image, size, method=getattr(Image, "Resampling", Image).LANCZOS
                )
                canvas = Image.new("RGB", size, "white")
                offset_x = (size[0] - fitted.width) // 2
                offset_y = (size[1] - fitted.height) // 2
                canvas.paste(fitted, (offset_x, offset_y))
                return canvas


def draw_text_center(
        draw: ImageDraw.ImageDraw,
        box: Tuple[int, int, int, int],
        text: str,
        font_obj,
        fill: str = "black",
) -> None:
        left, top, right, bottom = box
        bbox = draw.multiline_textbbox((0, 0), text, font=font_obj, align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = left + (right - left - text_w) // 2
        y = top + (bottom - top - text_h) // 2
        draw.multiline_text((x, y), text, font=font_obj, fill=fill, align="center")


def build_panel(
        run_dirs: Sequence[Tuple[str, Path]],
        dataset: str,
        sample_name: str,
        output_path: Path,
) -> Dict[str, str]:
        cell_w = 420
        cell_h = 280
        title_h = 76
        row_label_w = 180
        header_h = 48
        pad = 14
        image_h = cell_h - 44
        image_w = cell_w - 18

        width = row_label_w + len(PANEL_COLUMNS) * (cell_w + pad) + pad
        height = title_h + header_h + len(run_dirs) * (cell_h + pad) + pad
        panel = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(panel)

        title = f"{dataset} representative comparison"
        subtitle = f"Sample: {sample_name} | Rows: yolo26n, yolov8n, yolo26x | Columns: original, restored, original_detection, restored_detection"
        draw_text_center(draw, (pad, 10, width - pad, title_h - 28), title, TITLE_FONT)
        draw_text_center(
                draw, (pad, title_h - 28, width - pad, title_h), subtitle, SMALL_FONT
        )

        # Column headers.
        for col_index, column_name in enumerate(PANEL_COLUMNS):
                x0 = row_label_w + pad + col_index * (cell_w + pad)
                header_box = (x0, title_h + pad, x0 + cell_w, title_h + header_h)
                draw.rounded_rectangle(
                        header_box, radius=10, fill="#f1f3f5", outline="#c9ced6"
                )
                draw_text_center(
                        draw, header_box, COLUMN_LABELS[column_name], LABEL_FONT
                )

        # Row labels and images.
        manifest_rows = []
        for row_index, (run_label, run_dir) in enumerate(run_dirs):
                y0 = title_h + header_h + pad + row_index * (cell_h + pad)
                row_box = (pad, y0, row_label_w + pad, y0 + cell_h)
                draw.rounded_rectangle(
                        row_box, radius=12, fill="#eef6ff", outline="#b8d7ff"
                )
                draw_text_center(draw, row_box, RUN_LABELS[run_label], LABEL_FONT)

                sample_entry: Dict[str, str] = {"run": run_label}
                for col_index, column_name in enumerate(PANEL_COLUMNS):
                        x0 = row_label_w + pad + col_index * (cell_w + pad)
                        cell_box = (x0, y0, x0 + cell_w, y0 + cell_h)
                        draw.rounded_rectangle(
                                cell_box, radius=10, fill="white", outline="#d6dbe3"
                        )
                        image_folder = run_dir / "visuals" / dataset / column_name
                        image_path = image_folder / sample_name
                        if not image_path.exists():
                                raise FileNotFoundError(f"Missing visual: {image_path}")
                        sample_entry[column_name] = str(image_path)
                        fitted = load_panel_image(image_path, (image_w, image_h))
                        img_x = x0 + (cell_w - image_w) // 2
                        img_y = y0 + 8
                        panel.paste(fitted, (img_x, img_y))
                        caption_box = (
                                x0 + 8,
                                y0 + cell_h - 30,
                                x0 + cell_w - 8,
                                y0 + cell_h - 6,
                        )
                        draw_text_center(
                                draw,
                                caption_box,
                                sample_name,
                                SMALL_FONT,
                                fill="#333333",
                        )
                manifest_rows.append(sample_entry)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(output_path)
        return {
                "dataset": dataset,
                "sample_name": sample_name,
                "output_path": str(output_path),
                "rows": manifest_rows,
        }


def generate_panels(
        run_dirs: Sequence[Tuple[str, Path]],
        seed: int,
        output_dir: Path,
        sample_overrides: Dict[SampleKey, str],
) -> List[Dict[str, str]]:
        rng = random.Random(seed)
        records: List[Dict[str, str]] = []

        for dataset in DEFAULT_DATASETS:
                sample_name = resolve_sample_name(
                        rng, run_dirs, dataset, sample_overrides
                )
                output_path = (
                        output_dir
                        / "by_dataset"
                        / dataset
                        / f"{dataset}_comparison.png"
                )
                records.append(build_panel(run_dirs, dataset, sample_name, output_path))

        manifest_path = output_dir / "panels.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
                json.dumps({"seed": seed, "records": records}, indent=2),
                encoding="utf-8",
        )
        return records


def main() -> None:
        args = parse_args()
        if args.run_dir:
                run_dirs = []
                for raw_dir in args.run_dir:
                        run_path = Path(raw_dir).expanduser().resolve()
                        run_dirs.append(
                                (run_path.name.split("FSNet_UDPNet_OTS_")[-1], run_path)
                        )
        else:
                run_dirs = resolve_runs(None)

        sample_overrides = parse_sample_overrides(args.sample)
        records = generate_panels(
                run_dirs,
                args.seed,
                args.output_dir.expanduser().resolve(),
                sample_overrides,
        )
        print(
                json.dumps(
                        {"generated": len(records), "output_dir": str(args.output_dir)},
                        indent=2,
                )
        )


if __name__ == "__main__":
        main()
