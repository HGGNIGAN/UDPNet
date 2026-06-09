from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from PIL import Image

from pipeline.common.paths import resolve_from_project_root

ConfigDict = Dict[str, Any]


def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))


def _to_int_class_id(token: str, class_map: Dict[str, int]) -> Optional[int]:
        stripped = token.strip()
        if stripped == "":
                return None

        mapped = class_map.get(stripped)
        if mapped is not None:
                return mapped

        if stripped.lstrip("-").isdigit():
                return int(stripped)

        return None


def _read_image_size(image_path: Path) -> Tuple[int, int]:
        with Image.open(image_path) as image:
                return image.width, image.height


def _convert_yolo_txt(
        label_path: Path,
        image_width: int,
        image_height: int,
        class_map: Dict[str, int],
        assume_normalized: bool,
        clamp_boxes: bool,
        skip_unknown_classes: bool,
) -> Tuple[List[str], Dict[str, int]]:
        converted_lines: List[str] = []
        stats = {
                "unknown_class": 0,
                "invalid_box": 0,
                "empty_lines": 0,
                "difficult": 0,
        }

        with label_path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                        line = raw.strip()
                        if not line:
                                stats["empty_lines"] += 1
                                continue

                        parts = line.split()
                        if len(parts) < 5:
                                stats["invalid_box"] += 1
                                continue

                        class_id = _to_int_class_id(parts[0], class_map)
                        if class_id is None:
                                stats["unknown_class"] += 1
                                if skip_unknown_classes:
                                        continue
                                class_id = -1

                        try:
                                cx = float(parts[1])
                                cy = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                        except ValueError:
                                stats["invalid_box"] += 1
                                continue

                        if assume_normalized:
                                if max(abs(cx), abs(cy), abs(width), abs(height)) > 2.0:
                                        cx /= float(image_width)
                                        cy /= float(image_height)
                                        width /= float(image_width)
                                        height /= float(image_height)
                        else:
                                cx /= float(image_width)
                                cy /= float(image_height)
                                width /= float(image_width)
                                height /= float(image_height)

                        if clamp_boxes:
                                cx = _clamp01(cx)
                                cy = _clamp01(cy)
                                width = _clamp01(width)
                                height = _clamp01(height)

                        if width <= 0.0 or height <= 0.0:
                                stats["invalid_box"] += 1
                                continue

                        converted_lines.append(
                                f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
                        )

        return converted_lines, stats


def _convert_voc_xml(
        label_path: Path,
        image_width: int,
        image_height: int,
        class_map: Dict[str, int],
        clamp_boxes: bool,
        skip_unknown_classes: bool,
) -> Tuple[List[str], Dict[str, int]]:
        converted_lines: List[str] = []
        stats = {
                "unknown_class": 0,
                "invalid_box": 0,
                "empty_lines": 0,
                "difficult": 0,
        }

        root = ET.parse(label_path).getroot()

        for obj in root.findall("object"):
                difficult = obj.findtext("difficult")
                if difficult is not None and int((difficult.strip() or "0")) == 1:
                        stats["difficult"] += 1
                        continue

                class_name = (obj.findtext("name") or "").strip()
                class_id = _to_int_class_id(class_name, class_map)
                if class_id is None:
                        stats["unknown_class"] += 1
                        if skip_unknown_classes:
                                continue
                        class_id = -1

                bnd = obj.find("bndbox")
                if bnd is None:
                        stats["invalid_box"] += 1
                        continue

                try:
                        xmin = float((bnd.findtext("xmin") or "").strip())
                        ymin = float((bnd.findtext("ymin") or "").strip())
                        xmax = float((bnd.findtext("xmax") or "").strip())
                        ymax = float((bnd.findtext("ymax") or "").strip())
                except ValueError:
                        stats["invalid_box"] += 1
                        continue

                if xmax <= xmin or ymax <= ymin:
                        stats["invalid_box"] += 1
                        continue

                width_px = xmax - xmin
                height_px = ymax - ymin
                cx_px = xmin + width_px / 2.0
                cy_px = ymin + height_px / 2.0

                cx = cx_px / float(image_width)
                cy = cy_px / float(image_height)
                width = width_px / float(image_width)
                height = height_px / float(image_height)

                if clamp_boxes:
                        cx = _clamp01(cx)
                        cy = _clamp01(cy)
                        width = _clamp01(width)
                        height = _clamp01(height)

                if width <= 0.0 or height <= 0.0:
                        stats["invalid_box"] += 1
                        continue

                converted_lines.append(
                        f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
                )

        return converted_lines, stats


CONVERTER_REGISTRY = {
        "yolo_txt": _convert_yolo_txt,
        "voc_xml": _convert_voc_xml,
}


def _sum_stats(target: Dict[str, int], addend: Dict[str, int]) -> None:
        for key, value in addend.items():
                target[key] = target.get(key, 0) + value


def _load_pairs_csv(pairs_path: Path) -> List[Dict[str, str]]:
        if not pairs_path.exists():
                raise FileNotFoundError(f"Pairs manifest not found: {pairs_path}")

        with pairs_path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))


def normalize_ground_truth(
        config: ConfigDict,
        selected_datasets: Optional[Sequence[str]] = None,
        overwrite: bool = False,
) -> Dict[str, Any]:
        datasets_cfg = config.get("datasets", {})
        entries = datasets_cfg.get("entries", {})
        active = (
                list(selected_datasets)
                if selected_datasets
                else list(datasets_cfg.get("active", entries.keys()))
        )

        paths_cfg = config.get("paths", {})
        normalization_cfg = config.get("normalization", {})

        organized_root = resolve_from_project_root(
                config,
                str(paths_cfg.get("organized_root", "Datasets/organized")),
        )

        output_label_dir_name = str(
                normalization_cfg.get("output_label_dir_name", "labels_yolo")
        )
        clamp_boxes = bool(normalization_cfg.get("clamp_boxes", True))
        skip_unknown_classes = bool(normalization_cfg.get("skip_unknown_classes", True))
        assume_normalized = bool(
                normalization_cfg.get("yolo_txt_assume_normalized", True)
        )

        run_summary: Dict[str, Any] = {
                "organized_root": str(organized_root),
                "output_label_dir_name": output_label_dir_name,
                "overwrite": overwrite,
                "datasets": {},
        }

        for dataset_name in active:
                if dataset_name not in entries:
                        raise KeyError(
                                f"Dataset not found in config entries: {dataset_name}"
                        )

                dataset_cfg = entries[dataset_name]
                annotation_format = str(
                        dataset_cfg.get("annotation_format", "")
                ).lower()
                converter = CONVERTER_REGISTRY.get(annotation_format)
                if converter is None:
                        valid_formats = ", ".join(sorted(CONVERTER_REGISTRY.keys()))
                        raise ValueError(
                                f"Unsupported annotation_format '{annotation_format}' for {dataset_name}. Supported: {valid_formats}"
                        )

                dataset_dir = organized_root / dataset_name
                manifests_dir = dataset_dir / "manifests"
                pairs_path = manifests_dir / "pairs.csv"
                rows = _load_pairs_csv(pairs_path)

                out_label_dir = dataset_dir / output_label_dir_name
                out_label_dir.mkdir(parents=True, exist_ok=True)

                class_map = {
                        str(key): int(value)
                        for key, value in dict(dataset_cfg.get("class_map", {})).items()
                }

                aggregate_stats = {
                        "unknown_class": 0,
                        "invalid_box": 0,
                        "empty_lines": 0,
                        "difficult": 0,
                        "images_processed": 0,
                        "labels_written": 0,
                        "annotations_written": 0,
                }

                normalized_rows: List[Dict[str, str]] = []

                for row in rows:
                        image_path = Path(row["image_path"]).resolve()
                        label_path = Path(row["label_path"]).resolve()
                        image_id = row["image_id"]
                        yolo_label_path = out_label_dir / f"{image_id}.txt"

                        if yolo_label_path.exists() and (not overwrite):
                                normalized_rows.append(
                                        {
                                                **row,
                                                "normalized_label_path": str(
                                                        yolo_label_path
                                                ),
                                        }
                                )
                                aggregate_stats["images_processed"] += 1
                                aggregate_stats["labels_written"] += 1
                                continue

                        image_w, image_h = _read_image_size(image_path)

                        if annotation_format == "yolo_txt":
                                lines, stats = converter(
                                        label_path,
                                        image_w,
                                        image_h,
                                        class_map,
                                        assume_normalized,
                                        clamp_boxes,
                                        skip_unknown_classes,
                                )
                        else:
                                lines, stats = converter(
                                        label_path,
                                        image_w,
                                        image_h,
                                        class_map,
                                        clamp_boxes,
                                        skip_unknown_classes,
                                )

                        with yolo_label_path.open("w", encoding="utf-8") as handle:
                                if lines:
                                        handle.write("\n".join(lines) + "\n")

                        normalized_rows.append(
                                {
                                        **row,
                                        "normalized_label_path": str(yolo_label_path),
                                }
                        )

                        aggregate_stats["images_processed"] += 1
                        aggregate_stats["labels_written"] += 1
                        aggregate_stats["annotations_written"] += len(lines)
                        _sum_stats(aggregate_stats, stats)

                yolo_pairs_path = manifests_dir / "pairs_yolo.csv"
                fieldnames = (
                        list(normalized_rows[0].keys())
                        if normalized_rows
                        else [
                                "dataset",
                                "split",
                                "image_id",
                                "annotation_format",
                                "image_src",
                                "label_src",
                                "image_path",
                                "label_path",
                                "normalized_label_path",
                        ]
                )

                with yolo_pairs_path.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(normalized_rows)

                dataset_summary = {
                        "dataset": dataset_name,
                        "annotation_format": annotation_format,
                        "pairs_input": len(rows),
                        "pairs_output": len(normalized_rows),
                        "pairs_yolo_path": str(yolo_pairs_path),
                        "output_label_dir": str(out_label_dir),
                        "stats": aggregate_stats,
                }

                summary_path = manifests_dir / "normalization_summary.json"
                with summary_path.open("w", encoding="utf-8") as handle:
                        json.dump(dataset_summary, handle, indent=2)

                dataset_summary["summary_path"] = str(summary_path)
                run_summary["datasets"][dataset_name] = dataset_summary

        return run_summary
