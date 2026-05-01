from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pipeline.common.paths import resolve_from_project_root, resolve_under

ConfigDict = Dict[str, Any]


def _normalize_extensions(
        extensions: Optional[Sequence[str]], default: Sequence[str]
) -> Set[str]:
        base = extensions if extensions else default
        return {f".{str(ext).lstrip('.').lower()}" for ext in base}


def _iter_files(directory: Path, allowed_extensions: Set[str]) -> Iterable[Path]:
        if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")

        for item in sorted(directory.iterdir()):
                if item.is_file() and item.suffix.lower() in allowed_extensions:
                        yield item


def _index_files_by_stem(directory: Path, extensions: Set[str]) -> Dict[str, Path]:
        return {path.stem: path for path in _iter_files(directory, extensions)}


def _read_split_ids(split_file: Path) -> List[str]:
        if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")

        values: List[str] = []
        with split_file.open("r", encoding="utf-8") as handle:
                for raw in handle:
                        value = raw.strip()
                        if value:
                                values.append(value)

        return values


def _collect_pairs(
        image_index: Dict[str, Path],
        label_index: Dict[str, Path],
        allowed_ids: Optional[Sequence[str]],
) -> Tuple[List[str], List[str], List[str]]:
        if allowed_ids is None:
                paired_ids = sorted(set(image_index).intersection(label_index))
                missing_images = sorted(set(label_index) - set(image_index))
                missing_labels = sorted(set(image_index) - set(label_index))
                return paired_ids, missing_images, missing_labels

        paired_ids: List[str] = []
        missing_images: List[str] = []
        missing_labels: List[str] = []

        for image_id in allowed_ids:
                has_image = image_id in image_index
                has_label = image_id in label_index

                if has_image and has_label:
                        paired_ids.append(image_id)
                        continue

                if not has_image:
                        missing_images.append(image_id)
                if not has_label:
                        missing_labels.append(image_id)

        return paired_ids, missing_images, missing_labels


def _scan_common(
        dataset_name: str,
        dataset_cfg: ConfigDict,
        config: ConfigDict,
        default_label_extensions: Sequence[str],
) -> Dict[str, Any]:
        root = resolve_from_project_root(config, str(dataset_cfg["root"]))
        image_dir = resolve_under(root, str(dataset_cfg["image_dir"]))
        label_dir = resolve_under(root, str(dataset_cfg["label_dir"]))

        image_extensions = _normalize_extensions(
                dataset_cfg.get("image_extensions"),
                [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        )
        label_extensions = _normalize_extensions(
                dataset_cfg.get("label_extensions"), default_label_extensions
        )

        split_ids: Optional[List[str]] = None
        if dataset_cfg.get("split_file"):
                split_file = resolve_under(root, str(dataset_cfg["split_file"]))
                split_ids = _read_split_ids(split_file)

        image_index = _index_files_by_stem(image_dir, image_extensions)
        label_index = _index_files_by_stem(label_dir, label_extensions)

        paired_ids, missing_images, missing_labels = _collect_pairs(
                image_index, label_index, split_ids
        )

        return {
                "dataset": dataset_name,
                "split": str(dataset_cfg.get("split", "all")),
                "annotation_format": str(dataset_cfg["annotation_format"]).lower(),
                "root": root,
                "image_dir": image_dir,
                "label_dir": label_dir,
                "image_by_id": image_index,
                "label_by_id": label_index,
                "paired_ids": paired_ids,
                "missing_images": missing_images,
                "missing_labels": missing_labels,
                "used_split_file": bool(split_ids is not None),
        }


def _scan_yolo_txt_dataset(
        dataset_name: str, dataset_cfg: ConfigDict, config: ConfigDict
) -> Dict[str, Any]:
        return _scan_common(
                dataset_name, dataset_cfg, config, default_label_extensions=[".txt"]
        )


def _scan_voc_xml_dataset(
        dataset_name: str, dataset_cfg: ConfigDict, config: ConfigDict
) -> Dict[str, Any]:
        return _scan_common(
                dataset_name, dataset_cfg, config, default_label_extensions=[".xml"]
        )


SCANNER_REGISTRY = {
        "yolo_txt": _scan_yolo_txt_dataset,
        "voc_xml": _scan_voc_xml_dataset,
}


def _materialize_file(src: Path, dst: Path, link_mode: str, overwrite: bool) -> bool:
        if dst.exists() or dst.is_symlink():
                if not overwrite:
                        return False
                dst.unlink()

        dst.parent.mkdir(parents=True, exist_ok=True)

        if link_mode == "symlink":
                dst.symlink_to(src.resolve())
                return True

        if link_mode == "hardlink":
                try:
                        os.link(src, dst)
                        return True
                except OSError:
                        shutil.copy2(src, dst)
                        return True

        if link_mode == "copy":
                shutil.copy2(src, dst)
                return True

        raise ValueError(f"Unsupported link_mode: {link_mode}")


def _write_manifest(records: List[Dict[str, Any]], manifest_path: Path) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
                "dataset",
                "split",
                "image_id",
                "annotation_format",
                "image_src",
                "label_src",
                "image_path",
                "label_path",
        ]

        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)


def _write_summary(summary_path: Path, summary_data: Dict[str, Any]) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary_data, handle, indent=2)


def scan_and_pair_datasets(
        config: ConfigDict,
        selected_datasets: Optional[Sequence[str]] = None,
        link_mode: Optional[str] = None,
        overwrite: Optional[bool] = None,
        dry_run: bool = False,
) -> Dict[str, Any]:
        datasets_cfg = config.get("datasets", {})
        entries = datasets_cfg.get("entries", {})
        active = (
                list(selected_datasets)
                if selected_datasets
                else list(datasets_cfg.get("active", entries.keys()))
        )

        scan_cfg = datasets_cfg.get("scan", {})
        effective_link_mode = str(
                link_mode or scan_cfg.get("link_mode", "symlink")
        ).lower()
        if effective_link_mode not in {"symlink", "hardlink", "copy"}:
                raise ValueError("link_mode must be one of: symlink, hardlink, copy")

        effective_overwrite = bool(
                scan_cfg.get("overwrite", False) if overwrite is None else overwrite
        )

        organized_root = resolve_from_project_root(
                config,
                str(
                        config.get("paths", {}).get(
                                "organized_root", "Datasets/organized"
                        )
                ),
        )

        if not dry_run:
                organized_root.mkdir(parents=True, exist_ok=True)

        run_summary: Dict[str, Any] = {
                "organized_root": str(organized_root),
                "link_mode": effective_link_mode,
                "overwrite": effective_overwrite,
                "dry_run": dry_run,
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
                scanner = SCANNER_REGISTRY.get(annotation_format)
                if scanner is None:
                        valid_formats = ", ".join(sorted(SCANNER_REGISTRY.keys()))
                        raise ValueError(
                                f"Unsupported annotation_format '{annotation_format}' for dataset {dataset_name}. "
                                f"Supported: {valid_formats}"
                        )

                scan_data = scanner(dataset_name, dataset_cfg, config)

                dataset_out_dir = organized_root / dataset_name
                images_out_dir = dataset_out_dir / "images"
                labels_out_dir = dataset_out_dir / "labels_raw"
                manifests_dir = dataset_out_dir / "manifests"

                records: List[Dict[str, Any]] = []
                materialized_images = 0
                materialized_labels = 0

                for image_id in scan_data["paired_ids"]:
                        image_src = scan_data["image_by_id"][image_id]
                        label_src = scan_data["label_by_id"][image_id]

                        image_dst = images_out_dir / image_src.name
                        label_dst = labels_out_dir / label_src.name

                        if not dry_run:
                                if _materialize_file(
                                        image_src,
                                        image_dst,
                                        effective_link_mode,
                                        effective_overwrite,
                                ):
                                        materialized_images += 1
                                if _materialize_file(
                                        label_src,
                                        label_dst,
                                        effective_link_mode,
                                        effective_overwrite,
                                ):
                                        materialized_labels += 1

                        records.append(
                                {
                                        "dataset": dataset_name,
                                        "split": scan_data["split"],
                                        "image_id": image_id,
                                        "annotation_format": annotation_format,
                                        "image_src": str(image_src),
                                        "label_src": str(label_src),
                                        "image_path": str(image_dst),
                                        "label_path": str(label_dst),
                                }
                        )

                manifest_path = manifests_dir / "pairs.csv"
                summary_path = manifests_dir / "summary.json"

                dataset_summary = {
                        "dataset": dataset_name,
                        "split": scan_data["split"],
                        "annotation_format": annotation_format,
                        "paired_count": len(scan_data["paired_ids"]),
                        "missing_images_count": len(scan_data["missing_images"]),
                        "missing_labels_count": len(scan_data["missing_labels"]),
                        "missing_images_sample": scan_data["missing_images"][:20],
                        "missing_labels_sample": scan_data["missing_labels"][:20],
                        "used_split_file": scan_data["used_split_file"],
                        "manifest_path": str(manifest_path),
                        "summary_path": str(summary_path),
                        "materialized_images": materialized_images,
                        "materialized_labels": materialized_labels,
                }

                if not dry_run:
                        _write_manifest(records, manifest_path)
                        _write_summary(summary_path, dataset_summary)

                run_summary["datasets"][dataset_name] = dataset_summary

        return run_summary
