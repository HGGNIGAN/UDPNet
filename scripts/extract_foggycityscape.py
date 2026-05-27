#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ALLOWED_CLASSES = {
        "person": "person",
        "rider": "rider",
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "train": "train",
        "motorcycle": "motorbike",
        "motorbike": "motorbike",
        "bicycle": "bicycle",
}


@dataclass(frozen=True)
class SceneRef:
        split: str
        city: str
        scene_id: str


@dataclass(frozen=True)
class EligibleSample:
        split: str
        city: str
        scene_id: str
        beta: float
        image_source: Path
        label_source: Path
        image_dest: Path
        label_dest: Path
        xml_payload: bytes
        object_count: int


def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description=(
                        "Extract a deterministic, city-balanced FoggyCityscape subset into "
                        "a flat VOC-style tree for the UDPNet pipeline."
                )
        )
        parser.add_argument(
                "--source-root", type=str, default="Datasets/FoggyCityscape-raw"
        )
        parser.add_argument(
                "--output-root", type=str, default="Datasets/FoggyCityscape"
        )
        parser.add_argument(
                "--refined-list",
                type=str,
                default="Datasets/FoggyCityscape-raw/foggy_trainval_refined_filenames.txt",
        )
        parser.add_argument("--per-city", type=int, default=20)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--preferred-beta", type=float, default=0.01)
        parser.add_argument("--beta-order", nargs="*", type=float, default=None)
        parser.add_argument("--split-filter", nargs="*", default=["train", "val"])
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        return parser.parse_args()


def load_refined_refs(
        refined_list: Path, split_filter: Sequence[str]
) -> List[SceneRef]:
        if not refined_list.exists():
                raise FileNotFoundError(f"Refined list not found: {refined_list}")

        allowed_splits = {item.strip().lower() for item in split_filter if item.strip()}
        refs: List[SceneRef] = []
        with refined_list.open("r", encoding="utf-8") as handle:
                for raw in handle:
                        value = raw.strip()
                        if not value:
                                continue

                        parts = value.split("/")
                        if len(parts) < 3:
                                continue

                        split, city = parts[0], parts[1]
                        scene_id = parts[-1]
                        if allowed_splits and split.lower() not in allowed_splits:
                                continue
                        refs.append(SceneRef(split=split, city=city, scene_id=scene_id))

        return refs


def beta_order(
        preferred_beta: float, beta_overrides: Optional[Sequence[float]]
) -> List[float]:
        order: List[float] = [preferred_beta]
        if beta_overrides:
                order.extend(beta_overrides)

        deduped: List[float] = []
        seen = set()
        for value in order:
                if value in seen:
                        continue
                seen.add(value)
                deduped.append(value)
        return deduped


def stable_city_seed(seed: int, city: str) -> int:
        digest = sha256(f"{seed}:{city}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16)


def extract_beta(filename: str) -> Optional[float]:
        marker = "_beta_"
        if marker not in filename:
                return None
        value = filename.rsplit(marker, 1)[-1].rsplit(".", 1)[0]
        try:
                return float(value)
        except ValueError:
                return None


def choose_beta_variant(
        candidates: Sequence[Tuple[float, Path, Path]],
        preferred_order: Sequence[float],
) -> Optional[Tuple[float, Path, Path]]:
        by_beta = {
                beta: (beta, image_path, label_path)
                for beta, image_path, label_path in candidates
        }
        for beta in preferred_order:
                if beta in by_beta:
                        return by_beta[beta]
        return candidates[0] if candidates else None


def polygon_bounds(
        points: Sequence[Sequence[float]],
) -> Optional[Tuple[float, float, float, float]]:
        xs: List[float] = []
        ys: List[float] = []
        for point in points:
                if len(point) < 2:
                        continue
                xs.append(float(point[0]))
                ys.append(float(point[1]))

        if not xs or not ys:
                return None

        return min(xs), min(ys), max(xs), max(ys)


def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))


def convert_polygons_to_voc_xml(
        label_path: Path,
        image_name: str,
        allowed_classes: Dict[str, str],
) -> Tuple[bytes, int]:
        with label_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

        width = int(payload["imgWidth"])
        height = int(payload["imgHeight"])

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = "FoggyCityscape"
        ET.SubElement(annotation, "filename").text = image_name
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(annotation, "segmented").text = "0"

        kept = 0
        for obj in payload.get("objects", []):
                raw_label = str(obj.get("label", "")).strip()
                if raw_label.endswith("group"):
                        continue

                mapped = allowed_classes.get(raw_label)
                if mapped is None:
                        continue

                bounds = polygon_bounds(obj.get("polygon", []))
                if bounds is None:
                        continue

                xmin, ymin, xmax, ymax = bounds
                xmin = clamp(xmin, 0.0, float(width - 1))
                ymin = clamp(ymin, 0.0, float(height - 1))
                xmax = clamp(xmax, 0.0, float(width - 1))
                ymax = clamp(ymax, 0.0, float(height - 1))
                if xmax <= xmin or ymax <= ymin:
                        continue

                obj_el = ET.SubElement(annotation, "object")
                ET.SubElement(obj_el, "name").text = mapped
                ET.SubElement(obj_el, "pose").text = "Unspecified"
                ET.SubElement(obj_el, "truncated").text = "0"
                ET.SubElement(obj_el, "difficult").text = "0"
                bndbox = ET.SubElement(obj_el, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(round(xmin)))
                ET.SubElement(bndbox, "ymin").text = str(int(round(ymin)))
                ET.SubElement(bndbox, "xmax").text = str(int(round(xmax)))
                ET.SubElement(bndbox, "ymax").text = str(int(round(ymax)))
                kept += 1

        return ET.tostring(annotation, encoding="utf-8", xml_declaration=True), kept


def scan_scene(
        source_root: Path,
        scene: SceneRef,
        preferred_order: Sequence[float],
) -> Optional[EligibleSample]:
        image_dir = source_root / "leftImg8bit_foggy" / scene.split / scene.city
        label_dir = source_root / "gtFine" / scene.split / scene.city
        if not image_dir.exists() or not label_dir.exists():
                return None

        candidates: List[Tuple[float, Path, Path]] = []
        for image_path in sorted(
                image_dir.glob(f"{scene.scene_id}_leftImg8bit_foggy_beta_*.png")
        ):
                beta = extract_beta(image_path.name)
                if beta is None:
                        continue
                label_path = label_dir / f"{scene.scene_id}_gtFine_polygons.json"
                if not label_path.exists():
                        continue
                candidates.append((beta, image_path, label_path))

        chosen = choose_beta_variant(candidates, preferred_order)
        if chosen is None:
                return None

        beta, image_source, label_source = chosen
        xml_payload, object_count = convert_polygons_to_voc_xml(
                label_source,
                image_source.name,
                DEFAULT_ALLOWED_CLASSES,
        )
        if object_count <= 0:
                return None

        return EligibleSample(
                split=scene.split,
                city=scene.city,
                scene_id=scene.scene_id,
                beta=beta,
                image_source=image_source,
                label_source=label_source,
                image_dest=Path(),
                label_dest=Path(),
                xml_payload=xml_payload,
                object_count=object_count,
        )


def sample_per_city(
        eligible_by_city: Dict[str, List[EligibleSample]],
        per_city: int,
        seed: int,
) -> List[EligibleSample]:
        selected: List[EligibleSample] = []
        for city in sorted(eligible_by_city):
                items = list(eligible_by_city[city])
                rng = random.Random(stable_city_seed(seed, city))
                rng.shuffle(items)
                selected.extend(items[:per_city])
        return selected


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
                "split",
                "city",
                "scene_id",
                "beta",
                "image_source",
                "label_source",
                "image_path",
                "label_path",
                "object_count",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def materialize_sample(sample: EligibleSample, overwrite: bool) -> None:
        sample.image_dest.parent.mkdir(parents=True, exist_ok=True)
        sample.label_dest.parent.mkdir(parents=True, exist_ok=True)
        if not sample.image_dest.exists() or overwrite:
                shutil.copy2(sample.image_source, sample.image_dest)
        if not sample.label_dest.exists() or overwrite:
                sample.label_dest.write_bytes(sample.xml_payload)


def extract_foggycityscape(
        source_root: Path,
        output_root: Path,
        refined_list: Path,
        per_city: int,
        seed: int,
        preferred_beta: float,
        beta_overrides: Optional[Sequence[float]],
        split_filter: Sequence[str],
        overwrite: bool,
        dry_run: bool,
) -> Dict[str, object]:
        refs = load_refined_refs(refined_list, split_filter)
        preferred_order = beta_order(preferred_beta, beta_overrides)

        eligible_by_city: Dict[str, List[EligibleSample]] = defaultdict(list)
        total_candidates = 0

        for scene in refs:
                eligible = scan_scene(source_root, scene, preferred_order)
                if eligible is None:
                        continue
                total_candidates += 1
                eligible_by_city[scene.city].append(eligible)

        selected = sample_per_city(eligible_by_city, per_city, seed)

        if not dry_run:
                output_root.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, object]] = []
        per_city_summary: Dict[str, Dict[str, int]] = {}
        selected_by_city: Dict[str, int] = defaultdict(int)

        for sample in selected:
                image_dest = output_root / "images" / sample.image_source.name
                label_dest = (
                        output_root / "labels_raw" / f"{sample.image_source.stem}.xml"
                )
                sample = EligibleSample(
                        split=sample.split,
                        city=sample.city,
                        scene_id=sample.scene_id,
                        beta=sample.beta,
                        image_source=sample.image_source,
                        label_source=sample.label_source,
                        image_dest=image_dest,
                        label_dest=label_dest,
                        xml_payload=sample.xml_payload,
                        object_count=sample.object_count,
                )

                rows.append(
                        {
                                "split": sample.split,
                                "city": sample.city,
                                "scene_id": sample.scene_id,
                                "beta": sample.beta,
                                "image_source": str(sample.image_source),
                                "label_source": str(sample.label_source),
                                "image_path": str(sample.image_dest),
                                "label_path": str(sample.label_dest),
                                "object_count": sample.object_count,
                        }
                )
                selected_by_city[sample.city] += 1

                if not dry_run:
                        materialize_sample(sample, overwrite=overwrite)

        for city, items in eligible_by_city.items():
                per_city_summary[city] = {
                        "eligible": len(items),
                        "selected": selected_by_city.get(city, 0),
                }

        manifests_dir = output_root / "manifests"
        summary = {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "refined_list": str(refined_list),
                "per_city_limit": per_city,
                "seed": seed,
                "preferred_beta": preferred_beta,
                "beta_order": list(preferred_order),
                "split_filter": list(split_filter),
                "dry_run": dry_run,
                "overwrite": overwrite,
                "refined_refs": len(refs),
                "eligible_scenes": total_candidates,
                "selected_scenes": len(selected),
                "manifests": {
                        "pairs_csv": str(manifests_dir / "extracted_pairs.csv"),
                        "summary_json": str(manifests_dir / "summary.json"),
                },
                "cities": per_city_summary,
                "selected_samples": rows,
        }

        if not dry_run:
                write_csv(manifests_dir / "extracted_pairs.csv", rows)
                write_json(manifests_dir / "summary.json", summary)

        return summary


def main() -> int:
        args = parse_args()
        summary = extract_foggycityscape(
                source_root=Path(args.source_root).expanduser().resolve(),
                output_root=Path(args.output_root).expanduser().resolve(),
                refined_list=Path(args.refined_list).expanduser().resolve(),
                per_city=args.per_city,
                seed=args.seed,
                preferred_beta=args.preferred_beta,
                beta_overrides=args.beta_order,
                split_filter=args.split_filter,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
        raise SystemExit(main())
