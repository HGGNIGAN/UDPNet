from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.common.paths import resolve_from_project_root

ConfigDict = Dict[str, Any]


def _load_yolo_gt(label_path: Path) -> List[List[float]]:
        boxes: List[List[float]] = []
        if not label_path.exists():
                return boxes

        with label_path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                        line = raw.strip()
                        if not line:
                                continue
                        parts = line.split()
                        if len(parts) < 5:
                                continue
                        try:
                                class_id = int(float(parts[0]))
                                cx = float(parts[1])
                                cy = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                        except ValueError:
                                continue
                        boxes.append([class_id, cx, cy, width, height])

        return boxes


def _yolo_to_xyxy_abs(
        yolo_boxes: Sequence[Sequence[float]], w: int, h: int
) -> np.ndarray:
        if not yolo_boxes:
                return np.zeros((0, 6), dtype=np.float32)

        result = np.zeros((len(yolo_boxes), 6), dtype=np.float32)
        for idx, item in enumerate(yolo_boxes):
                class_id, cx, cy, bw, bh = item
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                result[idx] = [class_id, x1, y1, x2, y2, 1.0]

        return result


class OrganizedYoloDataset(Dataset):
        def __init__(
                self,
                config: ConfigDict,
                selected_datasets: Optional[Sequence[str]] = None,
        ) -> None:
                super().__init__()
                self.config = config

                datasets_cfg = config.get("datasets", {})
                entries = datasets_cfg.get("entries", {})
                active = (
                        list(selected_datasets)
                        if selected_datasets
                        else list(datasets_cfg.get("active", entries.keys()))
                )

                self.image_h = int(
                        config.get("vram", {}).get("image_resolution", [640, 640])[0]
                )
                self.image_w = int(
                        config.get("vram", {}).get("image_resolution", [640, 640])[1]
                )
                organized_root = resolve_from_project_root(
                        config,
                        str(
                                config.get("paths", {}).get(
                                        "organized_root", "Datasets/organized"
                                )
                        ),
                )

                self.records: List[Dict[str, str]] = []
                for dataset_name in active:
                        pairs_yolo = (
                                organized_root
                                / dataset_name
                                / "manifests"
                                / "pairs_yolo.csv"
                        )
                        if not pairs_yolo.exists():
                                raise FileNotFoundError(
                                        f"YOLO-normalized pairs manifest missing: {pairs_yolo}. Run step4 first."
                                )

                        with pairs_yolo.open(
                                "r", encoding="utf-8", newline=""
                        ) as handle:
                                rows = list(csv.DictReader(handle))
                        for row in rows:
                                row["dataset"] = dataset_name
                                self.records.append(row)

        def __len__(self) -> int:
                return len(self.records)

        def __getitem__(self, index: int) -> Dict[str, Any]:
                row = self.records[index]
                dataset_name = row["dataset"]
                image_id = row["image_id"]

                image_path = Path(row["image_path"])
                norm_label_path = Path(row["normalized_label_path"])
                depth_path = image_path.parents[1] / "DepthMaps" / image_path.name

                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                        raise FileNotFoundError(f"Could not read image: {image_path}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(
                        image_rgb,
                        (self.image_w, self.image_h),
                        interpolation=cv2.INTER_LINEAR,
                )

                depth_gray = None
                if depth_path.exists():
                        depth_gray = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
                if depth_gray is None:
                        depth_gray = np.zeros(
                                (self.image_h, self.image_w), dtype=np.uint8
                        )
                else:
                        depth_gray = cv2.resize(
                                depth_gray,
                                (self.image_w, self.image_h),
                                interpolation=cv2.INTER_LINEAR,
                        )

                image_tensor = (
                        torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                )
                depth_tensor = torch.from_numpy(depth_gray).unsqueeze(0).float() / 255.0

                gt_yolo = _load_yolo_gt(norm_label_path)
                gt_xyxy = _yolo_to_xyxy_abs(gt_yolo, self.image_w, self.image_h)

                return {
                        "dataset": dataset_name,
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "depth_path": str(depth_path),
                        "label_path": str(norm_label_path),
                        "image_tensor": image_tensor,
                        "depth_tensor": depth_tensor,
                        "image_rgb_uint8": image_rgb,
                        "gt_xyxy": gt_xyxy,
                }


def collate_eval_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
                "dataset": [item["dataset"] for item in items],
                "image_id": [item["image_id"] for item in items],
                "image_path": [item["image_path"] for item in items],
                "depth_path": [item["depth_path"] for item in items],
                "label_path": [item["label_path"] for item in items],
                "image_tensor": torch.stack(
                        [item["image_tensor"] for item in items], dim=0
                ),
                "depth_tensor": torch.stack(
                        [item["depth_tensor"] for item in items], dim=0
                ),
                "image_rgb_uint8": [item["image_rgb_uint8"] for item in items],
                "gt_xyxy": [item["gt_xyxy"] for item in items],
        }
