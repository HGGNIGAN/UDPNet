from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def _draw_boxes(
        image_rgb: np.ndarray, pred_arr: np.ndarray, id2name: dict | None = None
) -> np.ndarray:
        canvas = image_rgb.copy()
        for row in pred_arr:
                cls_id = int(row[0])
                x1, y1, x2, y2 = [int(v) for v in row[1:5]]
                conf = float(row[5])
                label = (
                        id2name.get(cls_id)
                        if (id2name and cls_id in id2name)
                        else f"c{cls_id}"
                )
                text = f"{label}:{conf:.2f}"
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 80, 0), 2)
                cv2.putText(
                        canvas,
                        text,
                        (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 80, 0),
                        1,
                        cv2.LINE_AA,
                )
        return canvas


def save_separate_visuals(
        output_dir: Path,
        dataset: str,
        image_id: str,
        original_rgb: np.ndarray,
        restored_rgb: np.ndarray,
        baseline_predictions: np.ndarray,
        restored_predictions: np.ndarray,
        id2name_map: dict | None = None,
) -> None:
        """Save four separate images: original, restored, and dual detections.

        Directory layout: <output_dir>/<dataset>/{original,restored,original_detection,restored_detection}/
        Filenames: <image_id>.jpg
        """
        base = output_dir / dataset

        orig_dir = base / "original"
        rest_dir = base / "restored"
        orig_det_dir = base / "original_detection"
        rest_det_dir = base / "restored_detection"

        # Draw detections on original and restored
        # resolve id2name for this dataset
        local_id2name = None
        if id2name_map and dataset in id2name_map:
                local_id2name = id2name_map.get(dataset)
        orig_det_vis = _draw_boxes(original_rgb, baseline_predictions, local_id2name)
        rest_det_vis = _draw_boxes(restored_rgb, restored_predictions, local_id2name)

        # convert to BGR for cv2
        orig_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        rest_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        orig_det_bgr = cv2.cvtColor(orig_det_vis, cv2.COLOR_RGB2BGR)
        rest_det_bgr = cv2.cvtColor(rest_det_vis, cv2.COLOR_RGB2BGR)

        # Create all directories
        orig_dir.mkdir(parents=True, exist_ok=True)
        rest_dir.mkdir(parents=True, exist_ok=True)
        orig_det_dir.mkdir(parents=True, exist_ok=True)
        rest_det_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{image_id}.jpg"
        cv2.imwrite(str(orig_dir / fname), orig_bgr)
        cv2.imwrite(str(rest_dir / fname), rest_bgr)
        cv2.imwrite(str(orig_det_dir / fname), orig_det_bgr)
        cv2.imwrite(str(rest_det_dir / fname), rest_det_bgr)


def save_visual_batch(
        output_dir: Path,
        datasets: Sequence[str],
        image_ids: Sequence[str],
        original_images: Sequence[np.ndarray],
        restored_images: Sequence[np.ndarray],
        baseline_predictions: Sequence[np.ndarray],
        restored_predictions: Sequence[np.ndarray],
        per_dataset_max: dict,
        per_dataset_saved: dict,
        id2name_map: dict | None = None,
) -> tuple[int, dict]:
        """Save visuals per image into dataset-specific subdirs.

        Saves: original, restored, original_detection (baseline YOLO), restored_detection (restored YOLO).
        Returns (saved_count, updated_per_dataset_saved).
        """
        saved = 0

        for idx, image_id in enumerate(image_ids):
                dataset = datasets[idx]
                max_entry = per_dataset_max.get(dataset, 0)
                if isinstance(max_entry, dict):
                        max_for_dataset = sum(int(v) for v in max_entry.values())
                else:
                        max_for_dataset = int(max_entry)

                already = int(per_dataset_saved.get(dataset, 0))
                if max_for_dataset > 0 and already >= max_for_dataset:
                        continue

                try:
                        save_separate_visuals(
                                output_dir,
                                dataset,
                                image_id,
                                original_images[idx],
                                restored_images[idx],
                                baseline_predictions[idx],
                                restored_predictions[idx],
                                id2name_map=id2name_map,
                        )
                        per_dataset_saved[dataset] = already + 1
                        saved += 1
                except Exception:
                        continue

        return saved, per_dataset_saved
