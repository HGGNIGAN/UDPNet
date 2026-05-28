from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


_DAWN_HAZARDS = (
        "dusttornado",
        "rain_storm",
        "sand_storm",
        "snow_storm",
        "foggy",
        "haze",
        "mist",
)


def _extract_dawn_hazard(text: str) -> str:
        normalized = (text or "").lower().replace("-", "_")
        for hazard in _DAWN_HAZARDS:
                if hazard in normalized:
                        return hazard
        return "unknown"


def _draw_boxes(image_rgb: np.ndarray, pred_arr: np.ndarray) -> np.ndarray:
        canvas = image_rgb.copy()
        for row in pred_arr:
                cls_id = int(row[0])
                x1, y1, x2, y2 = [int(v) for v in row[1:5]]
                conf = float(row[5])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 80, 0), 2)
                cv2.putText(
                        canvas,
                        f"c{cls_id}:{conf:.2f}",
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
        restored_predictions: np.ndarray,
) -> None:
        """Save three separate images for original, restored and detection overlay.

        Directory layout: <output_dir>/<dataset>/{original,restored,detection}/
        Filenames: <image_id>.jpg
        """
        # For DAWN, group by hazard token parsed from image_id (token before first underscore)
        base = output_dir / dataset
        if dataset == "DAWN":
                base = base / _extract_dawn_hazard(image_id)

        orig_dir = base / "original"
        rest_dir = base / "restored"
        det_dir = base / "detection"

        det_vis = _draw_boxes(restored_rgb, restored_predictions)

        # convert to BGR for cv2
        orig_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        rest_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        det_bgr = cv2.cvtColor(det_vis, cv2.COLOR_RGB2BGR)

        orig_dir.mkdir(parents=True, exist_ok=True)
        rest_dir.mkdir(parents=True, exist_ok=True)
        det_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{image_id}.jpg"
        cv2.imwrite(str(orig_dir / fname), orig_bgr)
        cv2.imwrite(str(rest_dir / fname), rest_bgr)
        cv2.imwrite(str(det_dir / fname), det_bgr)


def save_visual_batch(
        output_dir: Path,
        datasets: Sequence[str],
        image_ids: Sequence[str],
        original_images: Sequence[np.ndarray],
        restored_images: Sequence[np.ndarray],
        restored_predictions: Sequence[np.ndarray],
        per_dataset_max: dict,
        per_dataset_saved: dict,
) -> tuple[int, dict]:
        """Save visuals per image into dataset-specific subdirs.

        Returns (saved_count, updated_per_dataset_saved).
        """
        saved = 0
        for idx, image_id in enumerate(image_ids):
                dataset = datasets[idx]
                max_for_dataset = int(per_dataset_max.get(dataset, 0))
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
                                restored_predictions[idx],
                        )
                        per_dataset_saved[dataset] = already + 1
                        saved += 1
                except Exception:
                        # continue on individual save errors
                        continue

        return saved, per_dataset_saved
