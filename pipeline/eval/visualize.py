from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


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


def save_triptych(
        save_path: Path,
        original_rgb: np.ndarray,
        restored_rgb: np.ndarray,
        restored_predictions: np.ndarray,
) -> None:
        det_vis = _draw_boxes(restored_rgb, restored_predictions)

        merged = np.concatenate([original_rgb, restored_rgb, det_vis], axis=1)
        merged_bgr = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), merged_bgr)


def save_visual_batch(
        output_dir: Path,
        image_ids: Sequence[str],
        original_images: Sequence[np.ndarray],
        restored_images: Sequence[np.ndarray],
        restored_predictions: Sequence[np.ndarray],
        start_index: int,
        max_save: int,
) -> int:
        saved = 0
        for idx, image_id in enumerate(image_ids):
                global_idx = start_index + idx
                if global_idx >= max_save:
                        break
                save_path = output_dir / f"{global_idx:06d}_{image_id}.jpg"
                save_triptych(
                        save_path,
                        original_images[idx],
                        restored_images[idx],
                        restored_predictions[idx],
                )
                saved += 1
        return saved
