from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
                0.0, float(box_a[3]) - float(box_a[1])
        )
        area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
                0.0, float(box_b[3]) - float(box_b[1])
        )
        denom = area_a + area_b - inter_area + 1e-9
        return inter_area / denom


def extract_predictions_from_ultralytics(result_obj: Any) -> np.ndarray:
        boxes = getattr(result_obj, "boxes", None)
        if boxes is None or boxes.xyxy is None:
                return np.zeros((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls = boxes.cls.detach().cpu().numpy().astype(np.float32)

        out = np.zeros((xyxy.shape[0], 6), dtype=np.float32)
        out[:, 0] = cls
        out[:, 1:5] = xyxy
        out[:, 5] = conf
        return out


def mean_best_iou(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]) -> float:
        scores: List[float] = []
        for pred_arr, gt_arr in zip(preds, gts):
                if gt_arr.shape[0] == 0 or pred_arr.shape[0] == 0:
                        continue

                pred_boxes = pred_arr[:, 1:5]
                pred_cls = pred_arr[:, 0].astype(np.int32)

                for gt in gt_arr:
                        gt_cls = int(gt[0])
                        gt_box = gt[1:5]
                        matched = pred_boxes[pred_cls == gt_cls]
                        if matched.shape[0] == 0:
                                continue
                        best = 0.0
                        for box in matched:
                                best = max(best, box_iou_xyxy(gt_box, box))
                        scores.append(best)

        return float(np.mean(scores)) if scores else 0.0


def _compute_ap_for_class(
        cls_id: int,
        preds_per_image: Sequence[np.ndarray],
        gts_per_image: Sequence[np.ndarray],
        iou_thr: float,
) -> float:
        total_gt = 0
        detections: List[Tuple[int, float, np.ndarray]] = []
        gt_by_image: Dict[int, np.ndarray] = {}

        for img_idx, gt_arr in enumerate(gts_per_image):
                gt_cls = gt_arr[gt_arr[:, 0].astype(np.int32) == cls_id]
                gt_by_image[img_idx] = gt_cls
                total_gt += gt_cls.shape[0]

                pred_arr = preds_per_image[img_idx]
                pred_cls = pred_arr[pred_arr[:, 0].astype(np.int32) == cls_id]
                for row in pred_cls:
                        detections.append((img_idx, float(row[5]), row[1:5]))

        if total_gt == 0:
                return np.nan

        detections.sort(key=lambda x: x[1], reverse=True)

        matched_flags = {
                img_idx: np.zeros(gt_by_image[img_idx].shape[0], dtype=np.bool_)
                for img_idx in range(len(gts_per_image))
        }

        tp = np.zeros(len(detections), dtype=np.float32)
        fp = np.zeros(len(detections), dtype=np.float32)

        for det_idx, (img_idx, _conf, pred_box) in enumerate(detections):
                gt_arr = gt_by_image[img_idx]
                if gt_arr.shape[0] == 0:
                        fp[det_idx] = 1.0
                        continue

                ious = np.array(
                        [box_iou_xyxy(pred_box, gt[1:5]) for gt in gt_arr],
                        dtype=np.float32,
                )
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                if best_iou >= iou_thr and not matched_flags[img_idx][best_idx]:
                        tp[det_idx] = 1.0
                        matched_flags[img_idx][best_idx] = True
                else:
                        fp[det_idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        recall = tp_cum / max(total_gt, 1)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        for idx in range(mpre.size - 1, 0, -1):
                mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

        points = np.linspace(0.0, 1.0, 101)
        ap = np.mean(
                [np.max(mpre[mrec >= p]) if np.any(mrec >= p) else 0.0 for p in points]
        )
        return float(ap)


def compute_map(
        preds_per_image: Sequence[np.ndarray],
        gts_per_image: Sequence[np.ndarray],
        iou_thresholds: Sequence[float],
) -> Dict[str, Any]:
        classes = sorted({int(v) for arr in gts_per_image for v in arr[:, 0].tolist()})

        if not classes:
                return {
                        "classes": [],
                        "map_by_iou": {str(round(t, 2)): 0.0 for t in iou_thresholds},
                        "map": 0.0,
                }

        map_by_iou: Dict[str, float] = {}

        for thr in iou_thresholds:
                ap_values: List[float] = []
                for cls_id in classes:
                        ap = _compute_ap_for_class(
                                cls_id, preds_per_image, gts_per_image, float(thr)
                        )
                        if not np.isnan(ap):
                                ap_values.append(ap)
                map_by_iou[str(round(float(thr), 2))] = (
                        float(np.mean(ap_values)) if ap_values else 0.0
                )

        overall = float(np.mean(list(map_by_iou.values()))) if map_by_iou else 0.0
        return {
                "classes": classes,
                "map_by_iou": map_by_iou,
                "map": overall,
        }
