from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from pipeline.common.paths import resolve_from_project_root
from pipeline.data.yolo_dataset import OrganizedYoloDataset, collate_eval_batch
from pipeline.eval.metrics import (
        compute_map,
        extract_predictions_from_ultralytics,
        mean_iou_all_gt,
        mean_iou_matched,
        mean_iou_tp50,
)
from pipeline.eval.report import write_metrics_report
from pipeline.eval.visualize import save_visual_batch
from pipeline.eval import label_io
from pipeline.run.metrics_utils import compute_psnr, compute_ssim
from pipeline.run.summary_utils import aggregate_dataset_means, summarize_scalar_series
import pipeline.eval.metrics as eval_metrics
from collections import defaultdict
import csv
import json
from pipeline.models.detection_loader import (
        DetectionModelLoader,
        load_detection_inference_settings,
)
from pipeline.models.restoration_loader import RestorationModelLoader

ConfigDict = Dict[str, Any]

def _stem_from_path(value: Any, default_name: str) -> str:
        if value is None:
                return default_name
        return Path(str(value)).stem or default_name


def _unique_output_dir(base_dir: Path) -> Path:
        if not base_dir.exists():
                return base_dir

        suffix = 2
        while True:
                candidate = base_dir.with_name(f"{base_dir.name}_{suffix}")
                if not candidate.exists():
                        return candidate
                suffix += 1


def _prepare_output_dir(config: ConfigDict, run_name: Optional[str]) -> Path:
        paths_cfg = config.get("paths", {})
        restoration_cfg = config.get("restoration", {})
        detection_cfg = config.get("detection", {})
        out_root = resolve_from_project_root(
                config,
                str(paths_cfg.get("outputs_root", "outputs")),
        )

        checkpoint_name = _stem_from_path(
                restoration_cfg.get("checkpoint"), "restoration"
        )
        yolo_name = _stem_from_path(detection_cfg.get("weights"), "yolo")
        base_name = f"{checkpoint_name}_{yolo_name}"

        run_dir = _unique_output_dir(out_root / base_name)
        (run_dir / "visuals").mkdir(parents=True, exist_ok=True)
        return run_dir


def _to_uint8_rgb(images_float: torch.Tensor) -> List[np.ndarray]:
        arr = images_float.detach().cpu().permute(0, 2, 3, 1).numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return [arr[idx] for idx in range(arr.shape[0])]


def _cv2_interpolation(name: str) -> int:
        key = str(name).lower()
        if key == "nearest":
                return cv2.INTER_NEAREST
        if key == "area":
                return cv2.INTER_AREA
        if key == "cubic":
                return cv2.INTER_CUBIC
        if key == "lanczos":
                return cv2.INTER_LANCZOS4
        return cv2.INTER_LINEAR


def save_restored_original_resolution(
        restored_tensor: torch.Tensor,
        original_size: tuple[int, int],
        out_path: Path,
        interpolation: str = "lanczos",
) -> np.ndarray:
        arr = restored_tensor.detach().cpu().permute(1, 2, 0).numpy()
        rgb = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        width, height = original_size
        if rgb.shape[1] != width or rgb.shape[0] != height:
                rgb = cv2.resize(
                        rgb,
                        (int(width), int(height)),
                        interpolation=_cv2_interpolation(interpolation),
                )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return rgb


def _dataset_detection_imgsz(
        config: ConfigDict, dataset_name: str, default_imgsz: Any
) -> Any:
        dataset_cfg = (
                config.get("datasets", {})
                .get("entries", {})
                .get(dataset_name, {})
        )
        value = dataset_cfg.get("detection_imgsz")
        if value is None:
                value = config.get("detection", {}).get("imgsz", default_imgsz)
        if isinstance(value, int):
                return int(value)
        if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
        return int(value)


def _metric_bundle(
        preds: Sequence[np.ndarray], gts: Sequence[np.ndarray], primary: str
) -> Dict[str, float | str]:
        all_gt = mean_iou_all_gt(preds, gts)
        matched = mean_iou_matched(preds, gts)
        tp50 = mean_iou_tp50(preds, gts)
        primary_key = str(primary).lower()
        primary_value = {
                "all_gt": all_gt,
                "matched": matched,
                "tp50": tp50,
        }.get(primary_key, all_gt)
        return {
                "all_gt": all_gt,
                "matched": matched,
                "tp50": tp50,
                "primary": primary_key if primary_key in {"all_gt", "matched", "tp50"} else "all_gt",
                "primary_value": primary_value,
        }


def evaluate_pipeline(
        config: ConfigDict,
        selected_datasets: Optional[Sequence[str]] = None,
        max_images: Optional[int] = None,
        run_name: Optional[str] = None,
        dry_run: bool = False,
) -> Dict[str, Any]:
        runtime_cfg = config.get("runtime", {})
        vram_cfg = config.get("vram", {})
        eval_cfg = config.get("evaluation", {})

        batch_size = int(vram_cfg.get("batch_size", 1))
        num_workers = int(runtime_cfg.get("num_workers", 0))
        detector_input_mode = str(
                eval_cfg.get("detector_input_mode", "tensor")
        ).lower()
        if detector_input_mode not in {"file_path", "tensor"}:
                raise ValueError(
                        "evaluation.detector_input_mode must be 'file_path' or 'tensor'."
                )
        save_restored_full_resolution = bool(
                eval_cfg.get("save_restored_full_resolution", True)
        )
        if detector_input_mode == "file_path":
                save_restored_full_resolution = True
        restored_image_format = str(eval_cfg.get("restored_image_format", "png"))
        restored_image_format = restored_image_format.lstrip(".").lower() or "png"
        restore_output_interpolation = str(
                eval_cfg.get("restore_output_interpolation", "lanczos")
        )
        map_class_set = str(eval_cfg.get("map_class_set", "gt_only")).lower()
        iou_primary = str(eval_cfg.get("iou_primary", "all_gt")).lower()

        run_dir = _prepare_output_dir(config, run_name)

        dataset = OrganizedYoloDataset(
                config=config, selected_datasets=selected_datasets
        )
        # collect dataset names that will be processed
        try:
                dataset_names = sorted({rec["dataset"] for rec in dataset.records})
        except Exception:
                dataset_names = []
        if max_images is None:
                max_images = int(eval_cfg.get("max_images", 0))
                if max_images <= 0:
                        max_images = len(dataset)
        max_images = min(max_images, len(dataset))

        dataset_for_loader = dataset
        if max_images < len(dataset):
                dataset_for_loader = Subset(dataset, list(range(max_images)))

        dataloader = DataLoader(
                dataset_for_loader,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=bool(runtime_cfg.get("pin_memory", True)),
                collate_fn=collate_eval_batch,
        )

        restoration_runtime = RestorationModelLoader(config).load()
        detection_model = DetectionModelLoader(config).load()
        detection_kwargs = load_detection_inference_settings(config)
        detection_cfg = config.get("detection", {})
        restored_images_dir = run_dir / "restored_images"

        # Build id->name maps from config for visualization/label exports
        id2name_map: dict = {}
        datasets_cfg = config.get("datasets", {}).get("entries", {})
        for name, cfg in datasets_cfg.items():
                cm = cfg.get("class_map", {}) or {}
                # class_map may be name->id. We want id->name
                rev = {int(v): str(k) for k, v in dict(cm).items()}
                id2name_map[name] = rev

        if dry_run:
                # compute per-dataset sizes
                per_dataset_sizes = {}
                try:
                        for name in dataset_names:
                                per_dataset_sizes[name] = sum(
                                        1
                                        for r in dataset.records
                                        if r.get("dataset") == name
                                )
                except Exception:
                        per_dataset_sizes = {name: None for name in dataset_names}

                return {
                        "status": "dry_run",
                        "run_dir": str(run_dir),
                        "dataset_size": len(dataset),
                        "per_dataset_sizes": per_dataset_sizes,
                        "effective_eval_size": max_images,
                        "batch_size": batch_size,
                        "restoration_enabled": restoration_runtime.enabled,
                        "detector_input_mode": detector_input_mode,
                        "save_restored_full_resolution": save_restored_full_resolution,
                        "detection_weights": str(
                                detection_cfg.get("weights", "weights/yolo/yolo26n.pt")
                        ),
                        "detection_imgsz": detection_kwargs.get("imgsz"),
                        "datasets": dataset_names,
                }

        thresholds = eval_cfg.get(
                "map_iou_thresholds",
                [round(0.5 + 0.05 * idx, 2) for idx in range(10)],
        )
        thresholds = [float(v) for v in thresholds]

        max_visuals_cfg = eval_cfg.get("max_visuals", 100)
        visuals_dir = run_dir / "visuals"

        gt_collection: List[np.ndarray] = []
        baseline_preds: List[np.ndarray] = []
        restored_preds: List[np.ndarray] = []

        processed = 0
        visuals_saved = 0
        per_dataset_saved = defaultdict(int)
        per_dataset_rows: dict = defaultdict(list)
        per_dataset_collections: dict = defaultdict(
                lambda: {"gts": [], "baseline": [], "restored": []}
        )

        # build per-dataset max mapping
        default_max = 0
        per_dataset_max = {}

        if isinstance(max_visuals_cfg, dict):
                for k, v in max_visuals_cfg.items():
                        per_dataset_max[str(k)] = v if isinstance(v, dict) else int(v)
                # for any dataset not listed, default to 0 (no visuals)
        else:
                default_max = int(max_visuals_cfg)
                for name in dataset_names:
                        per_dataset_max[name] = default_max

        for batch in dataloader:
                if processed >= max_images:
                        break

                images = batch["restore_image_tensor"].to(restoration_runtime.device)
                depths = batch["restore_depth_tensor"].to(restoration_runtime.device)

                remaining = max_images - processed
                if images.shape[0] > remaining:
                        images = images[:remaining]
                        depths = depths[:remaining]
                        for key in (
                                "dataset",
                                "image_id",
                                "image_path",
                                "depth_path",
                                "label_path",
                                "original_width",
                                "original_height",
                                "gt_yolo",
                                "image_rgb_uint8",
                                "original_image_rgb_uint8",
                                "gt_xyxy",
                                "gt_xyxy_restore",
                                "gt_xyxy_original",
                        ):
                                batch[key] = batch[key][:remaining]

                with torch.no_grad():
                        restored_images = restoration_runtime.restore(images, depths)

                restored_tensor_cpu = restored_images.detach().cpu()
                restored_resized_np = _to_uint8_rgb(restored_images)

                restored_image_paths: List[str] = []
                restored_full_np: List[np.ndarray] = []
                for idx, img_id in enumerate(batch["image_id"]):
                        ds = batch["dataset"][idx]
                        out_path = (
                                restored_images_dir
                                / ds
                                / f"{img_id}.{restored_image_format}"
                        )
                        if save_restored_full_resolution:
                                restored_rgb = save_restored_original_resolution(
                                        restored_tensor_cpu[idx],
                                        (
                                                int(batch["original_width"][idx]),
                                                int(batch["original_height"][idx]),
                                        ),
                                        out_path,
                                        restore_output_interpolation,
                                )
                        else:
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                restored_rgb = restored_resized_np[idx]
                                cv2.imwrite(
                                        str(out_path),
                                        cv2.cvtColor(
                                                restored_rgb, cv2.COLOR_RGB2BGR
                                        ),
                                )
                        restored_image_paths.append(str(out_path))
                        restored_full_np.append(restored_rgb)

                if detector_input_mode == "file_path":
                        original_np = list(batch["original_image_rgb_uint8"])
                        restored_np = restored_full_np
                        baseline_pred_np = []
                        restored_pred_np = []
                        for idx in range(len(batch["image_id"])):
                                ds = batch["dataset"][idx]
                                kwargs = dict(detection_kwargs)
                                kwargs["imgsz"] = _dataset_detection_imgsz(
                                        config, ds, detection_kwargs.get("imgsz")
                                )
                                baseline_results = detection_model.predict(
                                        source=str(batch["image_path"][idx]), **kwargs
                                )
                                restored_results = detection_model.predict(
                                        source=restored_image_paths[idx], **kwargs
                                )
                                baseline_pred_np.append(
                                        extract_predictions_from_ultralytics(
                                                baseline_results[0]
                                        )
                                        if baseline_results
                                        else np.zeros((0, 6), dtype=np.float32)
                                )
                                restored_pred_np.append(
                                        extract_predictions_from_ultralytics(
                                                restored_results[0]
                                        )
                                        if restored_results
                                        else np.zeros((0, 6), dtype=np.float32)
                                )
                else:
                        original_np = _to_uint8_rgb(images)
                        restored_np = restored_resized_np
                        baseline_results = detection_model.predict(
                                source=original_np, **detection_kwargs
                        )
                        restored_results = detection_model.predict(
                                source=restored_np, **detection_kwargs
                        )

                        baseline_pred_np = [
                                extract_predictions_from_ultralytics(r)
                                for r in baseline_results
                        ]
                        restored_pred_np = [
                                extract_predictions_from_ultralytics(r)
                                for r in restored_results
                        ]

                gt_key = (
                        "gt_xyxy_original"
                        if detector_input_mode == "file_path"
                        else "gt_xyxy_restore"
                )
                gt_batch = [arr.astype(np.float32) for arr in batch[gt_key]]

                gt_collection.extend(gt_batch)
                baseline_preds.extend(baseline_pred_np)
                restored_preds.extend(restored_pred_np)

                for idx in range(len(batch["image_id"])):
                        dataset_name = batch["dataset"][idx]
                        per_dataset_collections[dataset_name]["gts"].append(
                                gt_batch[idx]
                        )
                        per_dataset_collections[dataset_name]["baseline"].append(
                                baseline_pred_np[idx]
                        )
                        per_dataset_collections[dataset_name]["restored"].append(
                                restored_pred_np[idx]
                        )

                saved_count, per_dataset_saved = save_visual_batch(
                        output_dir=visuals_dir,
                        datasets=batch["dataset"],
                        image_ids=batch["image_id"],
                        original_images=original_np,
                        restored_images=restored_np,
                        baseline_predictions=baseline_pred_np,
                        restored_predictions=restored_pred_np,
                        per_dataset_max=per_dataset_max,
                        per_dataset_saved=per_dataset_saved,
                        id2name_map=id2name_map,
                )
                visuals_saved += saved_count

                # compute per-image metrics (PSNR, SSIM, per-image mean best IOU)
                for idx in range(len(batch["image_id"])):
                        img_id = batch["image_id"][idx]
                        ds = batch["dataset"][idx]
                        orig = original_np[idx]
                        rest = restored_np[idx]
                        preds = restored_pred_np[idx]
                        baseline_preds_img = baseline_pred_np[idx]
                        gt = gt_batch[idx]

                        psnr_val = float(compute_psnr(orig, rest))
                        ssim_val = float(compute_ssim(orig, rest))

                        baseline_iou_img = _metric_bundle(
                                [baseline_preds_img], [gt], iou_primary
                        )
                        restored_iou_img = _metric_bundle([preds], [gt], iou_primary)
                        baseline_map50_img = compute_map(
                                [baseline_preds_img],
                                [gt],
                                [0.50],
                                class_set_mode=map_class_set,
                        )
                        restored_map50_img = compute_map(
                                [preds],
                                [gt],
                                [0.50],
                                class_set_mode=map_class_set,
                        )

                        # compute per-image detection precision/recall/F1 for baseline and restored
                        def _compute_prf(preds_arr, gts_arr, iou_thr=0.5):
                                # preds_arr: Nx6, gts_arr: Mx5 ([cls,x1,y1,x2,y2])
                                tp = 0
                                fp = 0
                                total_gt = (
                                        int(gts_arr.shape[0])
                                        if hasattr(gts_arr, "shape")
                                        else 0
                                )

                                # group by class
                                cls_set = set()
                                try:
                                        cls_set |= {
                                                int(v) for v in gts_arr[:, 0].tolist()
                                        }
                                except Exception:
                                        pass
                                try:
                                        cls_set |= {
                                                int(v) for v in preds_arr[:, 0].tolist()
                                        }
                                except Exception:
                                        pass

                                for cls in cls_set:
                                        gt_cls = gts_arr[
                                                gts_arr[:, 0].astype(int) == int(cls)
                                        ]
                                        pred_cls = preds_arr[
                                                preds_arr[:, 0].astype(int) == int(cls)
                                        ]
                                        if pred_cls.shape[0] == 0:
                                                continue
                                        # sort preds by confidence desc
                                        order = np.argsort(-pred_cls[:, 5])
                                        matched = (
                                                np.zeros(gt_cls.shape[0], dtype=bool)
                                                if gt_cls.shape[0] > 0
                                                else np.array([], dtype=bool)
                                        )
                                        for pi in order:
                                                pbox = pred_cls[pi, 1:5]
                                                if gt_cls.shape[0] == 0:
                                                        fp += 1
                                                        continue
                                                ious = np.array(
                                                        [
                                                                eval_metrics.box_iou_xyxy(
                                                                        pbox, g[1:5]
                                                                )
                                                                for g in gt_cls
                                                        ],
                                                        dtype=np.float32,
                                                )
                                                best_idx = int(np.argmax(ious))
                                                if (
                                                        ious[best_idx] >= iou_thr
                                                        and not matched[best_idx]
                                                ):
                                                        tp += 1
                                                        matched[best_idx] = True
                                                else:
                                                        fp += 1
                                fn = total_gt - tp
                                prec = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
                                rec = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
                                f1 = (
                                        2.0 * prec * rec / (prec + rec)
                                        if (prec + rec) > 0
                                        else 0.0
                                )
                                return (
                                        prec,
                                        rec,
                                        f1,
                                        int(total_gt),
                                        int(preds_arr.shape[0]),
                                )

                        (
                                baseline_prec,
                                baseline_rec,
                                baseline_f1,
                                num_gt_val,
                                baseline_num_preds,
                        ) = _compute_prf(
                                baseline_preds_img
                                if baseline_preds_img is not None
                                else np.zeros((0, 6)),
                                gt,
                        )
                        (
                                rest_prec,
                                rest_rec,
                                rest_f1,
                                _num_gt_val2,
                                restored_num_preds,
                        ) = _compute_prf(
                                preds if preds is not None else np.zeros((0, 6)), gt
                        )

                        per_dataset_rows[ds].append(
                                {
                                        "dataset": ds,
                                        "image_id": img_id,
                                        "image_path": batch["image_path"][idx],
                                        "restored_image_path": restored_image_paths[
                                                idx
                                        ],
                                        "original_width": int(
                                                batch["original_width"][idx]
                                        ),
                                        "original_height": int(
                                                batch["original_height"][idx]
                                        ),
                                        "num_gt": int(num_gt_val),
                                        "num_pred_original": int(baseline_num_preds),
                                        "num_pred_restored": int(restored_num_preds),
                                        "original_iou_all_gt": float(
                                                baseline_iou_img["all_gt"]
                                        ),
                                        "restored_iou_all_gt": float(
                                                restored_iou_img["all_gt"]
                                        ),
                                        "original_iou_matched": float(
                                                baseline_iou_img["matched"]
                                        ),
                                        "restored_iou_matched": float(
                                                restored_iou_img["matched"]
                                        ),
                                        "original_iou_tp50": float(
                                                baseline_iou_img["tp50"]
                                        ),
                                        "restored_iou_tp50": float(
                                                restored_iou_img["tp50"]
                                        ),
                                        "original_ap50": float(
                                                baseline_map50_img["map_by_iou"][
                                                        "0.5"
                                                ]
                                        ),
                                        "restored_ap50": float(
                                                restored_map50_img["map_by_iou"][
                                                        "0.5"
                                                ]
                                        ),
                                        "precision_original": float(baseline_prec),
                                        "recall_original": float(baseline_rec),
                                        "f1_original": float(baseline_f1),
                                        "precision_restored": float(rest_prec),
                                        "recall_restored": float(rest_rec),
                                        "f1_restored": float(rest_f1),
                                        "psnr": psnr_val,
                                        "ssim": ssim_val,
                                        "baseline_num_preds": int(baseline_num_preds),
                                        "restored_num_preds": int(restored_num_preds),
                                        "baseline_mean_iou_tp50": float(
                                                baseline_iou_img["tp50"]
                                        ),
                                        "restored_mean_iou_tp50": float(
                                                restored_iou_img["tp50"]
                                        ),
                                        "mean_iou_tp50": float(
                                                restored_iou_img["tp50"]
                                        ),
                                        "baseline_mean_iou": float(
                                                baseline_iou_img["matched"]
                                        ),
                                        "baseline_precision": float(baseline_prec),
                                        "baseline_recall": float(baseline_rec),
                                        "baseline_f1": float(baseline_f1),
                                        "restored_mean_iou": float(
                                                restored_iou_img["matched"]
                                        ),
                                        "mean_iou": float(
                                                restored_iou_img["matched"]
                                        ),
                                        "restored_precision": float(rest_prec),
                                        "restored_recall": float(rest_rec),
                                        "restored_f1": float(rest_f1),
                                }
                        )

                        # Export labels for both baseline and restored predictions
                        try:
                                labels_root = run_dir / "labels"
                                # YOLO txt
                                w = int(orig.shape[1])
                                h = int(orig.shape[0])
                                # yolo_txt original
                                yolo_orig_path = labels_root / "yolo_txt" / ds
                                yolo_orig_path = (
                                        yolo_orig_path / "original" / f"{img_id}.txt"
                                )
                                label_io.write_yolo_txt(
                                        baseline_preds_img
                                        if baseline_preds_img is not None
                                        else np.zeros((0, 6)),
                                        (w, h),
                                        yolo_orig_path,
                                )
                                # yolo_txt restored
                                yolo_rest_path = labels_root / "yolo_txt" / ds
                                yolo_rest_path = (
                                        yolo_rest_path / "restored" / f"{img_id}.txt"
                                )
                                label_io.write_yolo_txt(
                                        preds
                                        if preds is not None
                                        else np.zeros((0, 6)),
                                        (w, h),
                                        yolo_rest_path,
                                )

                                # VOC xml original
                                voc_orig_path = labels_root / "voc_xml" / ds
                                voc_orig_path = (
                                        voc_orig_path / "original" / f"{img_id}.xml"
                                )
                                label_io.write_voc_xml(
                                        baseline_preds_img
                                        if baseline_preds_img is not None
                                        else np.zeros((0, 6)),
                                        (w, h),
                                        voc_orig_path,
                                        Path(batch["image_path"][idx]).name,
                                        folder=ds,
                                        id2name=id2name_map.get(ds),
                                )

                                # VOC xml restored
                                voc_rest_path = labels_root / "voc_xml" / ds
                                voc_rest_path = (
                                        voc_rest_path / "restored" / f"{img_id}.xml"
                                )
                                label_io.write_voc_xml(
                                        preds
                                        if preds is not None
                                        else np.zeros((0, 6)),
                                        (w, h),
                                        voc_rest_path,
                                        Path(restored_image_paths[idx]).name,
                                        folder=ds,
                                        id2name=id2name_map.get(ds),
                                )
                        except Exception:
                                # don't fail the whole run for label export errors
                                pass

                processed += len(gt_batch)

        baseline_iou = _metric_bundle(baseline_preds, gt_collection, iou_primary)
        restored_iou = _metric_bundle(restored_preds, gt_collection, iou_primary)
        baseline_iou_tp50 = float(baseline_iou["tp50"])
        restored_iou_tp50 = float(restored_iou["tp50"])

        baseline_map_dict = compute_map(
                baseline_preds,
                gt_collection,
                thresholds,
                class_set_mode=map_class_set,
        )
        restored_map_dict = compute_map(
                restored_preds,
                gt_collection,
                thresholds,
                class_set_mode=map_class_set,
        )

        baseline_map = float(baseline_map_dict["map"])
        restored_map = float(restored_map_dict["map"])

        dataset_summaries = {}
        dataset_names = sorted(
                set(per_dataset_rows.keys()) | set(per_dataset_collections.keys())
        )
        for ds in dataset_names:
                rows = per_dataset_rows.get(ds, [])
                dataset_gts = per_dataset_collections[ds]["gts"]
                dataset_baseline_preds = per_dataset_collections[ds]["baseline"]
                dataset_restored_preds = per_dataset_collections[ds]["restored"]

                dataset_baseline_iou = _metric_bundle(
                        dataset_baseline_preds, dataset_gts, iou_primary
                )
                dataset_restored_iou = _metric_bundle(
                        dataset_restored_preds, dataset_gts, iou_primary
                )
                dataset_baseline_iou_tp50 = float(dataset_baseline_iou["tp50"])
                dataset_restored_iou_tp50 = float(dataset_restored_iou["tp50"])
                dataset_baseline_map_dict = compute_map(
                        dataset_baseline_preds,
                        dataset_gts,
                        thresholds,
                        class_set_mode=map_class_set,
                )
                dataset_restored_map_dict = compute_map(
                        dataset_restored_preds,
                        dataset_gts,
                        thresholds,
                        class_set_mode=map_class_set,
                )

                dataset_summaries[ds] = {
                        "psnr": summarize_scalar_series([row["psnr"] for row in rows]),
                        "ssim": summarize_scalar_series([row["ssim"] for row in rows]),
                        "mean_iou": summarize_scalar_series(
                                [row["mean_iou"] for row in rows]
                        ),
                        "mean_iou_tp50": summarize_scalar_series(
                                [row["mean_iou_tp50"] for row in rows]
                        ),
                        "baseline": {
                                "mean_iou_tp50": dataset_baseline_iou_tp50,
                                "mean_iou": float(dataset_baseline_iou["matched"]),
                                "iou": dataset_baseline_iou,
                                "map": float(dataset_baseline_map_dict["map"]),
                                "map_by_iou": dataset_baseline_map_dict["map_by_iou"],
                                "map_class_set": map_class_set,
                        },
                        "restored": {
                                "mean_iou_tp50": dataset_restored_iou_tp50,
                                "mean_iou": float(dataset_restored_iou["matched"]),
                                "iou": dataset_restored_iou,
                                "map": float(dataset_restored_map_dict["map"]),
                                "map_by_iou": dataset_restored_map_dict["map_by_iou"],
                                "map_class_set": map_class_set,
                        },
                        "improvement": {
                                "mean_iou_tp50": dataset_restored_iou_tp50
                                - dataset_baseline_iou_tp50,
                                "mean_iou": float(dataset_restored_iou["matched"])
                                - float(dataset_baseline_iou["matched"]),
                                "iou_all_gt": float(dataset_restored_iou["all_gt"])
                                - float(dataset_baseline_iou["all_gt"]),
                                "iou_matched": float(dataset_restored_iou["matched"])
                                - float(dataset_baseline_iou["matched"]),
                                "iou_tp50": float(dataset_restored_iou["tp50"])
                                - float(dataset_baseline_iou["tp50"]),
                                "map": float(dataset_restored_map_dict["map"])
                                - float(dataset_baseline_map_dict["map"]),
                        },
                }

        quality_summary = {
                "psnr": aggregate_dataset_means(dataset_summaries, "psnr"),
                "ssim": aggregate_dataset_means(dataset_summaries, "ssim"),
                "aggregation": "dataset_mean_then_equal_average",
        }

        report = {
                "run_dir": str(run_dir),
                "processed_images": processed,
                "batch_size": batch_size,
                "image_resolution": config.get("vram", {}).get(
                        "image_resolution", [640, 640]
                ),
                "run_name": run_name,
                "detector_input_mode": detector_input_mode,
                "save_restored_full_resolution": save_restored_full_resolution,
                "restored_image_format": restored_image_format,
                "restore_output_interpolation": restore_output_interpolation,
                "map_class_set": map_class_set,
                "iou_primary": iou_primary,
                "detection": {
                        "weights": str(
                                detection_cfg.get("weights", "weights/yolo/yolo26n.pt")
                        ),
                        "imgsz": detection_kwargs.get("imgsz"),
                        "conf_threshold": float(
                                detection_cfg.get("conf_threshold", 0.25)
                        ),
                        "nms_iou_threshold": float(
                                detection_cfg.get("nms_iou_threshold", 0.7)
                        ),
                        "max_det": int(detection_cfg.get("max_det", 300)),
                        "device": detection_kwargs.get("device"),
                },
                "map_iou_thresholds": thresholds,
                "baseline": {
                        "mean_iou_tp50": baseline_iou_tp50,
                        "mean_iou": float(baseline_iou["matched"]),
                        "iou": baseline_iou,
                        "map": baseline_map,
                        "map_by_iou": baseline_map_dict["map_by_iou"],
                        "map_class_set": map_class_set,
                },
                "restored": {
                        "mean_iou_tp50": restored_iou_tp50,
                        "mean_iou": float(restored_iou["matched"]),
                        "iou": restored_iou,
                        "map": restored_map,
                        "map_by_iou": restored_map_dict["map_by_iou"],
                        "map_class_set": map_class_set,
                },
                "improvement": {
                        "mean_iou_tp50": restored_iou_tp50 - baseline_iou_tp50,
                        "mean_iou": float(restored_iou["matched"])
                        - float(baseline_iou["matched"]),
                        "iou_all_gt": float(restored_iou["all_gt"])
                        - float(baseline_iou["all_gt"]),
                        "iou_matched": float(restored_iou["matched"])
                        - float(baseline_iou["matched"]),
                        "iou_tp50": float(restored_iou["tp50"])
                        - float(baseline_iou["tp50"]),
                        "map": restored_map - baseline_map,
                },
                "quality": quality_summary,
                "visuals_saved": visuals_saved,
        }

        report_paths = write_metrics_report(run_dir, report)
        report["report_files"] = report_paths

        # Write per-dataset per-image metrics and summaries
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        all_summary = {"__overall__": {"quality": quality_summary}}
        all_summary["__overall__"]["detection"] = {
                "baseline": {
                        "mean_iou_tp50": baseline_iou_tp50,
                        "mean_iou": float(baseline_iou["matched"]),
                        "iou": baseline_iou,
                        "map": baseline_map,
                        "map_class_set": map_class_set,
                },
                "restored": {
                        "mean_iou_tp50": restored_iou_tp50,
                        "mean_iou": float(restored_iou["matched"]),
                        "iou": restored_iou,
                        "map": restored_map,
                        "map_class_set": map_class_set,
                },
                "improvement": {
                        "mean_iou_tp50": restored_iou_tp50 - baseline_iou_tp50,
                        "mean_iou": float(restored_iou["matched"])
                        - float(baseline_iou["matched"]),
                        "iou_all_gt": float(restored_iou["all_gt"])
                        - float(baseline_iou["all_gt"]),
                        "iou_matched": float(restored_iou["matched"])
                        - float(baseline_iou["matched"]),
                        "iou_tp50": float(restored_iou["tp50"])
                        - float(baseline_iou["tp50"]),
                        "map": restored_map - baseline_map,
                },
        }

        for ds, rows in per_dataset_rows.items():
                ds_csv = metrics_dir / f"{ds}_per_image.csv"
                ds_json = metrics_dir / f"{ds}_summary.json"
                # write CSV
                if rows:
                        keys = list(rows[0].keys())
                else:
                        keys = [
                                "image_id",
                                "image_path",
                                "psnr",
                                "ssim",
                                "baseline_mean_iou_tp50",
                                "restored_mean_iou_tp50",
                                "mean_iou_tp50",
                                "baseline_mean_iou",
                                "restored_mean_iou",
                                "mean_iou",
                                "num_gt",
                                "baseline_num_preds",
                                "restored_num_preds",
                        ]

                with ds_csv.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=keys)
                        writer.writeheader()
                        for r in rows:
                                writer.writerow(r)

                # compute summary stats
                def _agg(field: str):
                        vals = [
                                float(r[field])
                                for r in rows
                                if r.get(field) is not None
                        ]
                        if not vals:
                                return {"mean": None, "std": None, "count": 0}
                        a = float(np.mean(vals))
                        s = float(np.std(vals))
                        return {"mean": a, "std": s, "count": len(vals)}

                summary = {
                        "psnr": _agg("psnr"),
                        "ssim": _agg("ssim"),
                        "mean_iou_tp50": _agg("mean_iou_tp50"),
                        "mean_iou_tp50_mean": _agg("mean_iou_tp50")["mean"],
                        "baseline_mean_iou_tp50": _agg("baseline_mean_iou_tp50"),
                        "restored_mean_iou_tp50": _agg("restored_mean_iou_tp50"),
                        "restored_mean_iou": _agg("restored_mean_iou")
                        if rows
                        and any(r.get("restored_mean_iou") is not None for r in rows)
                        else _agg("mean_iou"),
                        "baseline_precision": _agg("baseline_precision"),
                        "baseline_recall": _agg("baseline_recall"),
                        "baseline_f1": _agg("baseline_f1"),
                        "restored_precision": _agg("restored_precision"),
                        "restored_recall": _agg("restored_recall"),
                        "restored_f1": _agg("restored_f1"),
                        "baseline": dataset_summaries.get(ds, {}).get("baseline", {}),
                        "restored": dataset_summaries.get(ds, {}).get("restored", {}),
                        "improvement": dataset_summaries.get(ds, {}).get(
                                "improvement", {}
                        ),
                }

                with ds_json.open("w", encoding="utf-8") as handle:
                        json.dump(summary, handle, indent=2)

                all_summary[ds] = summary

        # write overall summary
        overall_path = metrics_dir / "all_datasets_summary.json"
        with overall_path.open("w", encoding="utf-8") as handle:
                json.dump(all_summary, handle, indent=2)

        # Aggregate detection precision/recall/F1 across all datasets
        all_rows = []
        for rows in per_dataset_rows.values():
                all_rows.extend(rows)

        def _agg_list(field: str):
                vals = [float(r[field]) for r in all_rows if r.get(field) is not None]
                if not vals:
                        return {"mean": None, "std": None, "count": 0}
                return {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "count": len(vals),
                }

        detection_agg = {
                "baseline_precision": _agg_list("baseline_precision"),
                "baseline_recall": _agg_list("baseline_recall"),
                "baseline_f1": _agg_list("baseline_f1"),
                "restored_precision": _agg_list("restored_precision"),
                "restored_recall": _agg_list("restored_recall"),
                "restored_f1": _agg_list("restored_f1"),
        }

        # attach AP per class from full-run compute_map results
        try:
                report["baseline"]["ap_per_class"] = baseline_map_dict.get(
                        "ap_per_class", {}
                )
                report["restored"]["ap_per_class"] = restored_map_dict.get(
                        "ap_per_class", {}
                )
        except Exception:
                pass

        report["detection_aggregate"] = detection_agg
        report["metrics_dir"] = str(metrics_dir)

        # rewrite the top-level metrics.json/metrics.csv with enriched report
        try:
                report_paths = write_metrics_report(run_dir, report)
                report["report_files"] = report_paths
        except Exception:
                pass

        return report
