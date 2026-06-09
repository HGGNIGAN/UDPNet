from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from pipeline.common.paths import resolve_from_project_root
from pipeline.data.yolo_dataset import OrganizedYoloDataset, collate_eval_batch
from pipeline.eval.metrics import (
        compute_map,
        extract_predictions_from_ultralytics,
        mean_best_iou,
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
                        "detection_weights": str(
                                detection_cfg.get("weights", "weights/yolo/yolo26n.pt")
                        ),
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

                images = batch["image_tensor"].to(restoration_runtime.device)
                depths = batch["depth_tensor"].to(restoration_runtime.device)

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
                                "image_rgb_uint8",
                                "gt_xyxy",
                        ):
                                batch[key] = batch[key][:remaining]

                with torch.no_grad():
                        restored_images = restoration_runtime.restore(images, depths)

                original_np = _to_uint8_rgb(images)
                restored_np = _to_uint8_rgb(restored_images)

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

                gt_batch = [arr.astype(np.float32) for arr in batch["gt_xyxy"]]

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
                        gt = batch["gt_xyxy"][idx]

                        psnr_val = float(compute_psnr(orig, rest))
                        ssim_val = float(compute_ssim(orig, rest))

                        def _per_image_mean_best_iou(preds_arr, gts_arr):
                                scores = []
                                if gts_arr.shape[0] and preds_arr.shape[0]:
                                        pred_boxes = preds_arr[:, 1:5]
                                        pred_cls = preds_arr[:, 0].astype(int)
                                        for gt_row in gts_arr:
                                                gt_cls = int(gt_row[0])
                                                gt_box = gt_row[1:5]
                                                matched = pred_boxes[pred_cls == gt_cls]
                                                if matched.shape[0] == 0:
                                                        continue
                                                best = 0.0
                                                for box in matched:
                                                        best = max(
                                                                best,
                                                                eval_metrics.box_iou_xyxy(
                                                                        gt_box, box
                                                                ),
                                                        )
                                                scores.append(best)
                                return float(np.mean(scores)) if scores else 0.0

                        baseline_mean_iou_img = _per_image_mean_best_iou(
                                baseline_preds_img, gt
                        )
                        mean_iou_img = _per_image_mean_best_iou(preds, gt)
                        baseline_mean_iou_tp50_img = mean_iou_tp50(
                                [baseline_preds_img], [gt]
                        )
                        restored_mean_iou_tp50_img = mean_iou_tp50([preds], [gt])

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
                                        "image_id": img_id,
                                        "image_path": batch["image_path"][idx],
                                        # restoration quality
                                        "psnr": psnr_val,
                                        "ssim": ssim_val,
                                        # counts
                                        "num_gt": int(num_gt_val),
                                        "baseline_num_preds": int(baseline_num_preds),
                                        "restored_num_preds": int(restored_num_preds),
                                        # detection quality
                                        "baseline_mean_iou_tp50": float(
                                                baseline_mean_iou_tp50_img
                                        ),
                                        "restored_mean_iou_tp50": float(
                                                restored_mean_iou_tp50_img
                                        ),
                                        "mean_iou_tp50": float(
                                                restored_mean_iou_tp50_img
                                        ),
                                        "baseline_mean_iou": float(
                                                baseline_mean_iou_img
                                        ),
                                        "baseline_precision": float(baseline_prec),
                                        "baseline_recall": float(baseline_rec),
                                        "baseline_f1": float(baseline_f1),
                                        "restored_mean_iou": float(mean_iou_img),
                                        # keep legacy key for compatibility
                                        "mean_iou": float(mean_iou_img),
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
                                        f"{img_id}.jpg",
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
                                        f"{img_id}.jpg",
                                        folder=ds,
                                        id2name=id2name_map.get(ds),
                                )
                        except Exception:
                                # don't fail the whole run for label export errors
                                pass

                processed += len(gt_batch)

        baseline_iou = mean_best_iou(baseline_preds, gt_collection)
        restored_iou = mean_best_iou(restored_preds, gt_collection)
        baseline_iou_tp50 = mean_iou_tp50(baseline_preds, gt_collection)
        restored_iou_tp50 = mean_iou_tp50(restored_preds, gt_collection)

        baseline_map_dict = compute_map(baseline_preds, gt_collection, thresholds)
        restored_map_dict = compute_map(restored_preds, gt_collection, thresholds)

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

                dataset_baseline_iou = mean_best_iou(
                        dataset_baseline_preds, dataset_gts
                )
                dataset_restored_iou = mean_best_iou(
                        dataset_restored_preds, dataset_gts
                )
                dataset_baseline_iou_tp50 = mean_iou_tp50(
                        dataset_baseline_preds, dataset_gts
                )
                dataset_restored_iou_tp50 = mean_iou_tp50(
                        dataset_restored_preds, dataset_gts
                )
                dataset_baseline_map_dict = compute_map(
                        dataset_baseline_preds, dataset_gts, thresholds
                )
                dataset_restored_map_dict = compute_map(
                        dataset_restored_preds, dataset_gts, thresholds
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
                                "mean_iou": dataset_baseline_iou,
                                "map": float(dataset_baseline_map_dict["map"]),
                                "map_by_iou": dataset_baseline_map_dict["map_by_iou"],
                        },
                        "restored": {
                                "mean_iou_tp50": dataset_restored_iou_tp50,
                                "mean_iou": dataset_restored_iou,
                                "map": float(dataset_restored_map_dict["map"]),
                                "map_by_iou": dataset_restored_map_dict["map_by_iou"],
                        },
                        "improvement": {
                                "mean_iou_tp50": dataset_restored_iou_tp50
                                - dataset_baseline_iou_tp50,
                                "mean_iou": dataset_restored_iou - dataset_baseline_iou,
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
                "detection": {
                        "weights": str(
                                detection_cfg.get("weights", "weights/yolo/yolo26n.pt")
                        ),
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
                        "mean_iou": baseline_iou,
                        "map": baseline_map,
                        "map_by_iou": baseline_map_dict["map_by_iou"],
                },
                "restored": {
                        "mean_iou_tp50": restored_iou_tp50,
                        "mean_iou": restored_iou,
                        "map": restored_map,
                        "map_by_iou": restored_map_dict["map_by_iou"],
                },
                "improvement": {
                        "mean_iou_tp50": restored_iou_tp50 - baseline_iou_tp50,
                        "mean_iou": restored_iou - baseline_iou,
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
                        "mean_iou": baseline_iou,
                        "map": baseline_map,
                },
                "restored": {
                        "mean_iou_tp50": restored_iou_tp50,
                        "mean_iou": restored_iou,
                        "map": restored_map,
                },
                "improvement": {
                        "mean_iou_tp50": restored_iou_tp50 - baseline_iou_tp50,
                        "mean_iou": restored_iou - baseline_iou,
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
