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
                        per_dataset_max[str(k)] = int(v)
                # for any dataset not listed, default to 0 (no visuals)
        else:
                default_max = int(max_visuals_cfg)
                for name in dataset_names:
                        per_dataset_max[name] = default_max

        # If DAWN has a scalar max > 0, split evenly across known hazards
        if "DAWN" in per_dataset_max:
                v = per_dataset_max["DAWN"]
                try:
                        v_int = int(v)
                except Exception:
                        v_int = 0
                if v_int > 0:
                        num_h = len(_DAWN_HAZARDS)
                        base = v_int // num_h
                        rem = v_int % num_h
                        per_dataset_max["DAWN"] = {
                                hazard: base + (1 if idx < rem else 0)
                                for idx, hazard in enumerate(_DAWN_HAZARDS)
                        }

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

                        # per-image mean best iou (restored predictions)
                        per_img_scores = []
                        if gt.shape[0] and preds.shape[0]:
                                pred_boxes = preds[:, 1:5]
                                pred_cls = preds[:, 0].astype(int)
                                for gt_row in gt:
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
                                        per_img_scores.append(best)
                        mean_iou_img = (
                                float(np.mean(per_img_scores))
                                if per_img_scores
                                else 0.0
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
                                        "baseline_mean_iou": 0.0,
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
                                # dataset-level folder (DAWN gets hazard added inside write paths below)
                                hazard = (
                                        _extract_dawn_hazard(img_id)
                                        if ds == "DAWN"
                                        else None
                                )
                                w = int(orig.shape[1])
                                h = int(orig.shape[0])
                                # yolo_txt original
                                yolo_orig_path = labels_root / "yolo_txt" / ds
                                if hazard:
                                        yolo_orig_path = yolo_orig_path / hazard
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
                                if hazard:
                                        yolo_rest_path = yolo_rest_path / hazard
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
                                if hazard:
                                        voc_orig_path = voc_orig_path / hazard
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
                                if hazard:
                                        voc_rest_path = voc_rest_path / hazard
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
                        "baseline": {
                                "mean_iou": dataset_baseline_iou,
                                "map": float(dataset_baseline_map_dict["map"]),
                                "map_by_iou": dataset_baseline_map_dict["map_by_iou"],
                        },
                        "restored": {
                                "mean_iou": dataset_restored_iou,
                                "map": float(dataset_restored_map_dict["map"]),
                                "map_by_iou": dataset_restored_map_dict["map_by_iou"],
                        },
                        "improvement": {
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
                        "mean_iou": baseline_iou,
                        "map": baseline_map,
                        "map_by_iou": baseline_map_dict["map_by_iou"],
                },
                "restored": {
                        "mean_iou": restored_iou,
                        "map": restored_map,
                        "map_by_iou": restored_map_dict["map_by_iou"],
                },
                "improvement": {
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
                        "mean_iou": baseline_iou,
                        "map": baseline_map,
                },
                "restored": {
                        "mean_iou": restored_iou,
                        "map": restored_map,
                },
                "improvement": {
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
                                "mean_iou",
                                "num_gt",
                                "num_preds",
                        ]

                with ds_csv.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=keys)
                        writer.writeheader()
                        for r in rows:
                                writer.writerow(r)

                # DAWN hazard grouping
                if ds == "DAWN":
                        hazards = list(_DAWN_HAZARDS)
                        hazard_groups = {h: [] for h in hazards}
                        hazard_groups["unknown"] = []
                        for r in rows:
                                token = _extract_dawn_hazard(str(r.get("image_id", "")))
                                hazard_groups[token].append(r)
                        # write per-hazard summaries
                        dawn_dir = metrics_dir / "DAWN"
                        dawn_dir.mkdir(parents=True, exist_ok=True)
                        for h, hrows in hazard_groups.items():
                                h_csv = dawn_dir / f"{h}_per_image.csv"
                                with h_csv.open(
                                        "w", encoding="utf-8", newline=""
                                ) as handle:
                                        if hrows:
                                                hk = list(hrows[0].keys())
                                        else:
                                                hk = keys
                                        writer = csv.DictWriter(handle, fieldnames=hk)
                                        writer.writeheader()
                                        for r in hrows:
                                                writer.writerow(r)

                                        # build per-hazard detection collections for mAP
                                        # match by positional order between rows and per_dataset_collections["DAWN"] entries
                                        gts_for_h = []
                                        baseline_for_h = []
                                        restored_for_h = []
                                        # We assume per_dataset_collections[ds] items are in same order as rows
                                        all_rows = [r["image_id"] for r in rows]
                                        # map image_id -> index in rows
                                        id_to_idx = {
                                                rid: idx
                                                for idx, rid in enumerate(all_rows)
                                        }
                                        dataset_gts = per_dataset_collections.get(
                                                "DAWN", {}
                                        ).get("gts", [])
                                        dataset_baseline_preds = (
                                                per_dataset_collections.get(
                                                        "DAWN", {}
                                                ).get("baseline", [])
                                        )
                                        dataset_restored_preds = (
                                                per_dataset_collections.get(
                                                        "DAWN", {}
                                                ).get("restored", [])
                                        )
                                        for r in hrows:
                                                iid = r.get("image_id")
                                                if iid in id_to_idx:
                                                        idx = id_to_idx[iid]
                                                        if idx < len(dataset_gts):
                                                                gts_for_h.append(
                                                                        dataset_gts[idx]
                                                                )
                                                                baseline_for_h.append(
                                                                        dataset_baseline_preds[
                                                                                idx
                                                                        ]
                                                                )
                                                                restored_for_h.append(
                                                                        dataset_restored_preds[
                                                                                idx
                                                                        ]
                                                                )

                                        # compute maps for this hazard
                                        try:
                                                h_baseline_map = compute_map(
                                                        baseline_for_h,
                                                        gts_for_h,
                                                        thresholds,
                                                )
                                                h_restored_map = compute_map(
                                                        restored_for_h,
                                                        gts_for_h,
                                                        thresholds,
                                                )
                                        except Exception:
                                                h_baseline_map = {
                                                        "map": 0.0,
                                                        "map_by_iou": {},
                                                        "ap_per_class": {},
                                                }
                                                h_restored_map = {
                                                        "map": 0.0,
                                                        "map_by_iou": {},
                                                        "ap_per_class": {},
                                                }

                                        # summary
                                def _agg_h(field: str):
                                        vals = [
                                                float(rr[field])
                                                for rr in hrows
                                                if rr.get(field) is not None
                                        ]
                                        if not vals:
                                                return {
                                                        "mean": None,
                                                        "std": None,
                                                        "count": 0,
                                                }
                                        return {
                                                "mean": float(np.mean(vals)),
                                                "std": float(np.std(vals)),
                                                "count": len(vals),
                                        }

                                h_summary = {
                                        "psnr": _agg_h("psnr"),
                                        "ssim": _agg_h("ssim"),
                                        "restored_mean_iou": _agg_h(
                                                "restored_mean_iou"
                                        ),
                                        "baseline_precision": _agg_h(
                                                "baseline_precision"
                                        ),
                                        "baseline_recall": _agg_h("baseline_recall"),
                                        "baseline_f1": _agg_h("baseline_f1"),
                                        "restored_precision": _agg_h(
                                                "restored_precision"
                                        ),
                                        "restored_recall": _agg_h("restored_recall"),
                                        "restored_f1": _agg_h("restored_f1"),
                                        "baseline_map": h_baseline_map,
                                        "restored_map": h_restored_map,
                                }
                                with (dawn_dir / f"{h}_summary.json").open(
                                        "w", encoding="utf-8"
                                ) as handle:
                                        json.dump(h_summary, handle, indent=2)

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

        # Write per-hazard AP-by-class CSVs for DAWN
        if "DAWN" in per_dataset_collections:
                dawn_dir = metrics_dir / "DAWN"
                for h in list(_DAWN_HAZARDS) + ["unknown"]:
                        h_json = dawn_dir / f"{h}_summary.json"
                        if not h_json.exists():
                                continue
                        # try load the JSON to get ap dicts
                        try:
                                with h_json.open("r", encoding="utf-8") as handle:
                                        hsum = json.load(handle)
                        except Exception:
                                continue

                        baseline_map_h = hsum.get("baseline_map", {}) or hsum.get(
                                "baseline", {}
                        )
                        restored_map_h = hsum.get("restored_map", {}) or hsum.get(
                                "restored", {}
                        )
                        # extract ap_per_class if present
                        baseline_ap = (
                                baseline_map_h.get("ap_per_class", {})
                                if isinstance(baseline_map_h, dict)
                                else {}
                        )
                        restored_ap = (
                                restored_map_h.get("ap_per_class", {})
                                if isinstance(restored_map_h, dict)
                                else {}
                        )

                        if not baseline_ap and not restored_ap:
                                continue

                        ap_csv = dawn_dir / f"{h}_ap_by_class.csv"
                        with ap_csv.open("w", encoding="utf-8", newline="") as handle:
                                writer = csv.DictWriter(
                                        handle,
                                        fieldnames=[
                                                "class_id",
                                                "class_name",
                                                "baseline_ap",
                                                "restored_ap",
                                        ],
                                )
                                writer.writeheader()
                                # union of keys
                                keys = sorted(
                                        {
                                                int(k)
                                                for k in list(baseline_ap.keys())
                                                + list(restored_ap.keys())
                                        }
                                )
                                for k in keys:
                                        name = None
                                        # try resolve class name from config id2name_map
                                        # use id2name_map if available
                                        try:
                                                name = id2name_map.get("DAWN", {}).get(
                                                        int(k)
                                                )
                                        except Exception:
                                                name = None
                                        writer.writerow(
                                                {
                                                        "class_id": int(k),
                                                        "class_name": name or "",
                                                        "baseline_ap": float(
                                                                baseline_ap.get(
                                                                        str(k),
                                                                        baseline_ap.get(
                                                                                int(k),
                                                                                0.0,
                                                                        ),
                                                                )
                                                        )
                                                        if baseline_ap
                                                        else 0.0,
                                                        "restored_ap": float(
                                                                restored_ap.get(
                                                                        str(k),
                                                                        restored_ap.get(
                                                                                int(k),
                                                                                0.0,
                                                                        ),
                                                                )
                                                        )
                                                        if restored_ap
                                                        else 0.0,
                                                }
                                        )

        # rewrite the top-level metrics.json/metrics.csv with enriched report
        try:
                report_paths = write_metrics_report(run_dir, report)
                report["report_files"] = report_paths
        except Exception:
                pass

        return report
