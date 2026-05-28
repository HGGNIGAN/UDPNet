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
        box_iou_xyxy,
)
from pipeline.eval.report import write_metrics_report
from pipeline.eval.visualize import save_visual_batch
from pipeline.run.metrics_utils import compute_psnr, compute_ssim
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

        # build per-dataset max mapping
        default_max = 0
        per_dataset_max = {}
        try:
                records = getattr(dataset_for_loader, "records", None)
                if records is None and hasattr(dataset_for_loader, "dataset"):
                        records = dataset_for_loader.dataset.records
                dataset_names = sorted({r["dataset"] for r in records})
        except Exception:
                dataset_names = []

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
                )
                visuals_saved += saved_count

                # compute per-image metrics (PSNR, SSIM, per-image mean best IOU)
                for idx in range(len(batch["image_id"])):
                        img_id = batch["image_id"][idx]
                        ds = batch["dataset"][idx]
                        orig = original_np[idx]
                        rest = restored_np[idx]
                        preds = restored_pred_np[idx]
                        gt = batch["gt_xyxy"][idx]

                        psnr_val = float(compute_psnr(orig, rest))
                        ssim_val = float(compute_ssim(orig, rest))

                        # per-image mean best iou
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

                        per_dataset_rows[ds].append(
                                {
                                        "image_id": img_id,
                                        "image_path": batch["image_path"][idx],
                                        "psnr": psnr_val,
                                        "ssim": ssim_val,
                                        "mean_iou": mean_iou_img,
                                        "num_gt": int(gt.shape[0])
                                        if hasattr(gt, "shape")
                                        else 0,
                                        "num_preds": int(preds.shape[0])
                                        if hasattr(preds, "shape")
                                        else 0,
                                }
                        )

                processed += len(gt_batch)

        baseline_iou = mean_best_iou(baseline_preds, gt_collection)
        restored_iou = mean_best_iou(restored_preds, gt_collection)

        baseline_map_dict = compute_map(baseline_preds, gt_collection, thresholds)
        restored_map_dict = compute_map(restored_preds, gt_collection, thresholds)

        baseline_map = float(baseline_map_dict["map"])
        restored_map = float(restored_map_dict["map"])

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
                "visuals_saved": visuals_saved,
        }

        report_paths = write_metrics_report(run_dir, report)
        report["report_files"] = report_paths

        # Write per-dataset per-image metrics and summaries
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        all_summary = {}
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
                                        "mean_iou": _agg_h("mean_iou"),
                                }
                                with (dawn_dir / f"{h}_summary.json").open(
                                        "w", encoding="utf-8"
                                ) as handle:
                                        json.dump(h_summary, handle, indent=2)

                # compute summary stats
                import math

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
                        "mean_iou": _agg("mean_iou"),
                }

                with ds_json.open("w", encoding="utf-8") as handle:
                        json.dump(summary, handle, indent=2)

                all_summary[ds] = summary

        # write overall summary
        overall_path = metrics_dir / "all_datasets_summary.json"
        with overall_path.open("w", encoding="utf-8") as handle:
                json.dump(all_summary, handle, indent=2)

        report["metrics_dir"] = str(metrics_dir)

        return report
