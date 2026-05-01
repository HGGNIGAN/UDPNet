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
                return {
                        "status": "dry_run",
                        "run_dir": str(run_dir),
                        "dataset_size": len(dataset),
                        "effective_eval_size": max_images,
                        "batch_size": batch_size,
                        "restoration_enabled": restoration_runtime.enabled,
                        "detection_weights": str(
                                detection_cfg.get("weights", "weights/yolo/yolo26n.pt")
                        ),
                }

        thresholds = eval_cfg.get(
                "map_iou_thresholds",
                [round(0.5 + 0.05 * idx, 2) for idx in range(10)],
        )
        thresholds = [float(v) for v in thresholds]

        max_visuals = int(eval_cfg.get("max_visuals", 100))
        visuals_dir = run_dir / "visuals"

        gt_collection: List[np.ndarray] = []
        baseline_preds: List[np.ndarray] = []
        restored_preds: List[np.ndarray] = []

        processed = 0
        visuals_saved = 0

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

                visuals_saved += save_visual_batch(
                        output_dir=visuals_dir,
                        image_ids=batch["image_id"],
                        original_images=original_np,
                        restored_images=restored_np,
                        restored_predictions=restored_pred_np,
                        start_index=processed,
                        max_save=max_visuals,
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

        return report
