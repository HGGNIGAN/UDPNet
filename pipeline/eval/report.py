from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict


def write_metrics_report(output_dir: Path, report: Dict[str, Any]) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "metrics.json"
        csv_path = output_dir / "metrics.csv"

        with json_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)

        baseline = report["baseline"]
        restored = report["restored"]
        improvement = report["improvement"]
        quality = report.get("quality", {})

        rows = [
                {
                        "category": "detection",
                        "metric": "iou_all_gt",
                        "baseline": baseline.get("iou", {}).get("all_gt"),
                        "restored": restored.get("iou", {}).get("all_gt"),
                        "delta": improvement.get("iou_all_gt"),
                        "value": "",
                },
                {
                        "category": "detection",
                        "metric": "iou_matched",
                        "baseline": baseline.get("iou", {}).get("matched"),
                        "restored": restored.get("iou", {}).get("matched"),
                        "delta": improvement.get("iou_matched"),
                        "value": "",
                },
                {
                        "category": "detection",
                        "metric": "mean_iou_tp50",
                        "baseline": baseline.get("mean_iou_tp50", 0.0),
                        "restored": restored.get("mean_iou_tp50", 0.0),
                        "delta": improvement.get("mean_iou_tp50", 0.0),
                        "value": "",
                },
                {
                        "category": "detection",
                        "metric": "mean_iou",
                        "baseline": baseline.get("mean_iou", 0.0),
                        "restored": restored.get("mean_iou", 0.0),
                        "delta": improvement.get("mean_iou", 0.0),
                        "value": "",
                },
                {
                        "category": "detection",
                        "metric": "map",
                        "baseline": baseline["map"],
                        "restored": restored["map"],
                        "delta": improvement["map"],
                        "value": "",
                },
                {
                        "category": "detection",
                        "metric": "precision",
                        "baseline": report.get("detection_aggregate", {})
                        .get("baseline_precision", {})
                        .get("mean"),
                        "restored": report.get("detection_aggregate", {})
                        .get("restored_precision", {})
                        .get("mean"),
                        "delta": None,
                        "value": "",
                },
                {
                        "category": "detection",
                        "metric": "recall",
                        "baseline": report.get("detection_aggregate", {})
                        .get("baseline_recall", {})
                        .get("mean"),
                        "restored": report.get("detection_aggregate", {})
                        .get("restored_recall", {})
                        .get("mean"),
                        "delta": None,
                        "value": "",
                },
                {
                        "category": "detection",
                        "metric": "f1",
                        "baseline": report.get("detection_aggregate", {})
                        .get("baseline_f1", {})
                        .get("mean"),
                        "restored": report.get("detection_aggregate", {})
                        .get("restored_f1", {})
                        .get("mean"),
                        "delta": None,
                        "value": "",
                },
                {
                        "category": "quality",
                        "metric": "psnr",
                        "baseline": "",
                        "restored": "",
                        "delta": "",
                        "value": quality.get("psnr", {}).get("mean"),
                },
                {
                        "category": "quality",
                        "metric": "ssim",
                        "baseline": "",
                        "restored": "",
                        "delta": "",
                        "value": quality.get("ssim", {}).get("mean"),
                },
        ]

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                                "category",
                                "metric",
                                "baseline",
                                "restored",
                                "delta",
                                "value",
                        ],
                )
                writer.writeheader()
                writer.writerows(rows)

        # append per-class AP rows if present
        ap_base = report.get("baseline", {}).get("ap_per_class", {})
        ap_rest = report.get("restored", {}).get("ap_per_class", {})
        if ap_base or ap_rest:
                with csv_path.open("a", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                                handle,
                                fieldnames=[
                                        "category",
                                        "metric",
                                        "baseline",
                                        "restored",
                                        "delta",
                                        "value",
                                ],
                        )
                        # no header; append rows for each class
                        keys = sorted(
                                {
                                        int(k)
                                        for k in list(ap_base.keys())
                                        + list(ap_rest.keys())
                                }
                        )
                        for k in keys:
                                b = ap_base.get(str(k), ap_base.get(k, None))
                                r = ap_rest.get(str(k), ap_rest.get(k, None))
                                writer.writerow(
                                        {
                                                "category": "detection_ap",
                                                "metric": f"ap_class_{k}",
                                                "baseline": float(b)
                                                if b is not None
                                                else "",
                                                "restored": float(r)
                                                if r is not None
                                                else "",
                                                "delta": "",
                                                "value": "",
                                        }
                                )

        return {
                "json": str(json_path),
                "csv": str(csv_path),
        }
