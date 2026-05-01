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

        rows = [
                {
                        "metric": "mean_iou",
                        "baseline": baseline["mean_iou"],
                        "restored": restored["mean_iou"],
                        "delta": improvement["mean_iou"],
                },
                {
                        "metric": "map",
                        "baseline": baseline["map"],
                        "restored": restored["map"],
                        "delta": improvement["map"],
                },
        ]

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                        handle, fieldnames=["metric", "baseline", "restored", "delta"]
                )
                writer.writeheader()
                writer.writerows(rows)

        return {
                "json": str(json_path),
                "csv": str(csv_path),
        }
