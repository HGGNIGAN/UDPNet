#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.common.config import load_yaml_config
from pipeline.run.evaluate_pipeline import evaluate_pipeline


def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description="Run end-to-end restoration -> detection -> IoU/mAP evaluation pipeline."
        )
        parser.add_argument(
                "--config",
                type=str,
                default="configs/pipeline.yaml",
                help="Path to YAML config.",
        )
        parser.add_argument(
                "--datasets",
                nargs="*",
                default=None,
                help="Optional subset. Example: --datasets DAWN RTTS",
        )
        parser.add_argument(
                "--max-images",
                type=int,
                default=None,
                help="Optional hard cap on number of images to process.",
        )
        parser.add_argument(
                "--run-name",
                type=str,
                default=None,
                help="Optional label recorded in metrics.json; output dir is auto-derived.",
        )
        parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Initialize components and data only; skip model inference.",
        )
        return parser.parse_args()


def main() -> int:
        args = parse_args()
        config = load_yaml_config(Path(args.config).expanduser().resolve())

        selected: Optional[Sequence[str]] = args.datasets if args.datasets else None

        summary = evaluate_pipeline(
                config=config,
                selected_datasets=selected,
                max_images=args.max_images,
                run_name=args.run_name,
                dry_run=args.dry_run,
        )

        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
        raise SystemExit(main())
