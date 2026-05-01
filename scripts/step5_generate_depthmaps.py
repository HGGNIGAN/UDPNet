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
from pipeline.depth.generate_depthmaps import generate_depthmaps_for_datasets


def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description=(
                        "Generate depth maps for organized datasets by calling scripts/depthmap-create.py."
                )
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
                "--overwrite",
                action="store_true",
                help="Regenerate even if output directory exists.",
        )
        parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Plan only; do not run depth generation.",
        )
        return parser.parse_args()


def main() -> int:
        args = parse_args()
        config = load_yaml_config(Path(args.config).expanduser().resolve())

        selected: Optional[Sequence[str]] = args.datasets if args.datasets else None

        summary = generate_depthmaps_for_datasets(
                config=config,
                selected_datasets=selected,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
        )

        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
        raise SystemExit(main())
