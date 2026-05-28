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
from pipeline.data.scan_and_pair import scan_and_pair_datasets


def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description=(
                        "Scan configured datasets (DAWN/RTTS/...) and organize image-label pairs "
                        "into a unified structure for detection training/evaluation."
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
                help="Optional subset of datasets to scan. Example: --datasets DAWN RTTS",
        )
        parser.add_argument(
                "--link-mode",
                choices=["symlink", "hardlink", "copy"],
                default=None,
                help="Override link mode from config.",
        )
        parser.add_argument(
                "--overwrite",
                action="store_true",
                help="Overwrite existing organized files.",
        )
        parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Scan only. No files/manifests written.",
        )
        return parser.parse_args()


def main() -> int:
        args = parse_args()

        config_path = Path(args.config).expanduser().resolve()
        config = load_yaml_config(config_path)

        overwrite_value: Optional[bool] = True if args.overwrite else None
        selected: Optional[Sequence[str]] = args.datasets if args.datasets else None

        summary = scan_and_pair_datasets(
                config=config,
                selected_datasets=selected,
                link_mode=args.link_mode,
                overwrite=overwrite_value,
                dry_run=args.dry_run,
        )

        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
        raise SystemExit(main())
