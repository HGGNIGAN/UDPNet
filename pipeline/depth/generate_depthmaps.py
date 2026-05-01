from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from pipeline.common.paths import resolve_from_project_root

ConfigDict = Dict[str, Any]


def generate_depthmaps_for_datasets(
        config: ConfigDict,
        selected_datasets: Optional[Sequence[str]] = None,
        overwrite: bool = False,
        dry_run: bool = False,
) -> Dict[str, Any]:
        datasets_cfg = config.get("datasets", {})
        entries = datasets_cfg.get("entries", {})
        active = (
                list(selected_datasets)
                if selected_datasets
                else list(datasets_cfg.get("active", entries.keys()))
        )

        paths_cfg = config.get("paths", {})
        runtime_cfg = config.get("runtime", {})
        organized_root = resolve_from_project_root(
                config,
                str(paths_cfg.get("organized_root", "Datasets/organized")),
        )
        depth_device = str(runtime_cfg.get("device", "auto"))
        depth_script = (
                Path(__file__).resolve().parents[2] / "scripts" / "depthmap-create.py"
        ).resolve()

        if not depth_script.exists():
                raise FileNotFoundError(f"Depth script not found: {depth_script}")

        run_summary: Dict[str, Any] = {
                "depth_script": str(depth_script),
                "organized_root": str(organized_root),
                "depth_device": depth_device,
                "dry_run": dry_run,
                "overwrite": overwrite,
                "datasets": {},
        }

        for dataset_name in active:
                dataset_dir = organized_root / dataset_name
                input_image_dir = dataset_dir / "images"
                output_depth_dir = dataset_dir / "DepthMaps"

                if not input_image_dir.exists():
                        raise FileNotFoundError(
                                f"Organized images dir not found for {dataset_name}: {input_image_dir}"
                        )

                if output_depth_dir.exists() and (not overwrite):
                        run_summary["datasets"][dataset_name] = {
                                "status": "skipped_exists",
                                "input_image_dir": str(input_image_dir),
                                "output_depth_dir": str(output_depth_dir),
                        }
                        continue

                if dry_run:
                        run_summary["datasets"][dataset_name] = {
                                "status": "dry_run",
                                "input_image_dir": str(input_image_dir),
                                "output_depth_dir": str(output_depth_dir),
                        }
                        continue

                output_depth_dir.mkdir(parents=True, exist_ok=True)

                command = [
                        "python",
                        str(depth_script),
                        str(input_image_dir),
                        str(output_depth_dir),
                        "--device",
                        depth_device,
                ]

                process = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                )

                run_summary["datasets"][dataset_name] = {
                        "status": "ok" if process.returncode == 0 else "failed",
                        "returncode": process.returncode,
                        "input_image_dir": str(input_image_dir),
                        "output_depth_dir": str(output_depth_dir),
                        "stdout_tail": "\n".join(
                                process.stdout.strip().splitlines()[-20:]
                        ),
                        "stderr_tail": "\n".join(
                                process.stderr.strip().splitlines()[-20:]
                        ),
                }

                if process.returncode != 0:
                        raise RuntimeError(
                                f"Depth generation failed for {dataset_name} (code={process.returncode})."
                        )

        return run_summary
