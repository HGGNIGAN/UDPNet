from __future__ import annotations

from typing import Any, Dict

from ultralytics import YOLO

from pipeline.common.device import resolve_detection_device_arg
from pipeline.common.paths import resolve_from_project_root

ConfigDict = Dict[str, Any]


class DetectionModelLoader:
        def __init__(self, config: ConfigDict):
                self.config = config

        def load(self) -> YOLO:
                detection_cfg = self.config.get("detection", {})

                weights_value = detection_cfg.get(
                        "weights",
                        "weights/yolo/yolo26n.pt",
                )
                weights_path = resolve_from_project_root(
                        self.config, str(weights_value)
                )
                if not weights_path.exists():
                        raise FileNotFoundError(
                                f"Detection weights not found: {weights_path}"
                        )

                return YOLO(str(weights_path))


def load_detection_inference_settings(config: ConfigDict) -> Dict[str, Any]:
        detection_cfg = config.get("detection", {})
        vram_cfg = config.get("vram", {})
        runtime_cfg = config.get("runtime", {})

        img_size = vram_cfg.get("image_resolution", [640, 640])
        if isinstance(img_size, int):
                img_size = [img_size, img_size]

        device_name = resolve_detection_device_arg(runtime_cfg)

        return {
                "conf": float(detection_cfg.get("conf_threshold", 0.25)),
                "iou": float(detection_cfg.get("nms_iou_threshold", 0.7)),
                "max_det": int(detection_cfg.get("max_det", 300)),
                "imgsz": [int(img_size[1]), int(img_size[0])],
                "device": device_name,
                "verbose": False,
        }
