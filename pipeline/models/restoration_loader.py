from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from pipeline.common.device import (
        resolve_runtime_gpu_ids,
        resolve_runtime_torch_device,
)
from pipeline.common.paths import resolve_from_project_root

ConfigDict = Dict[str, Any]


@dataclass
class RestorationRuntime:
        model: Optional[torch.nn.Module]
        device: torch.device
        enabled: bool
        use_depth: bool
        output_index: int

        @torch.no_grad()
        def restore(
                self, rgb: torch.Tensor, depth: Optional[torch.Tensor]
        ) -> torch.Tensor:
                if (not self.enabled) or self.model is None:
                        return rgb

                if self.use_depth:
                        if depth is None:
                                raise ValueError(
                                        "Restoration model configured with use_depth=true but depth is missing."
                                )
                        model_input = torch.cat([rgb, depth], dim=1)
                else:
                        model_input = rgb

                output = self.model(model_input)
                restored = _extract_output_tensor(output, self.output_index)
                return torch.clamp(restored, 0.0, 1.0)


class RestorationModelLoader:
        def __init__(self, config: ConfigDict):
                self.config = config
                runtime_cfg = config.get("runtime", {})
                self.device = resolve_runtime_torch_device(runtime_cfg)
                self.gpu_ids = resolve_runtime_gpu_ids(runtime_cfg)
                self.enable_data_parallel = bool(
                        runtime_cfg.get("enable_data_parallel", True)
                )

        def load(self) -> RestorationRuntime:
                restoration_cfg = self.config.get("restoration", {})
                enabled = bool(restoration_cfg.get("enabled", True))
                use_depth = bool(restoration_cfg.get("use_depth", True))
                output_index = int(restoration_cfg.get("output_index", -1))

                if not enabled:
                        return RestorationRuntime(
                                model=None,
                                device=self.device,
                                enabled=False,
                                use_depth=use_depth,
                                output_index=output_index,
                        )

                module_name = str(
                        restoration_cfg.get(
                                "model_module", "Dehazing.ITS.models.FSNet_UDPNet"
                        )
                )
                build_fn_name = str(restoration_cfg.get("build_function", "build_net"))

                module = importlib.import_module(module_name)
                build_fn = getattr(module, build_fn_name)
                model = build_fn()

                ckpt_path = _resolve_restoration_ckpt(self.config)
                checkpoint = torch.load(
                        ckpt_path, map_location="cpu", weights_only=False
                )
                state_dict = _extract_state_dict(checkpoint)
                state_dict = _strip_prefix(state_dict, "model.")

                strict = bool(restoration_cfg.get("strict_state_dict", False))
                if strict:
                        model.load_state_dict(state_dict, strict=True)
                else:
                        compatible = _filter_compatible_state_dict(model, state_dict)
                        model.load_state_dict(compatible, strict=False)

                if (
                        self.device.type == "cuda"
                        and self.enable_data_parallel
                        and len(self.gpu_ids) > 1
                ):
                        model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)

                model.to(self.device)
                model.eval()

                return RestorationRuntime(
                        model=model,
                        device=self.device,
                        enabled=True,
                        use_depth=use_depth,
                        output_index=output_index,
                )


def _resolve_restoration_ckpt(config: ConfigDict) -> Path:
        restoration_cfg = config.get("restoration", {})
        ckpt_value = restoration_cfg.get(
                "checkpoint",
                "UDPNet_pretrained_(checkpoints)/FSNet_UDPNet_OTS.ckpt",
        )
        ckpt_path = resolve_from_project_root(config, str(ckpt_value))
        if not ckpt_path.exists():
                raise FileNotFoundError(
                        f"Restoration checkpoint not found: {ckpt_path}"
                )
        return ckpt_path


def _extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint_obj, dict):
                if "state_dict" in checkpoint_obj and isinstance(
                        checkpoint_obj["state_dict"], dict
                ):
                        return checkpoint_obj["state_dict"]
                tensor_dict = {
                        key: value
                        for key, value in checkpoint_obj.items()
                        if isinstance(value, torch.Tensor)
                }
                if tensor_dict:
                        return tensor_dict

        raise ValueError("Could not extract state_dict from restoration checkpoint.")


def _strip_prefix(
        state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
        updated: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
                if key.startswith(prefix):
                        updated[key[len(prefix) :]] = value
                else:
                        updated[key] = value
        return updated


def _extract_output_tensor(output_obj: Any, output_index: int) -> torch.Tensor:
        if torch.is_tensor(output_obj):
                return output_obj

        if isinstance(output_obj, (list, tuple)):
                tensor_items = [item for item in output_obj if torch.is_tensor(item)]
                if not tensor_items:
                        raise ValueError(
                                "Restoration model output list/tuple has no tensors."
                        )
                return tensor_items[output_index]

        if isinstance(output_obj, dict):
                for key in ("restored", "output", "pred"):
                        value = output_obj.get(key)
                        if torch.is_tensor(value):
                                return value

        raise ValueError("Unsupported restoration model output type.")


def _filter_compatible_state_dict(
        model: torch.nn.Module,
        state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
        model_state = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
                if key not in model_state:
                        continue
                if model_state[key].shape != value.shape:
                        continue
                filtered[key] = value
        return filtered
