from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

ConfigDict = Dict[str, Any]


def _extract_digits(text: str) -> str:
        return "".join(ch for ch in text if ch.isdigit())


def _parse_gpu_ids_from_value(value: Any) -> List[int]:
        ids: List[int] = []

        if value is None:
                return ids

        if isinstance(value, int):
                return [value] if value >= 0 else ids

        if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
                for item in value:
                        parsed = _parse_gpu_ids_from_value(item)
                        ids.extend(parsed)
                return _dedupe_preserve_order(ids)

        text = str(value).strip().lower()
        if text in ("", "cpu", "auto"):
                return ids

        if text.startswith("cuda:"):
                text = text[len("cuda:") :]
        elif text == "cuda":
                return [0]

        if "," in text:
                for part in text.split(","):
                        part = part.strip()
                        if not part:
                                continue
                        digits = _extract_digits(part)
                        if digits:
                                ids.append(int(digits))
                return _dedupe_preserve_order(ids)

        digits = _extract_digits(text)
        if digits:
                return [int(digits)]

        return ids


def _dedupe_preserve_order(values: Sequence[int]) -> List[int]:
        out: List[int] = []
        seen = set()
        for value in values:
                if value in seen:
                        continue
                seen.add(value)
                out.append(value)
        return out


def resolve_runtime_gpu_ids(runtime_cfg: ConfigDict) -> List[int]:
        if not torch.cuda.is_available():
                return []

        requested = runtime_cfg.get("gpu_ids")
        ids = _parse_gpu_ids_from_value(requested)

        if not ids:
                ids = _parse_gpu_ids_from_value(runtime_cfg.get("device", "cuda:0"))

        if not ids:
                ids = [0]

        gpu_count = torch.cuda.device_count()
        valid = [idx for idx in ids if 0 <= idx < gpu_count]
        if not valid:
                return [0] if gpu_count > 0 else []

        return _dedupe_preserve_order(valid)


def resolve_runtime_torch_device(runtime_cfg: ConfigDict) -> torch.device:
        raw = str(runtime_cfg.get("device", "cuda:0")).strip().lower()
        if raw.startswith("cpu"):
                return torch.device("cpu")

        if not torch.cuda.is_available():
                return torch.device("cpu")

        gpu_ids = resolve_runtime_gpu_ids(runtime_cfg)
        if not gpu_ids:
                return torch.device("cpu")

        return torch.device(f"cuda:{gpu_ids[0]}")


def resolve_detection_device_arg(runtime_cfg: ConfigDict) -> str:
        raw = str(runtime_cfg.get("device", "cuda:0")).strip().lower()
        if raw.startswith("cpu"):
                return "cpu"

        if not torch.cuda.is_available():
                return "cpu"

        gpu_ids = resolve_runtime_gpu_ids(runtime_cfg)
        if not gpu_ids:
                return "cpu"

        if len(gpu_ids) == 1:
                return str(gpu_ids[0])

        return ",".join(str(idx) for idx in gpu_ids)
