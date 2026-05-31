from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


def summarize_scalar_series(values: Sequence[Any]) -> Dict[str, Any]:
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
                return {
                        "mean": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "count": 0,
                }

        array = np.asarray(cleaned, dtype=np.float64)
        return {
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "count": int(array.size),
        }


def aggregate_dataset_means(
        dataset_summaries: Mapping[str, Mapping[str, Any]],
        metric_key: str,
) -> Dict[str, Any]:
        values = []
        for summary in dataset_summaries.values():
                metric = summary.get(metric_key)
                if not isinstance(metric, Mapping):
                        continue
                value = metric.get("mean")
                if value is None:
                        continue
                values.append(float(value))
        return summarize_scalar_series(values)
