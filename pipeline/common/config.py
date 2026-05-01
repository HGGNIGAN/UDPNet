from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


def load_yaml_config(config_path: Path) -> ConfigDict:
        """Load pipeline YAML config from disk."""
        path = config_path.expanduser().resolve()
        if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)

        if not isinstance(data, dict):
                raise ValueError(f"Config root must be a dictionary: {path}")

        return data
