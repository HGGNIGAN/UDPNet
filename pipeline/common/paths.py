from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


ConfigDict = Dict[str, Any]


def get_project_root(config: ConfigDict) -> Path:
        """Resolve root path based on COLAB_MODE toggle."""
        project_cfg = config.get("project", {})
        colab_mode = bool(project_cfg.get("colab_mode", False))

        root_key = "colab_root" if colab_mode else "local_root"
        root_value = project_cfg.get(root_key, ".")

        return Path(root_value).expanduser().resolve()


def resolve_from_project_root(config: ConfigDict, path_value: str) -> Path:
        """Resolve an absolute/relative path with project root prefix logic."""
        path_obj = Path(path_value).expanduser()
        if path_obj.is_absolute():
                return path_obj.resolve()

        return (get_project_root(config) / path_obj).resolve()


def resolve_under(base_dir: Path, path_value: str) -> Path:
        """Resolve path under a specific base directory unless already absolute."""
        path_obj = Path(path_value).expanduser()
        if path_obj.is_absolute():
                return path_obj.resolve()

        return (base_dir / path_obj).resolve()
