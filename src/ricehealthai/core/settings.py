"""
Settings and configuration loader for RiceHealthAI.

This module provides utilities to load YAML configuration files
into Python dictionaries. It ensures consistent configuration
management across all project components.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def find_project_root() -> Path:
    """
    Returns the absolute path of the RiceHealthAI project root.

    The root directory is identified by locating the first parent directory
    containing a pyproject.toml file.
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (missing pyproject.toml).")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file (absolute or relative to project root).

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file, relative to the project root.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file is empty.
    yaml.YAMLError
        If the YAML file is invalid or cannot be parsed.
    """
    root = find_project_root()
    path = (root / config_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")

    if not config:
        raise ValueError(f"Configuration file is empty: {path}")

    return config
