"""
Logger module using Loguru for the RiceHealthAI project.

Provides a unified, configurable, and elegant logging system
with console + file output (auto-rotation).
"""

import os
from pathlib import Path
from loguru import logger
from ricehealthai.core.settings import find_project_root

# === Configuration globale des logs === #

# Dossier des logs à la racine du projet
LOG_DIR = Path(find_project_root()) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Supprime les handlers existants (évite les doublons)
logger.remove()

# === Handler console === #
logger.add(
    sink=lambda msg: print(msg, end=""),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<cyan>{level:<8}</cyan> | "
        "<yellow>{module}</yellow>:<yellow>{function}</yellow> - "
        "{message}"
    ),
    level="INFO",
)

# === Handler fichier avec rotation === #
logger.add(
    LOG_DIR / "ricehealthai.log",
    rotation="5 MB",
    retention="10 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function} - {message}",
    level="INFO",
    enqueue=True,
)


def setup_logger(module_name: str):
    """
    Returns a logger bound to the given module name.

    Parameters
    ----------
    module_name : str
        The module or class name using this logger.

    Returns
    -------
    loguru.Logger
        Configured logger bound to the module.
    """
    return logger.bind(module=module_name)
