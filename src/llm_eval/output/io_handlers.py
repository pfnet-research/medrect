"""Basic I/O handlers for JSON operations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


def save_json(data: Any, filepath: Path, add_timestamp: bool = True) -> str:
    """Save data to JSON file with optional timestamp metadata.

    Args:
        data: Data to save
        filepath: Path to save to
        add_timestamp: Whether to add timestamp to metadata

    Returns:
        String path to saved file
    """
    # Add timestamp metadata if requested and data is dict
    if add_timestamp and isinstance(data, dict):
        output_data = {"metadata": {"timestamp": datetime.now().isoformat()}, **data}
    else:
        output_data = data

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON data saved to {filepath}")
    return str(filepath)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file.

    Args:
        filepath: Path to load from

    Returns:
        Loaded data
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.debug(f"JSON data loaded from {filepath}")
    return data
