"""JMLE dataset loader."""

from typing import List, Dict, Any
import json
from pathlib import Path

from loguru import logger

from ..base import BaseLoader
from .samples import JMLESample


class JMLELoader(BaseLoader):
    """JMLE dataset loader."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize JMLE loader.

        Args:
            config: Configuration dict with data_dir and datasets
        """
        self.data_dir = Path(config["data_dir"])
        self.datasets_config = config["datasets"]

        # Build file paths from configuration
        self.file_paths = {}
        for dataset_type, filename in self.datasets_config.items():
            self.file_paths[dataset_type] = self.data_dir / filename

    def load(self, dataset_name: str) -> List[JMLESample]:
        """Load samples from dataset key."""
        if dataset_name not in self.file_paths:
            raise ValueError(
                f"Unknown dataset key: {dataset_name}. Available: {list(self.file_paths.keys())}"
            )

        file_path = self.file_paths[dataset_name]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load JSON data
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for i, item in enumerate(data):
            sample = self._create_jmle_sample(item, i)
            samples.append(sample)

        logger.info(
            f"Loaded {len(samples)} samples from {dataset_name} dataset ({file_path})"
        )
        return samples

    def _create_jmle_sample(self, item: Dict[str, Any], index: int) -> JMLESample:
        """Create JMLE sample from item."""
        sample_id = item.get("id", f"jmle-{index}")

        return JMLESample(
            sample_id=sample_id,
            question=item["question"],
            choices=item["choices"],
            answer=item["answer"],
            metadata={"row_index": index, "original_data": item},
        )
