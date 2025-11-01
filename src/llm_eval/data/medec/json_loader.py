"""MEDEC JSON dataset loader for synthesized data."""

from typing import List, Dict, Any
import json
from pathlib import Path

from loguru import logger

from ..base import BaseLoader
from .samples import MEDECSample


class MEDECJSONLoader(BaseLoader):
    """MEDEC JSON dataset loader for synthesized data."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MEDEC JSON loader.

        Args:
            config: Configuration dict with data_dir and datasets
        """
        self.data_dir = Path(config["data_dir"])
        self.datasets_config = config["datasets"]

        # Build file paths from configuration
        self.file_paths = {}
        for dataset_type, filename in self.datasets_config.items():
            self.file_paths[dataset_type] = self.data_dir / filename

    def load(self, dataset_name: str) -> List[MEDECSample]:
        """Load samples from dataset key."""
        if dataset_name not in self.file_paths:
            raise ValueError(
                f"Unknown dataset key: {dataset_name}. Available: {list(self.file_paths.keys())}"
            )

        file_path = self.file_paths[dataset_name]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for i, item in enumerate(data):
            sample = self._create_medec_sample(item, i)
            samples.append(sample)

        logger.info(
            f"Loaded {len(samples)} samples from {dataset_name} dataset ({file_path})"
        )
        return samples

    def _create_medec_sample(self, data: Dict[str, Any], index: int) -> MEDECSample:
        """Create MEDEC sample from JSON data."""
        # Extract all fields with direct access
        sample_id = data["sample_id"]
        sentences = data["sentences"]
        error_flag = data["error_flag"]
        error_type = data["error_type"]
        error_sentence_id = data["error_sentence_id"]
        error_sentence = data["error_sentence"]
        corrected_sentence = data["corrected_sentence"]

        # Extract metadata
        metadata = data["metadata"]
        metadata["row_index"] = index

        return MEDECSample(
            sample_id=sample_id,
            sentences=sentences,
            error_flag=error_flag,
            error_type=error_type,
            error_sentence_id=error_sentence_id,
            error_sentence=error_sentence,
            corrected_sentence=corrected_sentence,
            metadata=metadata,
        )
