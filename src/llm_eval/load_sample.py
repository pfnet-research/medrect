"""Sample loading utilities for loading and converting samples."""

from typing import List, Dict, Any, Tuple
from pathlib import Path

from loguru import logger

from .component_factory import component_factory
from .output.io_handlers import load_json


class SampleLoader:
    """Handles sample loading and conversion operations."""

    @staticmethod
    def load_dataset(dataset_config: Dict[str, Any], dataset_name: str) -> List[Any]:
        """Load dataset samples using configured loader.

        Args:
            dataset_config: Dataset configuration containing path and loader info
            dataset_name: Name of the dataset

        Returns:
            List of loaded sample objects
        """
        dataset_path = dataset_config["path"]
        loader_name = dataset_config["loader"]

        # Extract data_dir from path for loader compatibility
        data_dir = Path(dataset_path).parent
        filename = Path(dataset_path).name
        loader_config = {
            "data_dir": str(data_dir),
            "datasets": {dataset_name: filename},
        }
        loader = component_factory.get_component_instance(
            "loaders", loader_name, loader_config
        )

        samples = loader.load(dataset_name)
        logger.info(
            f"Loaded {len(samples)} samples from dataset '{dataset_name}' using loader '{loader_name}'"
        )
        return samples

    @staticmethod
    def load_raw_responses(
        filepath: str, sample_class: str
    ) -> Tuple[List[Any], List[List[Dict[str, str]]], List[Any]]:
        """Load raw responses from a saved raw_responses.json file and convert samples.

        Args:
            filepath: Path to the raw_responses.json file
            sample_class: Class name for sample component

        Returns:
            Tuple of (samples, messages_batch, outputs)
        """
        file_path = Path(filepath)
        data = load_json(file_path)

        # Extract samples, messages, and outputs from the saved format
        interactions = data.get("interactions", [])

        raw_samples = []
        messages_batch = []
        outputs = []

        for interaction in interactions:
            # Extract sample data
            raw_samples.append(interaction["sample_data"])

            # Extract messages
            messages_batch.append(interaction["messages"])

            # Extract full raw response (preserve all fields)
            raw_response = interaction.get("raw_response", {})
            outputs.append(raw_response)

        # Convert raw samples to proper sample objects
        sample_class_obj = component_factory.get_component_class(
            "samples", sample_class
        )
        samples = [sample_class_obj.from_dict(raw_sample) for raw_sample in raw_samples]

        logger.info(f"Loaded {len(samples)} samples from {file_path}")
        return samples, messages_batch, outputs
