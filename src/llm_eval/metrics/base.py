"""Base metric class for evaluation."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from loguru import logger

from ..data.base import BaseSample
from ..parsers.base import BaseParser


class BaseMetric(ABC):
    """Base class for all evaluation metrics."""

    def __init__(self, metric_name: str, parser: Optional[BaseParser] = None):
        self.metric_name = metric_name
        self.parser = parser

    @abstractmethod
    def compute_metric(
        self, predictions: List[Any], references: List[Any]
    ) -> Dict[str, Any]:
        """
        Compute metric scores.

        Args:
            predictions: List of predictions
            references: List of reference values

        Returns:
            Dictionary containing:
            - "aggregated": Dict[str, float] - aggregated scores
            - "per_item": List[Dict[str, Any]] - per-item scores
        """
        pass

    def extract_predictions(self, outputs: List[str]) -> List[Any]:
        """Extract predictions from model outputs using parser."""
        if self.parser is None:
            return outputs

        predictions = []
        for output in outputs:
            try:
                pred = self.parser.parse(output)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to parse output: {e}")
                predictions.append(None)

        return predictions

    def extract_references(self, samples: List[BaseSample]) -> List[Any]:
        """Extract reference values from samples."""
        return [self._get_reference_value(sample) for sample in samples]

    @abstractmethod
    def _get_reference_value(self, sample: BaseSample) -> Any:
        """Get reference value for a single sample."""
        pass

    def evaluate(self, outputs: List[str], samples: List[BaseSample]) -> Dict[str, Any]:
        """Evaluate model outputs against samples."""
        predictions = self.extract_predictions(outputs)
        references = self.extract_references(samples)

        # No longer filter out None predictions - parsers now return special values for failures
        # This ensures all samples are included in evaluation for fair comparison
        if not predictions:
            logger.warning("No predictions found")
            return {"aggregated": {f"{self.metric_name}_score": 0.0}, "per_item": []}

        return self.compute_metric(predictions, references)
