"""Base response formatter for benchmark viewer."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseResponseFormatter(ABC):
    """Base class for formatting model responses for display."""

    def __init__(self, task_config: Optional[Any] = None):
        self.task_config = task_config

    @abstractmethod
    def format_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Format prediction for display."""
        pass

    @abstractmethod
    def format_ground_truth(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format ground truth for display."""
        pass

    @abstractmethod
    def compare_prediction(
        self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare prediction with ground truth."""
        pass
