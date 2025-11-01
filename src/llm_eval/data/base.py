"""Base data structures and loaders for different tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Union, List


@dataclass
class BaseSample(ABC):
    """Base class for all samples."""

    sample_id: Union[str, int]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return {
            "sample_id": self.sample_id,
            "metadata": self.metadata,
            **{
                k: v
                for k, v in self.__dict__.items()
                if k not in ["sample_id", "metadata"]
            },
        }


class BaseLoader(ABC):
    """Base class for data loaders."""

    @abstractmethod
    def load(self, dataset_name: str) -> List[BaseSample]:
        """Load samples from dataset.

        Args:
            dataset_name: Key identifying the dataset to load

        Returns:
            List of BaseSample objects
        """
        pass
