"""MEDEC-specific sample structures."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..base import BaseSample


@dataclass
class MEDECSample(BaseSample):
    """Sample for MEDEC error detection and correction tasks."""

    sentences: str
    error_flag: int
    error_type: Optional[str] = None
    error_sentence_id: Optional[int] = None  # Changed from str to int
    error_sentence: Optional[str] = None
    corrected_sentence: Optional[str] = None

    def __post_init__(self):
        """Validate sample data."""
        assert self.error_flag in [0, 1], f"Invalid error_flag: {self.error_flag}"
        assert self.sentences.strip(), "Sentences cannot be empty"

    def has_error(self) -> bool:
        """Check if sample has error."""
        return self.error_flag == 1

    def get_error_info(self) -> Dict[str, Any]:
        """Get error information."""
        if not self.has_error():
            return {"has_error": False}

        return {
            "has_error": True,
            "error_type": self.error_type,
            "error_sentence_id": self.error_sentence_id,
            "error_sentence": self.error_sentence,
            "corrected_sentence": self.corrected_sentence,
        }

    @classmethod
    def from_dict(cls, raw_sample: Dict[str, Any]) -> "MEDECSample":
        """Create MEDECSample from dictionary (for re-evaluation).

        Args:
            raw_sample: Dictionary containing sample data

        Returns:
            MEDECSample object
        """
        return cls(
            sample_id=raw_sample["sample_id"],
            metadata=raw_sample["metadata"],
            sentences=raw_sample["sentences"],
            error_flag=raw_sample["error_flag"],
            error_type=raw_sample["error_type"],
            error_sentence_id=raw_sample["error_sentence_id"],
            error_sentence=raw_sample["error_sentence"],
            corrected_sentence=raw_sample["corrected_sentence"],
        )
