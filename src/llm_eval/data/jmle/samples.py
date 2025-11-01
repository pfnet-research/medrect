"""JMLE-specific sample structures."""

from dataclasses import dataclass
from typing import Dict, Optional, List
from ..base import BaseSample


@dataclass
class JMLESample(BaseSample):
    """Sample for JMLE (Japanese Medical Licensing Exam) data synthesis."""

    # Original JMLE data
    question: str
    choices: Dict[str, str]  # Dict format only: {"a": "text", "b": "text", ...}
    answer: List[str]

    # Generated clinical narrative for synthesis
    clinical_narrative: Optional[str] = None
    sentences: Optional[str] = None

    # MEDEC-style error injection data
    error_flag: Optional[int] = None
    error_type: Optional[str] = None
    error_sentence_id: Optional[int] = None
    error_sentence: Optional[str] = None
    corrected_sentence: Optional[str] = None

    def __post_init__(self):
        """Validate sample data."""
        assert self.question.strip(), "Question cannot be empty"
        assert isinstance(self.choices, dict), "Choices must be a dictionary"
        assert len(self.choices) > 0, "Choices cannot be empty"
        assert len(self.answer) > 0, "Answer cannot be empty"
        # Validate all answer keys exist in choices
        for answer_key in self.answer:
            assert answer_key in self.choices, (
                f"Answer key '{answer_key}' not found in choices"
            )
