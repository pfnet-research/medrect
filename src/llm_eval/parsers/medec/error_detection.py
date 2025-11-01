"""MEDEC error detection parser."""


from loguru import logger

from ..base import BaseParser
from .common import parse_medec_error_detection


class ErrorDetectionParser(BaseParser):
    """Parser for error detection metric (binary classification)."""

    def __init__(self):
        super().__init__({"parser_name": "error_detection"})

    def parse(self, output: str) -> int:
        """Parse output to extract error detection result (0/1).

        Only parses MEDEC format:
        - "CORRECT" -> 0 (no error)
        - "number: correction" -> 1 (error detected)
        - Unparseable output -> -1 (parsing failed, treated as incorrect)
        """
        output = self.preprocess_output(output)

        result = parse_medec_error_detection(output)
        if result is not None:
            return result

        logger.warning(f"Could not parse MEDEC format from: {output[:100]}...")
        return -1  # Return -1 for parsing failures to treat as incorrect
