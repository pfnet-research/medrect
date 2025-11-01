"""MEDEC error correction parser."""

import re

from loguru import logger

from ..base import BaseParser
from .common import parse_medec_error_correction


class ErrorCorrectionParser(BaseParser):
    """Parser for error correction metric (corrected text extraction)."""

    MAX_OUTPUT_LENGTH = 5000  # Character limit to prevent CUDA OOM

    def __init__(self):
        super().__init__({"parser_name": "error_correction"})

    def truncate_middle(self, text: str, max_length: int = None) -> str:
        """Truncate middle part of long text while preserving beginning and end."""
        if max_length is None:
            max_length = self.MAX_OUTPUT_LENGTH

        if len(text) <= max_length:
            return text

        # Calculate characters to keep at beginning and end (accounting for ellipsis)
        ellipsis = "\n...[TRUNCATED]...\n"
        available_length = max_length - len(ellipsis)
        half_length = available_length // 2

        # Get beginning and end portions
        start_text = text[:half_length]
        end_text = text[-half_length:]

        logger.warning(
            f"Output truncated from {len(text)} to {max_length} chars (middle omitted)"
        )
        return start_text + ellipsis + end_text

    def parse(self, output: str) -> str:
        """Parse output to extract corrected text.

        Only parses MEDEC format:
        - "CORRECT" -> "" (empty string for no correction needed)
        - "number: correction" -> correction text
        - Unparseable output -> "PARSE_FAILED" (parsing failed, treated as incorrect)
        """
        # Truncate middle part if output is too long to prevent CUDA OOM
        output = self.truncate_middle(output)
        output = self.preprocess_output(output)

        result = parse_medec_error_correction(output)
        if result is not None:
            return (
                result if result is not None else ""
            )  # Handle None from common parser

        # Check for CORRECT (no correction needed) - handled by common function
        if re.search(r"\bCORRECT\b", output, re.IGNORECASE) or re.search(
            r"_CORRECT_", output, re.IGNORECASE
        ):
            return ""

        logger.warning(f"Could not parse MEDEC format from: {output[:100]}...")
        return "PARSE_FAILED"  # Return special string for parsing failures
