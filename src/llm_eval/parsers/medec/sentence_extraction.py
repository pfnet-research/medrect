"""MEDEC sentence extraction parser."""


from loguru import logger

from ..base import BaseParser
from .common import parse_medec_sentence_extraction


class SentenceExtractionParser(BaseParser):
    """Parser for sentence extraction metric (sentence ID identification)."""

    def __init__(self):
        super().__init__({"parser_name": "sentence_extraction"})

    def parse(self, output: str) -> int:
        """Parse output to extract sentence ID.

        Only parses MEDEC format:
        - "CORRECT" -> 0 (no error sentence)
        - "number: correction" -> number (sentence ID)
        - Unparseable output -> -1 (parsing failed, treated as incorrect)
        """
        output = self.preprocess_output(output)

        result = parse_medec_sentence_extraction(output)
        if result is not None:
            return result

        logger.warning(f"Could not parse MEDEC format from: {output[:100]}...")
        return -1  # Return -1 for parsing failures to treat as incorrect
