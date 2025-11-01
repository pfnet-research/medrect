"""Common MEDEC parsing utilities with staged parsing logic."""

from typing import Optional
from .primary import (
    parse_primary_error_detection,
    parse_primary_error_correction,
    parse_primary_sentence_extraction,
)
from .secondary import (
    parse_secondary_error_detection,
    parse_secondary_error_correction,
    parse_secondary_sentence_extraction,
)
from .tertiary import (
    parse_tertiary_error_detection,
    parse_tertiary_error_correction,
    parse_tertiary_sentence_extraction,
)


def parse_medec_error_detection(output: str) -> Optional[int]:
    """Parse MEDEC format for error detection with staged parsing logic.

    Stage 1: Primary MEDEC format (number: correction, CORRECT) with decoration removal
    Stage 2: Secondary fallback patterns (文17:, 5., 文番号: 3:)
    Stage 3: Tertiary patterns (エラーなし, minor variations)

    Returns:
        0 if CORRECT or no error
        1 if error format found
        None if no format detected
    """
    # Stage 1: Try primary MEDEC format (with preprocessing)
    result = parse_primary_error_detection(output)
    if result is not None:
        return result

    # Stage 2: Try secondary fallback patterns
    result = parse_secondary_error_detection(output)
    if result is not None:
        return result

    # Stage 3: Try tertiary patterns
    return parse_tertiary_error_detection(output)


def parse_medec_error_correction(output: str) -> Optional[str]:
    """Parse MEDEC format for error correction with staged parsing logic.

    Stage 1: Primary MEDEC format (number: correction, CORRECT) with decoration removal
    Stage 2: Secondary fallback patterns (文17:, 5., 文番号: 3:)
    Stage 3: Tertiary patterns (エラーなし, minor variations)

    Returns:
        None if CORRECT or no error (no correction needed)
        correction text if found
        None if no format detected
    """
    # Stage 1: Try primary MEDEC format (with preprocessing)
    result = parse_primary_error_correction(output)
    if result is not None:
        return result

    # Stage 2: Try secondary fallback patterns
    result = parse_secondary_error_correction(output)
    if result is not None:
        return result

    # Stage 3: Try tertiary patterns
    return parse_tertiary_error_correction(output)


def parse_medec_sentence_extraction(output: str) -> Optional[int]:
    """Parse MEDEC format for sentence extraction with staged parsing logic.

    Stage 1: Primary MEDEC format (number: correction, CORRECT) with decoration removal
    Stage 2: Secondary fallback patterns (文17:, 5., 文番号: 3:)
    Stage 3: Tertiary patterns (エラーなし, minor variations)

    Returns:
        0 if CORRECT or no error (no error sentence)
        sentence_id if found
        None if no format detected
    """
    # Stage 1: Try primary MEDEC format (with preprocessing)
    result = parse_primary_sentence_extraction(output)
    if result is not None:
        return result

    # Stage 2: Try secondary fallback patterns
    result = parse_secondary_sentence_extraction(output)
    if result is not None:
        return result

    # Stage 3: Try tertiary patterns
    return parse_tertiary_sentence_extraction(output)
