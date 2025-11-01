"""Primary MEDEC standard format parsing."""

import re
from typing import Optional


def preprocess_decorative_elements(output: str) -> str:
    """Remove decorative elements like backticks and quotes."""
    # Remove code blocks with language specifiers (like ```text, ```plaintext)
    output = re.sub(r"```\w*\s*\n", "", output)
    output = re.sub(r"```\w*", "", output)

    # Remove backticks (including empty code blocks)
    output = re.sub(r"`([^`]*)`", r"\1", output)
    output = re.sub(r"```", "", output)  # Remove remaining triple backticks

    # Remove surrounding quotes
    output = re.sub(r'^["\']+|["\']+$', "", output.strip())

    return output.strip()


def is_correct_response(output: str) -> bool:
    """Check if output indicates a CORRECT response."""
    # Standard CORRECT
    if re.search(r"\bCORRECT\b", output, re.IGNORECASE):
        return True

    # Underscore variations: _CORRECT_
    if re.search(r"_CORRECT_", output, re.IGNORECASE):
        return True

    # Check if _CORRECT_ appears at the start of a line (common in code blocks)
    if re.search(r"^\s*_CORRECT_", output, re.MULTILINE | re.IGNORECASE):
        return True

    return False


def extract_medec_correction(output: str) -> Optional[str]:
    """Extract correction text from MEDEC standard format 'number: correction'."""
    # Try to match multiline correction text
    match = re.search(
        r"^\s*\d+\s*:\s*(.+?)(?:\n\n|\n[A-Z]|\n\*|\n-|$)",
        output,
        re.MULTILINE | re.DOTALL,
    )
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Fallback to single line
    match = re.search(r"^\s*\d+\s*:\s*(.+?)(?:\n|$)", output, re.MULTILINE)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction
    return None


def extract_medec_sentence_id(output: str) -> Optional[int]:
    """Extract sentence ID from MEDEC standard format 'number: correction'."""
    match = re.search(r"^\s*(\d+)\s*:\s*.+", output, re.MULTILINE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def parse_primary_error_detection(output: str) -> Optional[int]:
    """Parse primary MEDEC format for error detection (0/1)."""
    # Preprocess to remove decorative elements
    output = preprocess_decorative_elements(output)

    if is_correct_response(output):
        return 0

    if extract_medec_sentence_id(output) is not None:
        return 1

    return None


def parse_primary_error_correction(output: str) -> Optional[str]:
    """Parse primary MEDEC format for error correction."""
    # Preprocess to remove decorative elements
    processed_output = preprocess_decorative_elements(output)

    if is_correct_response(processed_output):
        return None

    return extract_medec_correction(processed_output)


def parse_primary_sentence_extraction(output: str) -> Optional[int]:
    """Parse primary MEDEC format for sentence extraction."""
    # Preprocess to remove decorative elements
    output = preprocess_decorative_elements(output)

    if is_correct_response(output):
        return 0

    return extract_medec_sentence_id(output)
