"""Secondary fallback pattern parsing for MEDEC."""

import re
from typing import Optional


def extract_fallback_sentence_id(output: str) -> Optional[int]:
    """Extract sentence ID from fallback patterns."""
    # Pattern 1: "文[number]: [text]"
    match = re.search(r"文(\d+)\s*:\s*.+", output)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    # Pattern 2: "[number]. [text]" at start of line
    match = re.search(r"^\s*(\d+)\.\s*.+", output, re.MULTILINE)
    if match:
        try:
            sentence_id = int(match.group(1))
            # Reasonable range check
            if 1 <= sentence_id <= 50:
                return sentence_id
        except ValueError:
            pass

    # Pattern 3: "文番号: [number]: [text]"
    match = re.search(r"文番号\s*:\s*(\d+)\s*:\s*.+", output)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    # Pattern 4: "文番号: [number]\n[number]. [text]"
    match = re.search(r"文番号\s*:\s*(\d+)\s*\n\s*\d+\.\s*.+", output, re.MULTILINE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    return None


def extract_fallback_correction(output: str) -> Optional[str]:
    """Extract correction text from fallback patterns."""
    # Pattern 1: "文[number]: [text]"
    match = re.search(r"文\d+\s*:\s*(.+?)(?:\n|$)", output)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Pattern 2: "[number]. [text]" at start of line
    match = re.search(r"^\s*\d+\.\s*(.+?)(?:\n|$)", output, re.MULTILINE)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Pattern 3: "文番号: [number]: [text]"
    match = re.search(r"文番号\s*:\s*\d+\s*:\s*(.+?)(?:\n|$)", output)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Pattern 4: "文番号: [number]\n[number]. [text]"
    match = re.search(
        r"文番号\s*:\s*\d+\s*\n\s*\d+\.\s*(.+?)(?:\n|$)", output, re.MULTILINE
    )
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    return None


def parse_secondary_error_detection(output: str) -> Optional[int]:
    """Parse fallback patterns for error detection (0/1)."""
    if extract_fallback_sentence_id(output) is not None:
        return 1
    return None


def parse_secondary_error_correction(output: str) -> Optional[str]:
    """Parse fallback patterns for error correction."""
    return extract_fallback_correction(output)


def parse_secondary_sentence_extraction(output: str) -> Optional[int]:
    """Parse fallback patterns for sentence extraction."""
    return extract_fallback_sentence_id(output)
