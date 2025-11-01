"""Tertiary patterns for MEDEC parsing (stage 3)."""

import re
from typing import Optional


def extract_tertiary_sentence_id(output: str) -> Optional[int]:
    """Extract sentence ID from tertiary patterns (minor variations)."""
    # Pattern 1: "文番号: [number]." (with period)
    match = re.search(r"文番号\s*:\s*(\d+)\.", output)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    # Pattern 2: "[number]文目:" (nth sentence)
    match = re.search(r"(\d+)文目\s*:", output)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    # Pattern 3: "文番号[number]:" (no space)
    match = re.search(r"文番号(\d+)\s*:", output)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    # Pattern 4: "- [number]:" (hyphen prefix)
    match = re.search(r"-\s*(\d+)\s*:", output)
    if match:
        try:
            sentence_id = int(match.group(1))
            # Reasonable range check
            if 1 <= sentence_id <= 50:
                return sentence_id
        except ValueError:
            pass

    # Pattern 5: "文番号: [number]" followed by newline and text (no colon after number)
    match = re.search(r"文番号\s*:\s*(\d+)\s*\n\s*.+", output, re.MULTILINE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    # Pattern 6: "文番号: [number] [text]" (space-separated, no colon after number)
    match = re.search(r"文番号\s*:\s*(\d+)\s+.+", output)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    return None


def extract_tertiary_correction(output: str) -> Optional[str]:
    """Extract correction text from tertiary patterns."""
    # Pattern 1: "文番号: [number]." followed by text
    match = re.search(r"文番号\s*:\s*\d+\.\s*(.+?)(?:\n|$)", output)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Pattern 2: "[number]文目:" followed by text
    match = re.search(r"\d+文目\s*:\s*(.+?)(?:\n|$)", output)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Pattern 3: "文番号[number]:" followed by text
    match = re.search(r"文番号\d+\s*:\s*(.+?)(?:\n|$)", output)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Pattern 4: "- [number]:" followed by text
    match = re.search(r"-\s*\d+\s*:\s*(.+?)(?:\n|$)", output)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Pattern 5: "文番号: [number]" followed by newline and text
    match = re.search(r"文番号\s*:\s*\d+\s*\n\s*(.+?)(?:\n|$)", output, re.MULTILINE)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    # Pattern 6: "文番号: [number] [text]" (space-separated)
    match = re.search(r"文番号\s*:\s*\d+\s+(.+?)(?:\n|$)", output)
    if match:
        correction = match.group(1).strip()
        if correction:
            return correction

    return None


def is_no_error_response(output: str) -> bool:
    """Check for explicit 'no error' responses."""
    # Japanese "no error" expressions
    no_error_patterns = [
        r"\bエラーなし\b",
        r"\b誤りなし\b",
        r"\b間違いなし\b",
        r"\b問題なし\b",
    ]

    for pattern in no_error_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return True

    return False


def parse_tertiary_error_detection(output: str) -> Optional[int]:
    """Parse tertiary patterns for error detection (0/1)."""
    # Check for explicit "no error" responses first
    if is_no_error_response(output):
        return 0

    # Check for sentence ID patterns
    if extract_tertiary_sentence_id(output) is not None:
        return 1

    return None


def parse_tertiary_error_correction(output: str) -> Optional[str]:
    """Parse tertiary patterns for error correction."""
    # No correction needed for "no error" responses
    if is_no_error_response(output):
        return None

    return extract_tertiary_correction(output)


def parse_tertiary_sentence_extraction(output: str) -> Optional[int]:
    """Parse tertiary patterns for sentence extraction."""
    # No error sentence for "no error" responses
    if is_no_error_response(output):
        return 0

    return extract_tertiary_sentence_id(output)
