"""Language detection utility for automatic tokenization method selection."""

import re
from typing import List


def detect_language_from_text(text: str) -> str:
    """Detect language from text content.

    Args:
        text: Input text to analyze

    Returns:
        Language code: 'ja' for Japanese, 'en' for English, 'unknown' for uncertain
    """
    if not text or not text.strip():
        return "unknown"

    # Count Japanese characters (hiragana, katakana, kanji)
    japanese_chars = len(re.findall(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]", text))

    # Count ASCII alphabetic characters
    ascii_chars = len(re.findall(r"[a-zA-Z]", text))

    total_chars = len(re.sub(r"\s", "", text))  # Remove whitespace for counting

    if total_chars == 0:
        return "unknown"

    japanese_ratio = japanese_chars / total_chars
    ascii_ratio = ascii_chars / total_chars

    # Decision thresholds
    if japanese_ratio > 0.1:  # If more than 10% Japanese characters
        return "ja"
    elif ascii_ratio > 0.5:  # If more than 50% ASCII characters
        return "en"
    else:
        return "unknown"


def detect_language_from_samples(samples: List[str]) -> str:
    """Detect language from multiple text samples.

    Args:
        samples: List of text samples to analyze

    Returns:
        Most common language detected across samples
    """
    if not samples:
        return "unknown"

    # Analyze first few samples (up to 5) to avoid processing too much data
    sample_subset = samples[: min(5, len(samples))]

    language_counts = {"ja": 0, "en": 0, "unknown": 0}

    for sample in sample_subset:
        if sample and sample.strip():
            lang = detect_language_from_text(sample)
            language_counts[lang] += 1

    # Return the most common language
    return max(language_counts.items(), key=lambda x: x[1])[0]
