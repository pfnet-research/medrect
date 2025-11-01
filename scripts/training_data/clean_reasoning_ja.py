#!/usr/bin/env python3
"""
Clean Japanese reasoning responses by removing meta-references.

This script removes sentences from reasoning that contain meta-references
to the task setup, prior knowledge, or instructions in Japanese text.

Usage:
    python scripts/training_data/clean_reasoning_ja.py --input raw_responses.json --output cleaned_raw_responses.json
    python scripts/training_data/clean_reasoning_ja.py --input_dir data/outputs/medec/jmedec_test_5samples_incorrect_ef0 --recursive
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any

from loguru import logger


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class ReasoningCleaner:
    """Clean Japanese reasoning text by removing meta-references."""

    def __init__(self):
        # Japanese keywords that indicate meta-references
        self.meta_keywords = [
            "エラーなし例",
            "エラーなしの例",
            "エラーなし問題",
            "エラーあり例",
            "エラーありの例",
            "エラーあり問題",
            "事前に",
            "事前確認",
            "事前情報",
            "事前に知",
            "事前に分",
            "事前に言",
            "事前に確認",
            "reasoning",
            "Reasoning",
            "REASONING",
            "指示通り",
            "指示に従",
            "指示では",
            "指示された",
            "ゼロから分析",
            "ゼロから検討",
            "ゼロから考",
            "確認されている",
            "分かっている",
            "言われている",
            "設定されている",
            "問題設定",
            "タスク設定",
            "課題設定",
            "メタ的",
            "メタ情報",
            "メタレベル",
            "実際には",
            "実際は",
            "前提として",
            "前提条件",
            "参考までに",
            "念のため",
            "言及はせず",
            "言及しない",
            "言及しません",
            "純粋に医学的",
            "医学的観点のみ",
            "医学的に純粋",
        ]

        # Compile regex patterns for efficient matching
        self.meta_patterns = [
            re.compile(rf".*{re.escape(keyword)}.*", re.IGNORECASE)
            for keyword in self.meta_keywords
        ]

        # Additional patterns for specific phrases
        self.specific_patterns = [
            re.compile(r".*今回は.*エラー.*例.*", re.IGNORECASE),
            re.compile(r".*その言及はせず.*", re.IGNORECASE),
            re.compile(r".*reasoning.*言及.*", re.IGNORECASE),
            re.compile(r".*事前.*確認.*問題なさそう.*", re.IGNORECASE),
            re.compile(r".*既に.*分かっている.*", re.IGNORECASE),
            re.compile(r".*設定上.*", re.IGNORECASE),
            re.compile(r".*課題として.*", re.IGNORECASE),
        ]

        self.all_patterns = self.meta_patterns + self.specific_patterns

    def find_sentences_to_remove(self, text: str) -> List[tuple]:
        """
        Find sentences that contain meta-references and their positions.

        Args:
            text: Text to analyze

        Returns:
            List of (start_pos, end_pos) tuples for sentences to remove
        """
        if not text:
            return []

        sentences_to_remove = []

        # Find all sentences ending with period
        sentence_pattern = re.compile(r"[^。]*。")

        for match in sentence_pattern.finditer(text):
            sentence = match.group()
            sentence_stripped = sentence.strip()

            if not sentence_stripped:
                continue

            # Check if sentence contains meta-references
            should_remove = False
            for pattern in self.all_patterns:
                if pattern.search(sentence_stripped):
                    should_remove = True
                    break

            if should_remove:
                sentences_to_remove.append((match.start(), match.end()))
                logger.debug(f"Found sentence to remove: {sentence_stripped[:50]}...")

        return sentences_to_remove

    def clean_reasoning_text(self, text: str) -> tuple[str, int]:
        """
        Clean reasoning text by removing meta-reference sentences while preserving formatting.

        Args:
            text: Original reasoning text

        Returns:
            Tuple of (cleaned_text, num_removed_sentences)
        """
        if not text:
            return text, 0

        # Find all sentences to remove
        sentences_to_remove = self.find_sentences_to_remove(text)

        if not sentences_to_remove:
            return text, 0

        # Sort by position (descending) to remove from end to beginning
        sentences_to_remove.sort(key=lambda x: x[0], reverse=True)

        # Remove sentences while preserving all other formatting
        cleaned_text = text
        removed_count = 0

        for start_pos, end_pos in sentences_to_remove:
            # Remove the sentence while preserving surrounding whitespace and newlines
            cleaned_text = cleaned_text[:start_pos] + cleaned_text[end_pos:]
            removed_count += 1

        return cleaned_text, removed_count

    def clean_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a single interaction."""
        cleaned_interaction = interaction.copy()

        if "raw_response" in interaction and isinstance(
            interaction["raw_response"], dict
        ):
            raw_response = interaction["raw_response"].copy()

            if "reasoning" in raw_response and raw_response["reasoning"]:
                original_reasoning = raw_response["reasoning"]
                cleaned_reasoning, removed_count = self.clean_reasoning_text(
                    original_reasoning
                )

                raw_response["reasoning"] = cleaned_reasoning

                # Add cleaning metadata
                if "metadata" not in raw_response:
                    raw_response["metadata"] = {}
                raw_response["metadata"]["reasoning_cleaned"] = True
                raw_response["metadata"]["removed_sentences"] = removed_count
                raw_response["metadata"]["original_reasoning_length"] = len(
                    original_reasoning
                )
                raw_response["metadata"]["cleaned_reasoning_length"] = len(
                    cleaned_reasoning
                )

                logger.debug(
                    f"Cleaned reasoning for sample {interaction.get('sample_id', 'unknown')}: "
                    f"removed {removed_count} sentences"
                )

            cleaned_interaction["raw_response"] = raw_response

        return cleaned_interaction


def detect_file_format(data: Dict[str, Any]) -> str:
    """
    Detect the format of the JSON file.

    Returns:
        'interactions': File with 'interactions' array (raw_responses.json format)
        'samples': File with direct sample array (samples.json format)
        'unknown': Unknown format
    """
    if "interactions" in data and isinstance(data["interactions"], list):
        return "interactions"
    elif isinstance(data, list) and len(data) > 0 and "raw_response" in data[0]:
        return "samples"
    else:
        return "unknown"


def clean_samples_file(input_path: Path, output_path: Path = None) -> None:
    """Clean a samples JSON file (direct array format)."""
    if output_path is None:
        output_path = input_path.parent / f"cleaned_{input_path.name}"

    logger.info(f"Processing samples file {input_path}")

    try:
        data = load_json(input_path)
    except Exception as e:
        logger.error(f"Error loading {input_path}: {e}")
        return

    if not isinstance(data, list):
        logger.warning(f"Expected array format in {input_path}")
        return

    cleaner = ReasoningCleaner()
    cleaned_data = []

    total_removed = 0
    processed_samples = 0

    for sample in data:
        if "raw_response" in sample:
            # Create a mock interaction for processing
            mock_interaction = {"raw_response": sample["raw_response"]}
            cleaned_interaction = cleaner.clean_interaction(mock_interaction)

            # Copy original sample and update raw_response
            cleaned_sample = sample.copy()
            cleaned_sample["raw_response"] = cleaned_interaction["raw_response"]
            cleaned_data.append(cleaned_sample)

            # Count removed sentences
            if (
                "metadata" in cleaned_interaction["raw_response"]
                and "removed_sentences"
                in cleaned_interaction["raw_response"]["metadata"]
            ):
                total_removed += cleaned_interaction["raw_response"]["metadata"][
                    "removed_sentences"
                ]
                processed_samples += 1
        else:
            # Keep samples without raw_response as-is
            cleaned_data.append(sample)

    save_json(cleaned_data, output_path)

    logger.info(f"[OK] Cleaned {input_path}")
    logger.info(f"   📤 Output: {output_path}")
    logger.info(
        f"   🧹 Removed {total_removed} sentences from {processed_samples} samples"
    )


def clean_raw_responses_file(input_path: Path, output_path: Path = None) -> None:
    """Clean a single JSON file (auto-detect format)."""
    if output_path is None:
        output_path = input_path.parent / f"cleaned_{input_path.name}"

    logger.info(f"Processing {input_path}")

    try:
        data = load_json(input_path)
    except Exception as e:
        logger.error(f"Error loading {input_path}: {e}")
        return

    # Detect file format
    file_format = detect_file_format(data)

    if file_format == "interactions":
        # Original format with interactions array
        if "interactions" not in data:
            logger.warning(f"No 'interactions' found in {input_path}")
            return

        cleaner = ReasoningCleaner()
        cleaned_data = data.copy()

        total_removed = 0
        processed_interactions = 0

        cleaned_interactions = []
        for interaction in data["interactions"]:
            cleaned_interaction = cleaner.clean_interaction(interaction)
            cleaned_interactions.append(cleaned_interaction)

            # Count removed sentences
            if (
                "raw_response" in cleaned_interaction
                and "metadata" in cleaned_interaction["raw_response"]
                and "removed_sentences"
                in cleaned_interaction["raw_response"]["metadata"]
            ):
                total_removed += cleaned_interaction["raw_response"]["metadata"][
                    "removed_sentences"
                ]
                processed_interactions += 1

        cleaned_data["interactions"] = cleaned_interactions

        # Add cleaning metadata to the file
        if "metadata" not in cleaned_data:
            cleaned_data["metadata"] = {}
        cleaned_data["metadata"]["reasoning_cleaned"] = True
        cleaned_data["metadata"]["total_removed_sentences"] = total_removed
        cleaned_data["metadata"]["processed_interactions"] = processed_interactions

        save_json(cleaned_data, output_path)

        logger.info(f"[OK] Cleaned {input_path}")
        logger.info(f"   📤 Output: {output_path}")
        logger.info(
            f"   🧹 Removed {total_removed} sentences from {processed_interactions} interactions"
        )

    elif file_format == "samples":
        # Direct samples array format
        clean_samples_file(input_path, output_path)

    else:
        logger.error(f"Unknown file format in {input_path}")
        logger.error("Expected either 'interactions' array or direct samples array")


def main():
    parser = argparse.ArgumentParser(
        description="Clean reasoning responses by removing meta-references"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Input raw_responses.json file path")
    group.add_argument(
        "--input_dir",
        type=str,
        help="Input directory to search for raw_responses.json files",
    )

    parser.add_argument(
        "--output", type=str, help="Output file path (only for --input mode)"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search recursively in input directory"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files instead of creating cleaned_ versions",
    )

    args = parser.parse_args()

    if args.input:
        # Single file mode
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1

        output_path = Path(args.output) if args.output else None
        if args.overwrite:
            output_path = input_path

        clean_raw_responses_file(input_path, output_path)

    else:
        # Directory mode
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 1

        # Find all raw_responses.json files
        if args.recursive:
            json_files = list(input_dir.rglob("raw_responses.json"))
        else:
            json_files = list(input_dir.glob("raw_responses.json"))

        if not json_files:
            logger.warning(f"No raw_responses.json files found in {input_dir}")
            return 1

        logger.info(f"Found {len(json_files)} raw_responses.json files")

        for json_file in json_files:
            output_path = None
            if args.overwrite:
                output_path = json_file

            clean_raw_responses_file(json_file, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
